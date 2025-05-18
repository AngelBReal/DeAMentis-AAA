import os
import pandas as pd
import joblib
import spacy
import nltk
from pathlib import Path
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE

import mlflow
from mlflow.tracking import MlflowClient

# ============ CONFIGURACIÓN DE ENTORNO ============
load_dotenv()  # Cargar variables desde .env

mlflow.set_tracking_uri("https://dagshub.com/AngelBReal/DeAMentis-AAA.mlflow")
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")
mlflow.set_experiment("Modelo Producción VotingClassifier")

# ============ RUTAS ============
DATA_PATH = Path("data/interim/combined_fakenews_dataset.csv")
SAVE_DIR = Path("models/final_model")
SAVE_DIR_ART = SAVE_DIR / "artifacts"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
SAVE_DIR_ART.mkdir(parents=True, exist_ok=True)

# ============ NLP Y LEMATIZACIÓN ============
nltk.download('stopwords')
from nltk.corpus import stopwords
spanish_stopwords = stopwords.words('spanish')

try:
    nlp = spacy.load("es_core_news_sm")
except OSError:
    print("Descargando modelo spaCy...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "es_core_news_sm"], check=True)
    nlp = spacy.load("es_core_news_sm")

def lemmatize_clean(text):
    doc = nlp(text.lower())
    return " ".join([t.lemma_ for t in doc if not t.is_stop and not t.is_punct and not t.is_space])

# ============ 1. CARGA Y LIMPIEZA DE DATOS ============
print("🔹 Cargando datos...")
df = pd.read_csv(DATA_PATH)
df = df.rename(columns={"content": "text"}) if "content" in df.columns else df
df['label'] = df['label'].replace({'false': 'fake'})
df = df[df['label'].isin(['true', 'fake'])]
df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")
df["text_clean"] = df["text"].apply(lemmatize_clean)

X_raw = df["text_clean"]
y = df["label"]

# ============ 2. DIVISIÓN TRAIN / TEST ============
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, stratify=y, random_state=42
)

# ============ 3. VECTORIZACIÓN Y SELECCIÓN ============
tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=6336)
X_train_tfidf = tfidf.fit_transform(X_train_raw)
X_test_tfidf = tfidf.transform(X_test_raw)

selector = SelectKBest(score_func=chi2, k=3000)
X_train_sel = selector.fit_transform(X_train_tfidf, y_train)
X_test_sel = selector.transform(X_test_tfidf)

# ============ 4. BALANCEO Y CODIFICACIÓN ============
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_sel, y_train)

encoder = LabelEncoder()
y_train_enc = encoder.fit_transform(y_train_bal)
y_test_enc = encoder.transform(y_test)

# ============ 5. MODELO FINAL ============
svm_C = 1.2873664552398318
xgb_lr = 0.2049193304913042
xgb_estimators = 149

svm = CalibratedClassifierCV(SVC(kernel='linear', C=svm_C, probability=True), cv=3)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                    learning_rate=xgb_lr, n_estimators=xgb_estimators)

ensemble = VotingClassifier(estimators=[("svm", svm), ("xgb", xgb)], voting='soft')

print("🔹 Entrenando modelo ensemble...")
ensemble.fit(X_train_bal, y_train_enc)

# ============ 6. EVALUACIÓN ============
y_pred = ensemble.predict(X_test_sel)
y_proba = ensemble.predict_proba(X_test_sel)[:, 1]

f1 = f1_score(y_test_enc, y_pred)
auc = roc_auc_score(y_test_enc, y_proba)

print(f"✅ Modelo entrenado. F1: {f1:.4f} | AUC: {auc:.4f}")

# ============ 7. GUARDADO LOCAL ============
print("💾 Guardando artefactos...")
joblib.dump(ensemble, SAVE_DIR / "final_voting_model.pkl")
joblib.dump(tfidf, SAVE_DIR_ART / "tfidf_vectorizer.pkl")
joblib.dump(selector, SAVE_DIR_ART / "selector.pkl")
joblib.dump(encoder, SAVE_DIR_ART / "label_encoder.pkl")

# ============ 8. MLFLOW + REGISTRY ============
with mlflow.start_run(run_name="final_model_voting") as run:
    mlflow.log_params({
        "svm_C": svm_C,
        "xgb_lr": xgb_lr,
        "xgb_estimators": xgb_estimators
    })
    mlflow.log_metrics({
        "f1_score": f1,
        "auc": auc
    })
    mlflow.set_tag("type", "production")
    mlflow.set_tag("pipeline", "TF-IDF + SelectKBest")

    mlflow.sklearn.log_model(ensemble, artifact_path="final_voting_model")

    model_uri = f"runs:/{run.info.run_id}/final_voting_model"
    result = mlflow.register_model(model_uri=model_uri, name="voting_classifier_produccion")

    print(f"📌 Modelo registrado como 'voting_classifier_produccion' (versión {result.version})")

print("🚀 Entrenamiento, guardado y registro completo.")

# CLI-compatible
def main():
    pass

if __name__ == "__main__":
    main()
