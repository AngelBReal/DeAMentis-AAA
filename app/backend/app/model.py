import mlflow
import mlflow.sklearn
import joblib
import re
import os
import spacy
from pathlib import Path
from typing import Tuple

# === 1. Configurar MLflow (DagsHub como servidor de tracking) ===
mlflow.set_tracking_uri("https://dagshub.com/AngelBReal/DeAMentis-AAA.mlflow")
MODEL_NAME = "voting_classifier_produccion"
MODEL_ALIAS = "champion"

# === 2. Cargar modelo desde MLflow Model Registry ===
MODEL = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")

# === 3. Cargar artefactos locales ===
ARTIFACTS_PATH = Path(__file__).resolve().parent / "../final_model/artifacts"
VECTORIZER = joblib.load(ARTIFACTS_PATH / "tfidf_vectorizer.pkl")
SELECTOR = joblib.load(ARTIFACTS_PATH / "selector.pkl")
ENCODER = joblib.load(ARTIFACTS_PATH / "label_encoder.pkl")

# === 4. Cargar modelo de lenguaje para lematización ===
try:
    nlp = spacy.load("es_core_news_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "es_core_news_sm"], check=True)
    nlp = spacy.load("es_core_news_sm")

# === 5. Función de preprocesamiento con lematización ===
def lemmatize_clean(text: str) -> str:
    doc = nlp(text.lower())
    return " ".join([t.lemma_ for t in doc if not t.is_stop and not t.is_punct and not t.is_space])

# === 6. Función principal de predicción ===
def predict_news(title: str, body: str) -> Tuple[str, int, float]:
    full_text = title + " " + body
    clean_text = lemmatize_clean(full_text)

    # Vectorización
    X_tfidf = VECTORIZER.transform([clean_text])
    X_selected = SELECTOR.transform(X_tfidf)

    # Predicción
    raw_pred = MODEL.predict(X_selected)[0]
    label_pred = ENCODER.inverse_transform([raw_pred])[0]
    prediction = "NOTICIA FALSA" if label_pred.lower() == "fake" else "NOTICIA VERDADERA"

    # Probabilidad clase "fake"
    proba = MODEL.predict_proba(X_selected)[0]
    class_index = list(ENCODER.classes_).index("fake")
    prob_fake = float(proba[class_index])

    return prediction, int(raw_pred), prob_fake
