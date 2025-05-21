import mlflow
import mlflow.pyfunc
import joblib
import re
import os
import pandas as pd

# === 1. Configurar MLflow (DagsHub como servidor de tracking) ===
mlflow.set_tracking_uri("https://dagshub.com/AngelBReal/DeAMentis-AAA.mlflow")
MODEL_NAME = "voting_classifier_produccion"
MODEL_STAGE = "Production"

# === 2. Cargar modelo desde el MLflow Model Registry ===
MODEL = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")

# === 3. Cargar artefactos locales (TF-IDF, selector) ===
ARTIFACTS_PATH = os.path.join(os.path.dirname(__file__), "../../models/artifacts")
VECTORIZER = joblib.load(os.path.join(ARTIFACTS_PATH, "tfidf_vectorizer.pkl"))
SELECTOR = joblib.load(os.path.join(ARTIFACTS_PATH, "selector.pkl"))

# === 4. Preprocesamiento idéntico al usado durante entrenamiento ===
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # eliminar URLs
    text = re.sub(r"[^a-zA-ZáéíóúüñÁÉÍÓÚÜÑ\s]", "", text)  # eliminar signos
    text = re.sub(r"\s+", " ", text).strip()  # espacios extra
    return text

# === 5. Función de predicción final ===
def predict_news(title: str, body: str) -> str:
    full_text = title + " " + body
    clean_text = preprocess_text(full_text)

    X_tfidf = VECTORIZER.transform([clean_text])
    X_selected = SELECTOR.transform(X_tfidf)

    prediction = MODEL.predict(X_selected)[0]
    return "NOTICIA FALSA" if prediction == 1 else "NOTICIA VERDADERA"
