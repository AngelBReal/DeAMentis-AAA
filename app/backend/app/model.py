import mlflow
import mlflow.pyfunc
import joblib
import re
import os
import pandas as pd
from datetime import datetime

# === 1. Configurar MLflow (DagsHub como servidor de tracking) ===
mlflow.set_tracking_uri("https://dagshub.com/AngelBReal/DeAMentis-AAA.mlflow")
MODEL_NAME = "voting_classifier_produccion"
MODEL_ALIAS = "champion"

# === 2. Cargar modelo desde el MLflow Model Registry ===
MODEL = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")

# === 3. Cargar artefactos locales (TF-IDF, selector) ===
ARTIFACTS_PATH = os.path.join(os.path.dirname(__file__), "../final_model/artifacts")
VECTORIZER = joblib.load(os.path.join(ARTIFACTS_PATH, "tfidf_vectorizer.pkl"))
SELECTOR = joblib.load(os.path.join(ARTIFACTS_PATH, "selector.pkl"))

# === 3. Preprocesamiento textual ===
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-ZáéíóúüñÁÉÍÓÚÜÑ\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# === 4. Predicción ===
def predict_news(title: str, body: str) -> str:
    full_text = title + " " + body
    clean_text = preprocess_text(full_text)

    X_tfidf = VECTORIZER.transform([clean_text])
    X_selected = SELECTOR.transform(X_tfidf)

    prediction = MODEL.predict(X_selected)[0]
    return "NOTICIA FALSA" if prediction == 1 else "NOTICIA VERDADERA"