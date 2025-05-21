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
import mlflow.sklearn  # importa esta versión explícitamente

MODEL = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
# === 3. Cargar artefactos locales (TF-IDF, selector) ===
ARTIFACTS_PATH = os.path.join(os.path.dirname(__file__), "../final_model/artifacts")
VECTORIZER = joblib.load(os.path.join(ARTIFACTS_PATH, "tfidf_vectorizer.pkl"))
SELECTOR = joblib.load(os.path.join(ARTIFACTS_PATH, "selector.pkl"))
ENCODER = joblib.load(os.path.join(ARTIFACTS_PATH, "label_encoder.pkl"))

# === 4. Preprocesamiento básico ===
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-ZáéíóúüñÁÉÍÓÚÜÑ\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# === 5. Función principal de predicción ===
def predict_news(title: str, body: str) -> tuple:
    full_text = title + " " + body
    clean_text = preprocess_text(full_text)

    # Vectorización
    X_tfidf = VECTORIZER.transform([clean_text])
    X_selected = SELECTOR.transform(X_tfidf)

    # Predicción cruda (0 o 1) y etiqueta ("fake"/"true")
    raw_pred = MODEL.predict(X_selected)[0]
    label_pred = ENCODER.inverse_transform([raw_pred])[0]

    # Etiqueta para mostrar
    prediction = "NOTICIA FALSA" if label_pred.lower() == "fake" else "NOTICIA VERDADERA"

    # Probabilidad de clase "fake"
    proba = MODEL.predict_proba(X_selected)[0]
    class_index = list(ENCODER.classes_).index("fake")
    prob_fake = float(proba[class_index])

    return prediction, int(raw_pred), prob_fake
