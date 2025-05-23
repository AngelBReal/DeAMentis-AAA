from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # <-- IMPORTANTE

from app.schemas import NewsInput, PredictionResponse
from app.model import predict_news
from app.analyzer import analyze_text
from app.logging import log_inference_to_mlflow

app = FastAPI()

# === HABILITAR CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # más flexible pero menos seguro  ⚠️ permite solo tu frontend local
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict", response_model=PredictionResponse)
def predict(news: NewsInput):
    raw_alerts = analyze_text(news.title + " " + news.body)
    # Asegurar que todos los elementos sean strings planos
    alerts = []
    for alert in raw_alerts:
        if isinstance(alert, list):
            alerts.extend(map(str, alert))
        else:
            alerts.append(str(alert))
    prediction_label, prediction_raw, prob_fake = predict_news(news.title, news.body)

    # Determinar nivel de riesgo
    if prob_fake < 0.4:
        alerts.insert(0, "🟢 BAJO RIESGO: Parece confiable, pero aún así usa juicio crítico.")
        risk_level = "low"
    elif prob_fake < 0.7:
        alerts.insert(0, "🟡 RIESGO MODERADO: Verifica fuentes y detalles antes de compartir.")
        risk_level = "moderate"
    else:
        alerts.insert(0, "🔴 ALTO RIESGO: Posible contenido falso o engañoso. Verifica cuidadosamente.")
        risk_level = "high"

    # Logging en MLflow
    log_inference_to_mlflow(
        news.title, news.body, prediction_label, alerts,
        prediction_raw, prob_fake
    )

    return {
        "prediction": "",  # vacío para frontend
        "alerts": alerts,
        "risk_level": risk_level
    }