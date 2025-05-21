from fastapi import FastAPI
from app.schemas import NewsInput, PredictionResponse
from app.model import predict_news
from app.analyzer import analyze_text
from app.logging import log_inference_to_mlflow


app = FastAPI()

@app.post("/predict", response_model=PredictionResponse)
def predict(news: NewsInput):
    alerts = analyze_text(news.title + " " + news.body)
    prediction = predict_news(news.title, news.body)

    # Registrar en MLflow
    log_inference_to_mlflow(news.title, news.body, prediction, alerts)

    return {
        "prediction": prediction,
        "alerts": alerts
    }