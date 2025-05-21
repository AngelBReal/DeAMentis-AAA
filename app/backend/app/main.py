from fastapi import FastAPI
from app.schemas import NewsInput, PredictionResponse
from app.model import predict_news
from app.analyzer import analyze_text
from app.logging import log_inference_to_mlflow


app = FastAPI()

@app.post("/predict", response_model=PredictionResponse)
def predict(news: NewsInput):
    alerts = analyze_text(news.title + " " + news.body)
    prediction, prediction_raw, prob_fake = predict_news(news.title, news.body)
    log_inference_to_mlflow(news.title, news.body, prediction, alerts, prediction_raw, prob_fake)
    return {
        "prediction": prediction,
        "alerts": alerts
    }