from fastapi import FastAPI
from app.schemas import NewsInput, PredictionResponse
from app.model import predict_news
from app.analyzer import analyze_text

app = FastAPI()

@app.post("/predict", response_model=PredictionResponse)
def predict(news: NewsInput):
    alerts = analyze_text(news.title + " " + news.body)
    prediction = predict_news(news.title, news.body)
    return {"prediction": prediction, "alerts": alerts}
