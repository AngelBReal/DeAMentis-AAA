from pydantic import BaseModel
from typing import List

class NewsInput(BaseModel):
    title: str
    body: str

class PredictionResponse(BaseModel):
    prediction: str  # sigue existiendo por compatibilidad
    alerts: List[str]
    risk_level: str  # nuevo campo: "low", "moderate", "high"


