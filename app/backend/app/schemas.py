from pydantic import BaseModel
from typing import List

class NewsInput(BaseModel):
    title: str
    body: str

class PredictionResponse(BaseModel):
    prediction: str
    alerts: List[str]
