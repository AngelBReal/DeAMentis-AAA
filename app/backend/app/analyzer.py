import spacy
import re

nlp = spacy.load("es_core_news_md")

def analyze_text(text):
    alerts = []
    doc = nlp(text)

    suspicious_phrases = ["según fuentes", "dicen que", "se rumora", "al parecer"]
    for phrase in suspicious_phrases:
        if phrase in text.lower():
            alerts.append(f"Frase ambigua detectada: '{phrase}'")

    if re.search(r"\d+\s*(%|pesos|millones|mil)", text.lower()):
        alerts.append("Se detectaron cifras que podrían requerir verificación.")
    
    return alerts
