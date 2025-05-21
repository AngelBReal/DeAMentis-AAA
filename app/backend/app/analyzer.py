import spacy
import re

# Cargar modelo de lenguaje de spaCy para español
nlp = spacy.load("es_core_news_md")

def analyze_text(text):
    alerts = []
    doc = nlp(text)

    # Frases ambiguas comunes
    suspicious_phrases = ["según fuentes", "dicen que", "se rumora", "al parecer"]
    for phrase in suspicious_phrases:
        if phrase in text.lower():
            alerts.append(f"Frase ambigua detectada: '{phrase}'")

    # Números relevantes
    if re.search(r"\d+\s*(%|pesos|millones|mil)", text.lower()):
        alerts.append("Se detectaron cifras que podrían requerir verificación.")
    
    return alerts
