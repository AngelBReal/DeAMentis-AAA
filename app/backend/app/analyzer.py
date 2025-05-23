import re
import spacy

# Cargar modelo de lenguaje spaCy en español
nlp = spacy.load("es_core_news_md")

# === 1. Patrones definidos manualmente ===
# Formato: (regex, mensaje, tipo_de_manipulación)
FACTCHECK_PATTERNS = [
    # Ambigüedad
    (r"\b(según fuentes|dicen que|se rumora|al parecer|testigos afirman|se dice que)\b",
     "Frase ambigua o sin fuente clara.", "misinformation"),

    # Cifras sin contexto
    (r"\d+\s*(%|por ciento|pesos|euros|dólares|millones|mil|personas|casos|muertes)",
     "Se detectaron cifras que podrían requerir verificación.", "misinformation"),

    # Sensacionalismo
    (r"\b(¡|increíble|impresionante|escándalo|alarmante|urgente|atención)\b",
     "Término sensacionalista detectado.", "disinformation"),

    # Polarización emocional
    (r"\b(traición|enemigo del pueblo|mentira absoluta|conspiración|dictadura|imperdonable)\b",
     "Lenguaje emocional o polarizante detectado.", "malinformation"),

    # Fuente vaga
    (r"\b(expertos|científicos|analistas|estudios|investigadores|autoridades)\b",
     "Mención de fuente sin especificar.", "misinformation"),

    # Llamados virales
    (r"\b(comparte|difunde|no lo verás en los medios|ellos no quieren que sepas esto)\b",
     "Llamado viral sin evidencia clara.", "disinformation"),
]

# === 2. NER: entidades sin contexto adecuado ===
def analyze_entities(doc):
    alerts = []
    for ent in doc.ents:
        if ent.label_ in ["ORG", "GPE", "PERSON"] and len(ent.text.split()) <= 1:
            alerts.append(f"Mención vaga o genérica detectada: '{ent.text}' ({ent.label_})")
    return alerts

# === 3. Coherencia de contexto numérico ===
def analyze_numerical_context(doc):
    alerts = []
    has_year = any(ent.label_ == "DATE" and re.match(r"\b20\d{2}\b", ent.text) for ent in doc.ents)
    has_number = any(ent.label_ in ["CARDINAL", "QUANTITY", "PERCENT"] for ent in doc.ents)
    has_place = any(ent.label_ == "GPE" for ent in doc.ents)

    if has_number and not has_year:
        alerts.append("Se mencionan cifras pero sin marco temporal explícito.")
    if has_number and not has_place:
        alerts.append("Se mencionan cifras pero sin ubicación geográfica asociada.")
    return alerts

# === 4. Contradicciones simples ===
CONTRADICTORY_PATTERNS = [
    (r"\bno\s+(hubo|existió|pasó)\b", r"\b(hubo|existió|pasó|ocurrió)\b"),
    (r"\b(nunca|jamás)\b", r"\b(sí|sí que|claro que)\b"),
]

def analyze_contradictions(text):
    alerts = []
    text_lower = text.lower()
    for neg, affirm in CONTRADICTORY_PATTERNS:
        if re.search(neg, text_lower) and re.search(affirm, text_lower):
            alerts.append("Posible contradicción detectada en el discurso.")
    return alerts

# === 5. Análisis principal ===
def analyze_text(text: str):
    alerts = []
    categories = {"misinformation": 0, "disinformation": 0, "malinformation": 0}
    text_lower = text.lower()
    doc = nlp(text_lower)

    # Reglas por patrones
    for pattern, message, category in FACTCHECK_PATTERNS:
        if re.search(pattern, text_lower):
            alerts.append(message)
            categories[category] += 1

    # Entidades nombradas vagas
    alerts += analyze_entities(doc)

    # Incoherencia de contexto
    alerts += analyze_numerical_context(doc)

    # Contradicciones internas
    alerts += analyze_contradictions(text)

    # Clasificación dominante
    total_flags = sum(categories.values())
    classification = (
        max(categories, key=categories.get) if total_flags > 0 else "neutral"
    )

    return alerts, classification
