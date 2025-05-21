# Imagen base ligera de Python
FROM python:3.10-slim

# Establecer el directorio de trabajo en el contenedor
WORKDIR /app

# Copiar todo el contenido del backend al contenedor
COPY . /app

# Instalar spaCy y las dependencias
RUN apt-get update && apt-get install -y build-essential \
    && pip install --no-cache-dir -r requirements.txt \
    && python -m spacy download es_core_news_md

# Comando para arrancar la app con Uvicorn
CMD ["./start.sh"]
