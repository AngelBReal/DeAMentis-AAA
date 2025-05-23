# Despliegue en Render

Este proyecto tiene:
- Backend: FastAPI
- Frontend: React
- Configurado con `render.yaml`

## Estructura relevante

app/
├── backend/
│ └── app/
│ └── main.py
├── frontend/
│ └── package.json
render.yaml

markdown
Copy
Edit

## Despliegue

- Subida de modelos: `.pkl` en carpeta `artifacts/`
- MLflow local y remoto
- Conexión por CORS desde React a FastAPI