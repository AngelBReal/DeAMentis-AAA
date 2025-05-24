# Despliegue de la Aplicación

La aplicación **De A Mentis** se encuentra desplegada utilizando la plataforma **Render.com**, permitiendo acceder al servicio desde cualquier navegador.

🔗 **App en producción**: [https://deamentis-frontend.onrender.com](https://deamentis-frontend.onrender.com)

---

## Arquitectura del Proyecto

```text
           ┌───────────────┐           ┌─────────────┐
           │   Frontend    │ ───────▶  │  Usuario    │
           │   (React)     │           └─────────────┘
           └──────┬────────┘
                  │
     https://deamentis-frontend.onrender.com
                  │
           ┌──────▼──────┐
           │  Backend    │
           │ (FastAPI)   │
           └──────┬──────┘
                  │
                  ▼
    ┌─────────────────────────────┐
    │ Módulo ML (VotingClassifier)│
    └─────────────────────────────┘
                  │
    ┌─────────────────────────────┐
    │ Módulo de Análisis Lingüístico │
    └─────────────────────────────┘
````

---

## Tecnologías de Despliegue

| Componente      | Tecnología                                                    |
| --------------- | ------------------------------------------------------------- |
| Frontend        | React + Bootstrap                                             |
| Backend         | FastAPI                                                       |
| Modelo          | VotingClassifier (SVM + XGBoost) entrenado y serializado      |
| Infraestructura | Render.com (frontend + backend como servicios independientes) |
| Configuración   | `render.yaml` para CI/CD                                      |
| Seguimiento     | MLflow + DVC + DagsHub                                        |

---

## Backend: FastAPI

El backend expone una **API REST** que recibe un texto de noticia y realiza dos procesos separados:

### 1. Clasificación automática

* Mediante un modelo VotingClassifier (`SVM + XGBoost`) entrenado con TF-IDF y calibrado con Optuna.
* Devuelve la **probabilidad de falsedad (`prob_fake`)**, sin clasificar binariamente.

### 2. Análisis lingüístico interpretativo

* Un módulo externo analiza el texto y detecta patrones de riesgo como:

  * **Cifras sin fuente**
  * **Citas ambiguas**
  * **Lenguaje sensacionalista**
* Estas alertas son **reglas definidas manualmente** y no son generadas por el modelo de machine learning.

### Ruta principal

```bash
POST /predict
```

### Ejemplo de respuesta

```json
{
  "prob_fake": 0.82,
  "riesgo": "alto",
  "alertas": [
    "Contiene cifras sin fuente",
    "Lenguaje sensacionalista"
  ]
}
```

---

## Frontend: React

El frontend fue desarrollado con **React** y estilizado con **Bootstrap**. Está diseñado para ser claro, accesible y educativo.

### Funcionalidades

* Formulario para ingresar noticia
* Indicador visual del nivel de riesgo
* Lista de advertencias generadas por el analizador lingüístico
* Indicaciones al usuario sobre cómo interpretar los resultados

---

## Archivo `render.yaml`

Para automatizar el despliegue continuo, se utilizó un archivo `render.yaml` que define dos servicios:

```yaml
services:
  - type: web
    name: deamentis-backend
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn main:app --host 0.0.0.0 --port 10000
    envVars:
      - key: MLFLOW_TRACKING_URI
        value: https://dagshub.com/AngelBReal/DeAMentis-AAA.mlflow

  - type: web
    name: deamentis-frontend
    env: static
    staticPublishPath: ./build
    buildCommand: npm install && npm run build
```

---

## Monitoreo Post-Despliegue

Cada predicción en producción (inferencia desde el frontend) se registra como un **nuevo run en MLflow** bajo el experimento `100 Noticias - Inference Log`, permitiendo:

* Auditar predicciones en tiempo real
* Identificar patrones sospechosos
* Validar estabilidad del modelo en producción

---

## Consideraciones

* Las advertencias provienen de un módulo de reglas lingüísticas, no del modelo.
* El sistema está pensado para fomentar pensamiento crítico, no como herramienta de censura.
* En producción real, se podría integrar:

  * Autenticación JWT
  * Logs remotos
  * Monitorización con Prometheus/Sentry

---

## 📂 Estructura relevante del proyecto

```text
app/
├── backend/
│   ├── main.py               # FastAPI app
│   ├── final_model/          # Modelos y artefactos
│   └── analyzer.py           # Analizador lingüístico (alertas educativas)
├── frontend/
│   └── src/                  # React components
render.yaml                   # Configuración de despliegue Render
```

---
