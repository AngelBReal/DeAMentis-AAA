# Proyecto De A Mentis  
**Detector Educativo y Contextualizador de Noticias**

![Project Status](https://img.shields.io/badge/status-en%20desarrollo-yellow)
![DVC Tracked](https://img.shields.io/badge/DVC-enabled-blue)
![MLflow Tracking](https://img.shields.io/badge/MLflow-integrated-green)
![License](https://img.shields.io/badge/license-Pendiente-lightgrey)
![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)

---

## Descripción general

**De A Mentis** es una herramienta orientada a combatir la desinformación en entornos digitales mediante un enfoque educativo. No se limita a etiquetar una noticia como verdadera o falsa, sino que busca fortalecer la alfabetización informacional a través de:

- Detección de señales de alerta en el lenguaje (como cifras sin fuente, lenguaje sensacionalista o citas ambiguas).
- Generación de advertencias interpretables para el usuario.
- Clasificación de noticias según su nivel de veracidad.

---

## Problema que resuelve

En un entorno saturado de información, el reto no es únicamente identificar contenido falso, sino dotar a la ciudadanía de herramientas para evaluar de forma crítica el contenido que consume. Este proyecto se alinea con objetivos de educación cívica y alfabetización digital, con potencial de ser utilizado por instituciones educativas, organizaciones civiles y medios de comunicación.

---

## Flujo de experimentación

### Primera etapa – Comparación de modelos clásicos
- Modelos evaluados: Random Forest, SVM, Logistic Regression, AdaBoost.
- Representaciones de texto usadas: unigramas y bigramas, con y sin stemming y lematización.

### Segunda etapa – Selección del mejor dataset
- Experimento: `01 Best Dataset – De A Mentis`.
- Se probaron seis variantes de preprocesamiento combinadas con cuatro modelos clásicos (Naive Bayes, SVM, AdaBoost, Random Forest).
- Resultados: SVM y modelos Boosting fueron los más prometedores.

### Tercera etapa – Calibración y ensambles
- Experimento: `01 Best Dataset Calibrado`.
  - XGBoost alcanzó un F1 score de 0.74.
- Experimento: `01 Best Dataset Calibrado V2`.
  - Voting ensemble con SVM + XGBoost obtuvo:
    - Accuracy: 0.731
    - AUC: 0.794
    - F1: 0.775
    - Precision: 0.796
    - Recall: 0.756

### Cuarta etapa – Optimización con Optuna
- Mejores hiperparámetros encontrados:
  - svm_C: 1.287
  - xgb_lr: 0.205
  - xgb_estimators: 149
- Métricas finales:
  - F1 score: 0.783
  - AUC: 0.795

### Quinta etapa – Pruebas con Transformers
- Se evaluaron modelos basados en BERT.
- Los resultados no superaron los obtenidos por el Voting Ensemble.

Todos los experimentos están registrados y visualizables en la interfaz MLflow a través de DagsHub.

---

## Seguimiento de experimentos

- MLflow para seguimiento de métricas, artefactos y parámetros.
- Conexión a DagsHub para versionado y visualización.
- El modelo final se encuentra registrado como modelo Champion.
- Notebooks reproducibles conectados al tracking local (`mlruns/`).

---

## Estructura del repositorio

```

app/
├── backend/
│   ├── app/
│   ├── final\_model/
│   ├── Dockerfile
│   ├── start.sh
│   └── requirements.txt
├── frontend/
│   ├── public/
│   └── src/
data/
├── raw/
├── interim/
├── processed/
src/
├── data\_download/
├── data\_preprocess/
├── models/
│   ├── final\_models/
│   └── notebooks/
├── cli.py
mlruns/
dvc.yaml
requirements.txt

````

---

## Arquitectura del proyecto y automatización

Este repositorio sigue una arquitectura modular y reproducible para el desarrollo de sistemas de machine learning. Utiliza `Typer` como CLI, `DVC` como orquestador de pipelines y `MLflow` para rastreo de experimentos.

### Flujo general automatizado

```text
Descarga → Preprocesamiento → Entrenamiento del modelo final
````

### Scripts principales

#### `src/data_download/`

* `download.py`: Descarga dos datasets (`omdena` y `FakeNewsCorpusSpanish`).
* `process.py`: Une ambos datasets en uno solo.
* `utils.py`: Funciones de utilidad generales.

#### `src/data_preprocess/`

* `preprocessing.py`: Aplica limpieza de texto, TF-IDF y selección de características.

#### `src/models/final_models/`

* `build_final_model.py`: Entrena y guarda el modelo final y lo registra en MLflow.

#### `src/models/notebooks/`

* Notebooks utilizados en Google Colab para experimentación y visualización de métricas.

---

### CLI (`src/cli.py`)

El archivo `cli.py` permite ejecutar cada etapa del pipeline desde línea de comandos usando Typer:

| Comando          | Acción                                                          |
| ---------------- | --------------------------------------------------------------- |
| `download-all`   | Descarga ambos datasets.                                        |
| `process-all`    | Une y limpia los datasets descargados.                          |
| `preprocess-all` | Aplica el pipeline de preprocesamiento textual.                 |
| `build-model`    | Entrena y guarda el modelo final en `app/backend/final_model/`. |

---

### Pipeline en `dvc.yaml`

```yaml
stages:
  raw_pipeline:
    cmd: set PYTHONPATH=src && python src/cli.py download-all && python src/cli.py process-all
    deps:
      - src/cli.py
    outs:
      - data/interim/combined_fakenews_dataset.csv
      - data/raw

  preprocess_pipeline:
    cmd: python src/cli.py preprocess-all
    deps:
      - data/interim/combined_fakenews_dataset.csv
      - src/cli.py
      - src/data_preprocess/preprocessing.py
    outs:
      - data/processed

  train_model_pipeline:
    cmd: python src/cli.py build-model
    deps:
      - data/interim/combined_fakenews_dataset.csv
      - src/cli.py
      - src/models/final_models/build_final_model.py
    outs:
      - app/backend/final_model/final_voting_model.pkl
      - app/backend/final_model/artifacts/tfidf_vectorizer.pkl
      - app/backend/final_model/artifacts/selector.pkl
```

Ejecuta el pipeline completo con:

```bash
dvc repro
```

---

## Estado actual

* Modelo Voting Ensemble entrenado y optimizado
* Flujo de trabajo reproducible con DVC y MLflow
* Back-end (FastAPI) y front-end (React) funcionales en entorno local
* Conexión establecida con DagsHub para trazabilidad de experimentos

---

## Próximos pasos

* Despliegue de la aplicación web en un entorno público (ej. Render)
* Mejora de la interfaz educativa con explicación de resultados
* Pruebas piloto con usuarios reales para evaluación participativa
* Publicación de guía de uso para entornos educativos

---

## Autor

**Angel Barraza**
Maestría en Ciencia de Datos – UNISON
Contacto: angelbarrazareal@gmail.com

---

## Licencia

Pendiente de definir

