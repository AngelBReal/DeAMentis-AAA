
# Proyecto **De A Mentis**  
**Detector Educativo y Contextualizador de Noticias**

![Project Status](https://img.shields.io/badge/status-en%20desarrollo-yellow)
![DVC Tracked](https://img.shields.io/badge/DVC-enabled-blue)
![MLflow Tracking](https://img.shields.io/badge/MLflow-integrated-green)
![License](https://img.shields.io/badge/license-Pendiente-lightgrey)
![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)

---

## Descripción General

**De A Mentis** es una herramienta desarrollada para combatir la desinformación en entornos digitales, con un enfoque educativo e interpretativo. Más allá de clasificar una noticia como *falsa* o *verdadera*, la aplicación busca fortalecer la alfabetización informacional del usuario.

### Funcionalidades clave:

- Detección de señales lingüísticas problemáticas (cifras sin fuente, lenguaje sensacionalista, citas ambiguas).
- Generación de advertencias interpretables que fomenten el pensamiento crítico.
- Clasificación automática del contenido por su nivel de veracidad.

---

## Problema que Resuelve

El desafío no radica únicamente en detectar contenido falso, sino en brindar herramientas para que cualquier persona pueda analizar críticamente la información. Esta solución se alinea con fines educativos y cívicos, con aplicaciones potenciales en:

- Instituciones educativas  
- Organizaciones civiles  
- Medios de comunicación

---

## TL;DR – Modelado y Backend

El modelo usado es un **VotingClassifier** que combina `SVM` y `XGBoost`, entrenado con:

- **Preprocesamiento**: limpieza textual, lematización, vectorización con **TF-IDF**, selección de características.
- **Balanceo de clases**: mediante **SMOTE**.
- **Seguimiento de experimentos**: usando **MLflow**, con registro y trazabilidad completos mediante **DagsHub**.

> La salida del modelo (`prob_fake`) se interpreta a través de **umbrales de riesgo** para mostrar advertencias y nivel de veracidad al usuario (sin clasificaciones binarias directas).
---

## Seguimiento y Trazabilidad

- **MLflow** para métricas, artefactos y parámetros
- **DVC** para gestión y orquestación de datos/modelos
- **DagsHub** para control de versiones y visualización del historial

---

### MLflow + DagsHub

Todo el flujo de experimentación del proyecto es trazable y está registrado utilizando **MLflow**, con visualización e integración completa en **DagsHub**.

#### Visualización de experimentos

* **MLflow UI local**: Si trabajas localmente, puedes levantar la interfaz con:

  ```bash
  mlflow ui
  ```

  Luego navega a [http://localhost:5000](http://localhost:5000) para ver los experimentos.

* **MLflow vía DagsHub**: Los experimentos se visualizan también desde DagsHub, con trazabilidad completa de:

  * Métricas (F1, AUC, Precision, Recall)
  * Parámetros (hiperparámetros de modelos)
  * Artefactos (modelos, vectorizadores, selectores)
  * Comparación de versiones

#### 🔗 Enlaces relevantes

* **Repositorio en DagsHub**:
  [https://dagshub.com/TU\_USUARIO/TU\_REPO]([https://dagshub.com/TU_USUARIO/TU_REPO](https://dagshub.com/AngelBReal/DeAMentis-AAA))

* **Experimentos MLflow en DagsHub**:
  [https://dagshub.com/TU\_USUARIO/TU\_REPO.mlflow]([https://dagshub.com/TU_USUARIO/TU_REPO.mlflow](https://dagshub.com/AngelBReal/DeAMentis-AAA.mlflow/#/experiments/7?searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D))

---

## 🌐 Aplicación Web

Una vez desplegada, la aplicación estará disponible públicamente aquí:

🔗 **[Enlace a la app desplegada](https://TU_LINK_RENDER_AQUI)**  
*(En proceso de deployment)*

### Backend

- Construido con **FastAPI**
- Expone una API REST para recibir texto y entregar:
  - Nivel de riesgo (*bajo*, *moderado*, *alto*)
  - Advertencias generadas por el analizador lingüístico
  - Probabilidad de falsedad (`prob_fake`)
- Utiliza el modelo VotingClassifier entrenado y registrado vía MLflow

### Frontend

- Desarrollado en **React** con **Bootstrap**
- Presenta resultados de forma clara y educativa:
  - Muestra una barra de riesgo
  - Lista las advertencias lingüísticas detectadas
  - Indica si el contenido requiere una lectura crítica

---

## 🗂️ Estructura del Repositorio

```

app/
├── backend/              # API con FastAPI y modelo final
├── frontend/             # Interfaz React
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

## Arquitectura y Automatización

Este proyecto sigue una arquitectura **modular, reproducible y automatizada**, usando:

- `Typer` como interfaz CLI
- `DVC` como orquestador de pipelines
- `MLflow` para seguimiento de modelos y experimentos

### Flujo general:

```text
Descarga → Preprocesamiento → Entrenamiento del modelo final
````

---

##  CLI (`src/cli.py`)

```bash
download-all      # Descarga datasets
process-all       # Une y limpia los datasets
preprocess-all    # Aplica preprocesamiento
build-model       # Entrena y guarda el modelo final
```

---

## Pipeline DVC

```yaml
stages:
  raw_pipeline:
    cmd: set PYTHONPATH=src && python src/cli.py download-all && python src/cli.py process-all
    outs:
      - data/interim/combined_fakenews_dataset.csv
      - data/raw

  preprocess_pipeline:
    cmd: python src/cli.py preprocess-all
    outs:
      - data/processed

  train_model_pipeline:
    cmd: python src/cli.py build-model
    outs:
      - app/backend/final_model/final_voting_model.pkl
      - app/backend/final_model/artifacts/tfidf_vectorizer.pkl
      - app/backend/final_model/artifacts/selector.pkl
```

Ejecuta el flujo completo con:

```bash
dvc repro
```

---

## 👤 Autor

**Ing. Angel Barraza Real**
Maestría en Ciencia de Datos – UNISON
📧 [angelbarrazareal@gmail.com](mailto:angelbarrazareal@gmail.com)

---

## 🧾 Licencia

**Pendiente de definir**
¿
