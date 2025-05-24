### Sección: Seguimiento de experimentos con MLflow (`mlflow.md`)

# Seguimiento de Modelos con MLflow

Todo el ciclo de vida del modelo en **De A Mentis** fue gestionado utilizando **MLflow**, una herramienta de experiment tracking que permite registrar, visualizar y comparar métricas, hiperparámetros y artefactos.

---

## ¿Por qué MLflow?

MLflow facilita:

- Comparar múltiples experimentos de forma visual y estructurada.
- Reproducir ejecuciones anteriores con los mismos parámetros.
- Versionar modelos, vectorizadores y selectores de features.
- Integrar con DVC y DagsHub para trazabilidad completa.

---

## Infraestructura

Los experimentos fueron ejecutados en **Google Colab**, con sincronización de artefactos y métricas a **MLflow**, conectado a **DagsHub** mediante URI remota.

### Configuración típica

```python
import mlflow

mlflow.set_tracking_uri("https://dagshub.com/AngelBReal/DeAMentis-AAA.mlflow")
mlflow.set_experiment("Nombre del experimento")
```

---

## Organización de Experimentos

La convención de nombres ayuda a rastrear fácilmente el propósito de cada ejecución:

| Prefijo | Descripción                                                        |
| ------- | ------------------------------------------------------------------ |
| `00_`   | Exploración inicial, pruebas de preprocesamiento y modelos         |
| `01_`   | Experimentación con datasets curados, comparación de modelos       |
| `02_`   | Optimización con Optuna (Voting Ensemble, hiperparámetros)         |
| `03_`   | Evaluaciones adicionales y combinaciones de datasets               |
| `04_`   | Modelo final registrado y preparado para despliegue                |
| `100_`  | Inference Logs (cada petición del backend genera un run en MLflow) |

---

## ¿Qué se registra?

Cada run en MLflow contiene:

* 📈 **Métricas**: `f1_score`, `precision`, `recall`, `AUC`, `accuracy`
* ⚙️ **Parámetros**: hiperparámetros del modelo (e.g. `svm_C`, `xgb_lr`)
* 📦 **Artefactos**:

  * Modelo serializado (`.pkl`)
  * Vectorizador TF-IDF
  * Selector de características
* 📄 **Tags**: tipo de modelo, tipo de preprocesamiento, n-gramas

---

## Visualización Local

Puedes levantar la interfaz local de MLflow con:

```bash
mlflow ui
```

Y acceder en [http://localhost:5000](http://localhost:5000)

---

## Visualización en DagsHub

Todos los experimentos están disponibles en línea en:

🔗 [MLflow en DagsHub](https://dagshub.com/AngelBReal/DeAMentis-AAA.mlflow)

---

## Ejemplos de Runs

### 🧪 Experimentación con SVM

| Run Name                   | F1-score | AUC   | Preprocesamiento | N-grama   |
| -------------------------- | -------- | ----- | ---------------- | --------- |
| Bigram\_lemmatized\_\_SVM  | 0.647    | 0.228 | Lematización     | Bigramas  |
| Unigram\_lemmatized\_\_SVM | 0.632    | 0.236 | Lematización     | Unigramas |
| Unigram\_raw\_\_SVM        | 0.600    | 0.262 | Raw              | Unigramas |

---

### 🧪 Ensamble calibrado

| Run Name         | Modelo        | F1-score | AUC   |
| ---------------- | ------------- | -------- | ----- |
| Voting\_Ensemble | SVM + XGBoost | 0.775    | 0.794 |
| XGBoost          | XGBoost       | 0.743    | 0.764 |
| SVM\_Calibrated  | SVM           | 0.665    | 0.786 |

---

### 🧪 Optimización con Optuna

| svm\_C | xgb\_estimators | xgb\_lr | F1-score | AUC   |
| ------ | --------------- | ------- | -------- | ----- |
| 1.287  | 149             | 0.205   | 0.783    | 0.795 |
| 1.840  | 176             | 0.239   | 0.783    | 0.802 |
| 4.217  | 165             | 0.172   | 0.782    | 0.797 |

---

##  Inferencias como Experimentos

Cada vez que el backend (FastAPI) recibe una noticia, genera un **nuevo run en MLflow** en la categoría `100 Noticias - Inference Log`, registrando:

* Texto recibido
* Probabilidad (`prob_fake`)
* Advertencias generadas
* Tiempo de inferencia

Esto permite tener trazabilidad total del modelo en producción.

---

## Conclusión

MLflow, combinado con DagsHub y DVC, permitió construir una infraestructura robusta y reproducible de experimentación, ideal para proyectos educativos y científicos como **De A Mentis**.

