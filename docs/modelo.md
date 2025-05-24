# Proceso de Modelado

Este proyecto exploró diversas estrategias de clasificación de noticias con el objetivo de construir un modelo efectivo, trazable y explicable. A continuación se describe el flujo completo de experimentación, desde pruebas iniciales hasta la selección final del modelo.

---

## 1. Infraestructura de Experimentación

Todos los modelos fueron entrenados en **Google Colab**, aprovechando créditos gratuitos para ejecución de GPU/CPU intensiva. Cada experimento fue registrado usando **MLflow**, con artefactos sincronizados en **DagsHub**, lo que permite una trazabilidad total del proceso de modelado.

- 🔗 Visualización completa de experimentos: [MLflow en DagsHub](https://dagshub.com/AngelBReal/DeAMentis-AAA.mlflow)

---

## 2. Fases de Experimentación

### `00_`: Experimentos exploratorios

Experimentos iniciales con distintos modelos base (`SVM`, `RegLog`, `GradientBoosting`, etc.) y preprocesamiento textual básico. El objetivo fue entender la sensibilidad del rendimiento ante distintas transformaciones de texto.

📌 Conclusión: El preprocesamiento (stopwords, lematización, ngramas) influye más que el modelo en sí. Esto motivó la siguiente fase más sistemática.

---

### `01_`: Comparación estructurada de datasets y modelos

Se probaron combinaciones de modelos clásicos (`Naive Bayes`, `SVM`, `AdaBoost`, `RandomForest`) con 5 variantes de preprocesamiento:

- Raw unigramas
- Raw bigramas
- Stopwords unigramas
- Stemmed bigramas
- Lematizado + unigramas / bigramas

#### Resultados (SVM con distintas variantes):

| Modelo                        | Preprocesamiento | N-grama  | F1-score | AUC     |
|------------------------------|------------------|----------|----------|---------|
| Bigram_lemmatized__SVM       | Lematizado       | Bigrama  | 0.647    | 0.228   |
| Unigram_lemmatized__SVM      | Lematizado       | Unigrama | 0.632    | 0.236   |
| Unigram_stopwords__SVM       | Stopwords        | Unigrama | 0.600    | 0.268   |
| Unigram_raw__SVM             | Raw              | Unigrama | 0.600    | 0.262   |
| Unigram_stemmed__SVM         | Stemmed          | Unigrama | 0.600    | 0.276   |

Conclusión: SVM combinado con lematización y unigramas obtuvo los mejores resultados.

---

### `01 Best Dataset Calibrado V2`

Aquí se desarrolló el **Voting Ensemble** (SVM + XGBoost) sobre el mejor dataset. Se compararon diferentes modelos:

#### Comparación de modelos calibrados

| Modelo            | Accuracy | F1-score | AUC     | Precisión | Recall  |
|-------------------|----------|----------|---------|-----------|---------|
| Voting Ensemble   | 0.731    | 0.775    | 0.794   | 0.796     | 0.756   |
| XGBoost           | 0.693    | 0.743    | 0.764   | 0.762     | 0.726   |
| SVM Calibrado     | 0.729    | 0.665    | 0.786   | 0.637     | 0.696   |
| LightGBM          | 0.703    | 0.632    | 0.785   | 0.607     | 0.659   |

🏆 El Voting Ensemble superó a todos los modelos individuales.

---

### `02 Optuna Ensemble Search`

Se aplicó **Optuna** para encontrar los hiperparámetros óptimos del ensamble VotingClassifier.

#### Top combinaciones encontradas:

| svm_C   | xgb_estimators | xgb_lr  | F1-score | AUC     |
|---------|----------------|---------|----------|---------|
| 1.287   | 149            | 0.205   | 0.783    | 0.795   |
| 1.840   | 176            | 0.239   | 0.783    | 0.802   |
| 4.217   | 165            | 0.172   | 0.782    | 0.797   |
| 9.882   | 120            | 0.122   | 0.780    | 0.797   |
| 5.307   | 200            | 0.040   | 0.779    | 0.797   |

 El modelo con `svm_C = 1.287`, `xgb_lr = 0.205`, `xgb_estimators = 149` fue el elegido por su balance rendimiento/estabilidad.

---

## 3. Selección del Modelo Final

El modelo ganador fue el **VotingClassifier (SVM + XGBoost)** entrenado con:

- TF-IDF con unigramas
- Lematización
- Selección de características con `SelectKBest`
- Balanceo con `SMOTE`

Este modelo fue registrado como artefacto final en MLflow y es utilizado en producción por el backend de FastAPI.

---

## 4. Alternativas probadas sin éxito

También se evaluaron modelos con **transformers** como **BETO**, pero no superaron al Voting Ensemble en F1-score ni en interpretabilidad, por lo cual se descartaron en esta fase.

---

## Trazabilidad completa

Todos los experimentos pueden ser visualizados y comparados desde:

🔗 [MLflow UI en DagsHub](https://dagshub.com/AngelBReal/DeAMentis-AAA.mlflow)

Cada ejecución incluye:
- Hiperparámetros
- Métricas (F1, AUC, Precision, Recall)
- Artefactos (modelo, vectorizador, selector)
