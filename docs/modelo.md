# Pipeline de Modelado

El modelo está basado en un `VotingClassifier` que combina:
- `XGBoost`
- `SVM`

## Flujo de entrenamiento

1. Preprocesamiento (TF-IDF, SelectKBest)
2. Balanceo con SMOTE
3. Entrenamiento
4. Registro con MLflow
5. Exportación del modelo y artefactos `.pkl`

### Artefactos
- `tfidf_vectorizer.pkl`
- `selector.pkl`
- `model.pkl`
