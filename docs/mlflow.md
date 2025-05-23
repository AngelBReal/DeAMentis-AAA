# Seguimiento con MLflow

Utilizamos MLflow conectado con DagsHub para:

- Registrar experimentos
- Comparar métricas por variante
- Almacenar artefactos
- Versionar modelos (`model.pkl`, `vectorizer.pkl`, etc.)

## Estructura de seguimiento

- Experimento: `FakeNewsSpanish`
- Métricas: Accuracy, Precision, Recall, F1
- Tags: versión de preprocessing, features

## Ejemplo de uso

```bash
mlflow ui
```
