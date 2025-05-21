import mlflow
from datetime import datetime

# === Establecer experimento separado para predicciones ===
mlflow.set_tracking_uri("https://dagshub.com/AngelBReal/DeAMentis-AAA.mlflow")
mlflow.set_experiment("Noticias - Inference Log")


def log_inference_to_mlflow(title: str, body: str, prediction: str, alerts: list):
    """
    Registra en MLflow una predicción realizada por el API, con tags útiles.
    """
    run_name = "api_inference_" + datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    with mlflow.start_run(run_name=run_name, nested=True):
        mlflow.log_param("title", title[:80])
        mlflow.log_param("body", body[:80])
        mlflow.log_param("prediction", prediction)
        mlflow.log_param("n_alerts", len(alerts))

        mlflow.set_tag("type", "inference")
        mlflow.set_tag("timestamp", datetime.utcnow().isoformat())
