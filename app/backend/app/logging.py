import mlflow
from datetime import datetime

# === Establecer experimento separado para predicciones ===
mlflow.set_tracking_uri("https://dagshub.com/AngelBReal/DeAMentis-AAA.mlflow")
mlflow.set_experiment("Noticias - Inference Log")


def log_inference_to_mlflow(title, body, prediction, alerts, prediction_raw, prob_fake):
    run_name = "api_inference_" + datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    with mlflow.start_run(run_name=run_name, nested=True):
        mlflow.log_param("title", title[:80])
        mlflow.log_param("body", body[:80])
        mlflow.log_param("prediction_label", prediction)
        mlflow.log_param("alert_text", "; ".join(alerts))

        mlflow.log_metric("n_alerts", len(alerts))
        mlflow.log_metric("title_length", len(title))
        mlflow.log_metric("body_length", len(body))
        mlflow.log_metric("prediction_raw", prediction_raw)
        mlflow.log_metric("prob_fake", prob_fake)

        mlflow.set_tag("type", "inference")
        mlflow.set_tag("timestamp", datetime.utcnow().isoformat())
