import mlflow
from datetime import datetime

# === Establecer experimento separado para predicciones ===
mlflow.set_tracking_uri("https://dagshub.com/AngelBReal/DeAMentis-AAA.mlflow")
mlflow.set_experiment("Noticias - Inference Log")


def log_inference_to_mlflow(title, body, prediction, alerts, prediction_raw, prob_fake, extra_tags=None):
    run_name = "api_inference_" + datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    with mlflow.start_run(run_name=run_name, nested=True):
        # Logs básicos
        mlflow.log_param("title", title[:80])
        mlflow.log_param("body", body[:80])
        mlflow.log_param("prediction_label", prediction)
        # Asegura que alerts sea una lista plana de strings
        flat_alerts = []
        for alert in alerts:
            if isinstance(alert, list):
                flat_alerts.extend(alert)
            else:
                flat_alerts.append(str(alert))

        mlflow.log_param("alert_text", "; ".join(flat_alerts))

        mlflow.log_metric("n_alerts", len(alerts))
        mlflow.log_metric("title_length", len(title))
        mlflow.log_metric("body_length", len(body))
        mlflow.log_metric("prediction_raw", prediction_raw)
        mlflow.log_metric("prob_fake", prob_fake)

        # Tags base
        mlflow.set_tag("type", "inference")
        mlflow.set_tag("timestamp", datetime.utcnow().isoformat())

        # Tags adicionales (si se mandan)
        if extra_tags:
            for k, v in extra_tags.items():
                mlflow.set_tag(k, v)