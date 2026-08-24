import os
from dotenv import load_dotenv
import mlflow


load_dotenv()


def configure_mlflow():

    user = os.getenv("MLFLOW_DB_USER")
    password = os.getenv("MLFLOW_DB_PASSWORD")
    host = os.getenv("MLFLOW_DB_HOST")
    port = os.getenv("MLFLOW_DB_PORT", "3306")
    database = os.getenv("MLFLOW_DB_NAME")

    # Check required variables
    required = {
        "MLFLOW_DB_USER": user,
        "MLFLOW_DB_PASSWORD": password,
        "MLFLOW_DB_HOST": host,
        "MLFLOW_DB_NAME": database,
    }

    missing = [
        key for key, value in required.items()
        if not value
    ]

    if missing:
        raise ValueError(
            f"Missing MLflow environment variables: {missing}"
        )

    tracking_uri = (
        f"mysql+pymysql://"
        f"{user}:{password}@"
        f"{host}:{port}/"
        f"{database}"
    )

    mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment("AI-Risk-Manager")

    return tracking_uri