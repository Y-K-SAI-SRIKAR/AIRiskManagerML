import os
from dotenv import load_dotenv
import mlflow


load_dotenv()


def configure_mlflow():
    """
    Configure MLflow tracking server/database and artifact storage.

    Tracking metadata:
        AWS RDS MySQL

    Model artifacts:
        AWS S3
    """

    # ============================================================
    # MLflow database configuration
    # ============================================================

    user = os.getenv("MLFLOW_DB_USER")
    password = os.getenv("MLFLOW_DB_PASSWORD")
    host = os.getenv("MLFLOW_DB_HOST")
    port = os.getenv("MLFLOW_DB_PORT", "3306")
    database = os.getenv("MLFLOW_DB_NAME")

    required = {
        "MLFLOW_DB_USER": user,
        "MLFLOW_DB_PASSWORD": password,
        "MLFLOW_DB_HOST": host,
        "MLFLOW_DB_NAME": database,
    }

    missing = [
        key
        for key, value in required.items()
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

    # ============================================================
    # S3 artifact configuration
    # ============================================================

    artifact_bucket = os.getenv(
        "MLFLOW_ARTIFACT_BUCKET",
        "ai-risk-manager-mlflow-artifacts"
    )

    artifact_prefix = os.getenv(
        "MLFLOW_ARTIFACT_PREFIX",
        "mlflow"
    )

    artifact_uri = (
        f"s3://{artifact_bucket}/{artifact_prefix}"
    )

    # ============================================================
    # AWS credentials
    #
    # Do NOT hardcode credentials here.
    #
    # Locally:
    #   AWS CLI credentials / environment variables
    #
    # Docker:
    #   environment variables / IAM credentials
    #
    # Render:
    #   Render environment variables
    # ============================================================

    aws_region = os.getenv(
        "AWS_DEFAULT_REGION",
        os.getenv("AWS_REGION", "ap-south-2")
    )

    os.environ["AWS_DEFAULT_REGION"] = aws_region

    # ============================================================
    # Configure MLflow
    # ============================================================

    mlflow.set_tracking_uri(tracking_uri)

    # Store the artifact location for new MLflow runs.
    #
    # This does NOT move existing local artifacts.
    # Existing runs must be migrated/uploaded separately.
    os.environ["MLFLOW_DEFAULT_ARTIFACT_ROOT"] = artifact_uri

    mlflow.set_experiment("AI-Risk-Manager")

    return {
        "tracking_uri": tracking_uri,
        "artifact_uri": artifact_uri,
        "aws_region": aws_region,
    }