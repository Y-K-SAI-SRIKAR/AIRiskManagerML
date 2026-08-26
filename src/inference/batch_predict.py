import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import boto3
import mlflow.xgboost
import pandas as pd
from botocore.config import Config

from src.inference.predict import (
    FEATURE_COLUMNS,
    REGISTERED_MODEL_NAME,
    CHAMPION_ALIAS,
    load_preprocessor,
    load_production_config,
)
from src.utils.mlflow_config import configure_mlflow


# ============================================================
# REPORT S3 CONFIGURATION
# ============================================================

REPORT_BUCKET = os.getenv(
    "REPORT_S3_BUCKET",
    "ai-risk-manager-reports-bucket",
)

REPORT_PREFIX = os.getenv(
    "REPORT_S3_PREFIX",
    "reports",
)

REPORT_S3_REGION = "ap-south-1"

BATCH_SIZE = int(
    os.getenv(
        "BATCH_SIZE",
        "10000",
    )
)


# ============================================================
# S3 CLIENT
# ============================================================

def get_s3_client():
    """
    Create a dedicated S3 client for the report bucket.

    IMPORTANT:
    MLflow artifacts are stored in ap-south-2.
    Report files are stored in ap-south-1.

    Therefore this client MUST NOT inherit the MLflow
    AWS_REGION configuration.
    """

    return boto3.client(
        "s3",
        region_name=REPORT_S3_REGION,
        endpoint_url=(
            f"https://s3.{REPORT_S3_REGION}.amazonaws.com"
        ),
        config=Config(
            signature_version="s3v4",
            region_name=REPORT_S3_REGION,
        ),
    )


# ============================================================
# JOB ID
# ============================================================

def create_job_id():
    return uuid.uuid4().hex


# ============================================================
# S3 UPLOAD
# ============================================================

def upload_to_s3(
    local_path,
    s3_key,
):
    s3 = get_s3_client()

    s3.upload_file(
        str(local_path),
        REPORT_BUCKET,
        s3_key,
    )


# ============================================================
# PRESIGNED DOWNLOAD URL
# ============================================================

def create_presigned_url(
    s3_key,
    expires=3600,
):
    """
    Generate a SigV4 presigned URL using the exact
    report-bucket region and regional S3 endpoint.
    """

    s3 = get_s3_client()

    return s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={
            "Bucket": REPORT_BUCKET,
            "Key": s3_key,
        },
        ExpiresIn=expires,
        HttpMethod="GET",
    )


# ============================================================
# BATCH PROCESSING
# ============================================================

def process_batch_csv(input_path):
    """
    Run the existing production champion XGBoost model
    against a large CSV.

    IMPORTANT:
    This function performs inference only.

    It does NOT:
        - retrain the model
        - modify the model
        - modify MLflow model versions
        - modify the champion alias
    """

    # --------------------------------------------------------
    # CONFIGURE MLFLOW
    # --------------------------------------------------------

    configure_mlflow()

    # --------------------------------------------------------
    # CREATE JOB
    # --------------------------------------------------------

    job_id = create_job_id()

    output_dir = (
        Path("/tmp")
        / "ai-risk-manager"
        / job_id
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_path = (
        output_dir
        / "predictions.csv"
    )

    report_path = (
        output_dir
        / "report.json"
    )

    # --------------------------------------------------------
    # LOAD MLflow CLIENT
    # --------------------------------------------------------

    from mlflow.tracking import MlflowClient

    client = MlflowClient()

    # --------------------------------------------------------
    # GET CURRENT CHAMPION
    # --------------------------------------------------------

    champion = (
        client.get_model_version_by_alias(
            REGISTERED_MODEL_NAME,
            CHAMPION_ALIAS,
        )
    )

    # --------------------------------------------------------
    # MODEL URI
    # --------------------------------------------------------

    model_uri = (
        f"models:/{REGISTERED_MODEL_NAME}"
        f"@{CHAMPION_ALIAS}"
    )

    # --------------------------------------------------------
    # LOAD XGBOOST MODEL
    # --------------------------------------------------------

    # Direct XGBoost loading is intentional here because
    # batch inference requires predict_proba().
    model = mlflow.xgboost.load_model(
        model_uri
    )

    # --------------------------------------------------------
    # LOAD PREPROCESSOR
    # --------------------------------------------------------

    preprocessor = load_preprocessor()

    # --------------------------------------------------------
    # LOAD PRODUCTION CONFIG
    # --------------------------------------------------------

    config = load_production_config()

    threshold = float(
        config["production_threshold"]
    )

    # --------------------------------------------------------
    # COUNTERS
    # --------------------------------------------------------

    total = 0

    fraud_count = 0

    legitimate_count = 0

    probability_sum = 0.0

    first_chunk = True

    # ========================================================
    # READ CSV IN CHUNKS
    # ========================================================

    for dataframe in pd.read_csv(
        input_path,
        chunksize=BATCH_SIZE,
    ):

        # ----------------------------------------------------
        # REMOVE NON-MODEL COLUMNS
        # ----------------------------------------------------

        dataframe = dataframe.drop(
            columns=[
                "isFraud",
                "TransactionAmt_Bin",
            ],
            errors="ignore",
        )

        # ----------------------------------------------------
        # REQUIRED FEATURE VALIDATION
        # ----------------------------------------------------

        missing = [
            column
            for column in FEATURE_COLUMNS
            if column not in dataframe.columns
        ]

        if missing:

            raise ValueError(
                "Missing required features: "
                + ", ".join(missing)
            )

        # ----------------------------------------------------
        # UNEXPECTED FEATURE VALIDATION
        # ----------------------------------------------------

        extra = [
            column
            for column in dataframe.columns
            if column not in FEATURE_COLUMNS
        ]

        if extra:

            raise ValueError(
                "Unexpected features: "
                + ", ".join(extra)
            )

        # ----------------------------------------------------
        # SELECT EXACT PRODUCTION FEATURES
        # ----------------------------------------------------

        model_input = dataframe[
            FEATURE_COLUMNS
        ]

        # ----------------------------------------------------
        # PREPROCESS
        # ----------------------------------------------------

        encoded = preprocessor.transform(
            model_input
        )

        # ----------------------------------------------------
        # FRAUD PROBABILITY
        # ----------------------------------------------------

        probabilities = (
            model.predict_proba(
                encoded
            )[:, 1]
        )

        # ----------------------------------------------------
        # APPLY PRODUCTION THRESHOLD
        # ----------------------------------------------------

        predictions = (
            probabilities >= threshold
        ).astype(int)

        labels = [
            "Fraud"
            if value == 1
            else "Legitimate"
            for value in predictions
        ]

        # ----------------------------------------------------
        # ADD OUTPUT COLUMNS
        # ----------------------------------------------------

        dataframe[
            "fraud_probability"
        ] = probabilities

        dataframe[
            "threshold"
        ] = threshold

        dataframe[
            "prediction"
        ] = predictions

        dataframe[
            "label"
        ] = labels

        # ----------------------------------------------------
        # WRITE OUTPUT CSV
        # ----------------------------------------------------

        dataframe.to_csv(
            result_path,
            mode=(
                "w"
                if first_chunk
                else "a"
            ),
            header=first_chunk,
            index=False,
        )

        first_chunk = False

        # ----------------------------------------------------
        # UPDATE COUNTERS
        # ----------------------------------------------------

        chunk_size = len(dataframe)

        total += chunk_size

        current_fraud = int(
            predictions.sum()
        )

        fraud_count += current_fraud

        legitimate_count += (
            chunk_size
            - current_fraud
        )

        probability_sum += float(
            probabilities.sum()
        )

    # ========================================================
    # EMPTY DATASET CHECK
    # ========================================================

    if total == 0:

        raise ValueError(
            "The uploaded CSV contains no transactions."
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    average_probability = (
        probability_sum / total
    )

    fraud_rate = (
        fraud_count
        / total
        * 100
    )

    created_at = datetime.now(
        timezone.utc
    ).isoformat()

    # ========================================================
    # REPORT
    # ========================================================

    report = {
        "job_id": job_id,

        "created_at": created_at,

        "model": REGISTERED_MODEL_NAME,

        "alias": CHAMPION_ALIAS,

        "model_version": int(
            champion.version
        ),

        "run_id": champion.run_id,

        "total_transactions": total,

        "fraud_transactions": fraud_count,

        "legitimate_transactions": (
            legitimate_count
        ),

        "fraud_rate": fraud_rate,

        "average_fraud_probability": (
            average_probability
        ),

        "production_threshold": threshold,

        "model_type": config.get(
            "model_type"
        ),

        "xgb_weight": config.get(
            "xgb_weight"
        ),

        "nn_weight": config.get(
            "nn_weight"
        ),

        "selection_metric": config.get(
            "selection_metric"
        ),

        "threshold_selection_metric": (
            config.get(
                "threshold_selection_metric"
            )
        ),

        "report_s3_region": REPORT_S3_REGION,
    }

    # ========================================================
    # SAVE REPORT
    # ========================================================

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
        )

    # ========================================================
    # S3 KEYS
    # ========================================================

    result_key = (
        f"{REPORT_PREFIX}/"
        f"{job_id}/"
        f"predictions.csv"
    )

    report_key = (
        f"{REPORT_PREFIX}/"
        f"{job_id}/"
        f"report.json"
    )

    # ========================================================
    # UPLOAD RESULTS
    # ========================================================

    upload_to_s3(
        result_path,
        result_key,
    )

    upload_to_s3(
        report_path,
        report_key,
    )

    # ========================================================
    # PRESIGNED URLS
    # ========================================================

    result_download_url = (
        create_presigned_url(
            result_key
        )
    )

    report_download_url = (
        create_presigned_url(
            report_key
        )
    )

    # ========================================================
    # FINAL API RESPONSE
    # ========================================================

    return {
        **report,

        "result_download_url":
            result_download_url,

        "report_download_url":
            report_download_url,
    }