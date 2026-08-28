import json
import os
from typing import Any

import joblib
import mlflow
import mlflow.xgboost
import pandas as pd

from src.utils.mlflow_config import configure_mlflow


REGISTERED_MODEL_NAME = "AI-Risk-Manager-XGBoost"
CHAMPION_ALIAS = "champion"

PREPROCESSOR_PATH = "models/preprocessor.pkl"
CONFIG_PATH = "models/best_ensemble_config.json"


# Exact 117 model-input features used during training.
FEATURE_COLUMNS = [
    "TransactionDT",
    "TransactionAmt",
    "ProductCD",
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "addr2",
    "dist1",
    "P_emaildomain",
    "R_emaildomain",
    "C2",
    "C3",
    "C9",
    "D1",
    "D3",
    "D5",
    "D11",
    "D15",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
    "M7",
    "M8",
    "M9",
    "V1",
    "V3",
    "V5",
    "V6",
    "V12",
    "V14",
    "V20",
    "V23",
    "V26",
    "V29",
    "V35",
    "V38",
    "V41",
    "V45",
    "V47",
    "V52",
    "V53",
    "V56",
    "V62",
    "V65",
    "V67",
    "V68",
    "V83",
    "V86",
    "V89",
    "V107",
    "V111",
    "V117",
    "V120",
    "V123",
    "V169",
    "V173",
    "V174",
    "V197",
    "V199",
    "V220",
    "V222",
    "V223",
    "V235",
    "V239",
    "V240",
    "V247",
    "V257",
    "V262",
    "V271",
    "V281",
    "V283",
    "V284",
    "V286",
    "V287",
    "V289",
    "V290",
    "V301",
    "V302",
    "V305",
    "V312",
    "V315",
    "id_01",
    "id_02",
    "id_05",
    "id_06",
    "id_11",
    "id_12",
    "id_13",
    "id_15",
    "id_16",
    "id_17",
    "id_19",
    "id_20",
    "id_28",
    "id_29",
    "id_31",
    "id_35",
    "id_36",
    "id_37",
    "id_38",
    "TransactionHour",
    "TransactionDay",
    "TransactionWeek",
    "TransactionWeekday",
    "TransactionAmt_Log",
    "card1_freq",
    "EmailDomainMatch",
    "P_email_Missing",
    "R_email_Missing",
    "CardType",
]


def load_production_model():
    """Load the model currently assigned to MLflow @champion."""

    configure_mlflow()

    model_uri = (
        f"models:/{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}"
    )

    model = mlflow.pyfunc.load_model(model_uri)

    return model


def load_preprocessor():
    """Load the exact preprocessing pipeline used during training."""

    if not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError(
            f"Preprocessor not found: {PREPROCESSOR_PATH}"
        )

    return joblib.load(PREPROCESSOR_PATH)


def load_production_config():
    """Load the production threshold configuration."""

    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(
            f"Production configuration not found: {CONFIG_PATH}"
        )

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def validate_transaction(transaction: dict[str, Any]):
    """Validate that the incoming transaction has the required schema."""

    if not isinstance(transaction, dict):
        raise TypeError("Transaction must be a dictionary.")

    missing_features = [
        feature
        for feature in FEATURE_COLUMNS
        if feature not in transaction
    ]

    if missing_features:
        raise ValueError(
            "Missing required features: "
            + ", ".join(missing_features)
        )

    extra_features = [
        feature
        for feature in transaction
        if feature not in FEATURE_COLUMNS
    ]

    if extra_features:
        raise ValueError(
            "Unexpected features: "
            + ", ".join(extra_features)
        )


def preprocess_transaction(transaction):
    """Transform one raw transaction into the 422-feature representation."""

    validate_transaction(transaction)

    preprocessor = load_preprocessor()

    dataframe = pd.DataFrame(
        [[transaction[feature] for feature in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS,
    )

    encoded = preprocessor.transform(dataframe)

    if encoded.shape[1] != 422:
        raise ValueError(
            f"Expected 422 encoded features, "
            f"received {encoded.shape[1]}."
        )

    return encoded


def predict_transaction(transaction):
    """Run production fraud prediction for one transaction."""

    model = load_production_model()

    config = load_production_config()

    threshold = float(
        config["production_threshold"]
    )

    encoded = preprocess_transaction(
        transaction
    )

    prediction_result = model.predict(encoded)

    print(
        "DEBUG MLflow prediction type:",
        type(prediction_result),
    )
    
    print(
        "DEBUG MLflow prediction:",
        prediction_result,
    )
    
    raise RuntimeError(
        "DEBUG: inspected MLflow predict() output."
    )

    prediction = int(
        probability >= threshold
    )

    return {
        "model": REGISTERED_MODEL_NAME,
        "alias": CHAMPION_ALIAS,
        "fraud_probability": probability,
        "threshold": threshold,
        "prediction": prediction,
        "label": (
            "Fraud"
            if prediction == 1
            else "Legitimate"
        ),
    }
