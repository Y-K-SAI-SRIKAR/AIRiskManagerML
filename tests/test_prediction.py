import json
import os

import mlflow
import mlflow.xgboost
import pandas as pd

from src.data.split_data import split_data
from src.data.model_processing import process_data
from src.utils.mlflow_config import configure_mlflow


DATA_PATH = "data/processed/feature_engineered.csv"

REGISTERED_MODEL = "AI-Risk-Manager-XGBoost"

CONFIG_PATH = "models/best_ensemble_config.json"


def load_production_model():

    configure_mlflow()

    model_uri = (
        f"models:/{REGISTERED_MODEL}@champion"
    )

    model = mlflow.xgboost.load_model(
        model_uri
    )

    return model


def test_production_model_prediction_shape():

    model = load_production_model()

    data = pd.read_csv(DATA_PATH)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(data)

    (
        X_train_encoded,
        X_val_encoded,
        X_test_encoded,
        preprocessor
    ) = process_data(
        X_train,
        X_val,
        X_test
    )

    sample = X_test_encoded[:10]

    probabilities = model.predict_proba(sample)

    assert probabilities.shape == (10, 2)


def test_probability_range():

    model = load_production_model()

    data = pd.read_csv(DATA_PATH)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(data)

    (
        X_train_encoded,
        X_val_encoded,
        X_test_encoded,
        preprocessor
    ) = process_data(
        X_train,
        X_val,
        X_test
    )

    sample = X_test_encoded[:20]

    probabilities = model.predict_proba(sample)[:, 1]

    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_prediction_threshold():

    assert os.path.exists(CONFIG_PATH)

    with open(CONFIG_PATH) as file:
        config = json.load(file)

    threshold = config["production_threshold"]

    assert 0.0 < threshold < 1.0

    model = load_production_model()

    data = pd.read_csv(DATA_PATH)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(data)

    (
        X_train_encoded,
        X_val_encoded,
        X_test_encoded,
        preprocessor
    ) = process_data(
        X_train,
        X_val,
        X_test
    )

    probability = model.predict_proba(
        X_test_encoded[:1]
    )[0, 1]

    prediction = int(
        probability >= threshold
    )

    assert prediction in [0, 1]