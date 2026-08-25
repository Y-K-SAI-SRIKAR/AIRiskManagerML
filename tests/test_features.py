import pandas as pd

from src.data.split_data import split_data
from src.data.model_processing import process_data

from src.features.feature_config import (
    CATEGORICAL_FEATURES,
    EXPECTED_NUMERIC_COUNT,
    EXPECTED_ENCODED_FEATURE_COUNT
)


DATA_PATH = "data/processed/feature_engineered.csv"


def test_feature_configuration():

    assert len(CATEGORICAL_FEATURES) == 25

    assert EXPECTED_NUMERIC_COUNT == 92

    assert EXPECTED_ENCODED_FEATURE_COUNT == 422


def test_dataset_feature_count():

    data = pd.read_csv(DATA_PATH)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(data)

    assert X_train.shape[1] == 117

    assert X_val.shape[1] == 117

    assert X_test.shape[1] == 117


def test_preprocessing_encoded_features():

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

    assert X_train_encoded.shape[1] == EXPECTED_ENCODED_FEATURE_COUNT

    assert X_val_encoded.shape[1] == EXPECTED_ENCODED_FEATURE_COUNT

    assert X_test_encoded.shape[1] == EXPECTED_ENCODED_FEATURE_COUNT


def test_preprocessor_feature_names():

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

    feature_names = preprocessor.get_feature_names_out()

    assert len(feature_names) == EXPECTED_ENCODED_FEATURE_COUNT
