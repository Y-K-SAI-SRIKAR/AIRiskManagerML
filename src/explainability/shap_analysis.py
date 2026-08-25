import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import shap

from src.data.split_data import split_data
from src.data.model_processing import process_data
from src.utils.mlflow_config import configure_mlflow


# ============================================================
# CONFIGURATION
# ============================================================

REGISTERED_MODEL_NAME = "AI-Risk-Manager-XGBoost"
CHAMPION_ALIAS = "champion"

DATA_PATH = Path(
    "data/processed/feature_engineered.csv"
)

PREPROCESSOR_PATH = Path(
    "models/preprocessor.pkl"
)

OUTPUT_DIR = Path(
    "reports/shap"
)

DEFAULT_SAMPLE_SIZE = 1000
RANDOM_STATE = 42
PRODUCTION_THRESHOLD = 0.70
EXPECTED_ENCODED_FEATURE_COUNT = 422


# ============================================================
# HELPERS
# ============================================================

def require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )


def get_champion_model():
    """
    Load the exact model currently assigned to @champion
    from MLflow Model Registry.
    """

    client = mlflow.MlflowClient()

    champion = client.get_model_version_by_alias(
        name=REGISTERED_MODEL_NAME,
        alias=CHAMPION_ALIAS,
    )

    version = str(champion.version)

    model_uri = (
        f"models:/{REGISTERED_MODEL_NAME}@"
        f"{CHAMPION_ALIAS}"
    )

    print(
        f"Production model: {REGISTERED_MODEL_NAME}"
    )
    print(
        f"Production version: {version}"
    )
    print(
        f"Production alias: @{CHAMPION_ALIAS}"
    )
    print(
        f"Model URI: {model_uri}"
    )

    model = mlflow.xgboost.load_model(
        model_uri
    )

    return model, champion


def get_feature_names(preprocessor, encoded_count):
    """
    Recover transformed feature names from the fitted
    ColumnTransformer.
    """

    try:
        names = list(
            preprocessor.get_feature_names_out()
        )
    except Exception:
        names = [
            f"feature_{i}"
            for i in range(encoded_count)
        ]

    if len(names) != encoded_count:
        names = [
            f"feature_{i}"
            for i in range(encoded_count)
        ]

    return names


def make_shap_explainer(model):
    """
    TreeExplainer is appropriate for the XGBoost model.
    """

    return shap.TreeExplainer(model)


def normalize_shap_values(shap_values):
    """
    Normalize SHAP output to:
        (samples, features)
    """

    if isinstance(shap_values, list):
        if len(shap_values) == 2:
            shap_values = shap_values[1]
        else:
            shap_values = shap_values[0]

    shap_values = np.asarray(
        shap_values
    )

    if shap_values.ndim == 3:
        if shap_values.shape[-1] == 2:
            shap_values = shap_values[:, :, 1]
        else:
            shap_values = shap_values[:, :, 0]

    if shap_values.ndim != 2:
        raise ValueError(
            "Unexpected SHAP value shape: "
            f"{shap_values.shape}"
        )

    return shap_values


def to_dense(matrix):
    """
    Convert a scipy sparse matrix to a dense numpy array.

    Only the selected SHAP samples/row should be converted,
    never the complete test set.
    """

    if hasattr(matrix, "toarray"):
        return matrix.toarray()

    return np.asarray(matrix)


def save_json(path, payload):
    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
            ensure_ascii=False
        )


# ============================================================
# GLOBAL EXPLANATION
# ============================================================

def create_global_summary(
    shap_values,
    X_sample,
    feature_names,
):
    plt.figure(
        figsize=(12, 9)
    )

    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        show=False,
        max_display=20,
    )

    plt.tight_layout()

    output = (
        OUTPUT_DIR
        / "shap_summary.png"
    )

    plt.savefig(
        output,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    return output


def create_global_bar(
    shap_values,
    X_sample,
    feature_names,
):
    plt.figure(
        figsize=(12, 9)
    )

    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
        max_display=20,
    )

    plt.tight_layout()

    output = (
        OUTPUT_DIR
        / "shap_bar.png"
    )

    plt.savefig(
        output,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    return output


# ============================================================
# LOCAL EXPLANATION
# ============================================================

def create_local_explanation(
    model,
    explainer,
    X_row,
    feature_names,
    transaction_index,
):
    """
    Explain one transaction and save a waterfall plot plus
    machine-readable JSON.
    """

    row_shap = normalize_shap_values(
        explainer.shap_values(
            X_row
        )
    )[0]

    row_values = np.asarray(
        X_row
    )[0]

    expected_value = explainer.expected_value

    if isinstance(
        expected_value,
        (list, np.ndarray)
    ):
        expected_value = np.asarray(
            expected_value
        ).reshape(-1)[-1]

    expected_value = float(
        expected_value
    )

    probability = float(
        model.predict_proba(
            X_row
        )[0, 1]
    )

    prediction = (
        "Fraud"
        if probability >= PRODUCTION_THRESHOLD
        else "Legitimate"
    )

    order = np.argsort(
        np.abs(row_shap)
    )[::-1]

    top_features = []

    for position in order[:20]:
        top_features.append({
            "feature": str(
                feature_names[position]
            ),
            "encoded_value": float(
                row_values[position]
            ),
            "shap_value": float(
                row_shap[position]
            ),
            "absolute_shap_value": float(
                abs(row_shap[position])
            ),
            "direction": (
                "increases_fraud_risk"
                if row_shap[position] > 0
                else "decreases_fraud_risk"
            ),
        })

    explanation = {
        "model": {
            "name": REGISTERED_MODEL_NAME,
            "alias": CHAMPION_ALIAS,
        },
        "transaction_index": int(
            transaction_index
        ),
        "prediction": prediction,
        "fraud_probability": probability,
        "threshold": PRODUCTION_THRESHOLD,
        "base_value": expected_value,
        "top_features": top_features,
    }

    json_path = (
        OUTPUT_DIR
        / f"shap_values_{transaction_index}.json"
    )

    save_json(
        json_path,
        explanation
    )

    explanation_object = shap.Explanation(
        values=row_shap,
        base_values=expected_value,
        data=row_values,
        feature_names=feature_names,
    )

    plt.figure(
        figsize=(12, 9)
    )

    shap.plots.waterfall(
        explanation_object,
        max_display=15,
        show=False,
    )

    plt.tight_layout()

    plot_path = (
        OUTPUT_DIR
        / f"shap_local_{transaction_index}.png"
    )

    plt.savefig(
        plot_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    return (
        plot_path,
        json_path,
        explanation,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate SHAP explanations for the "
            "production @champion fraud model."
        )
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=(
            "Number of test transactions used for "
            "global SHAP analysis."
        ),
    )

    parser.add_argument(
        "--transaction-index",
        type=int,
        default=0,
        help=(
            "Index within the test set to explain "
            "locally."
        ),
    )

    args = parser.parse_args()

    if args.samples <= 0:
        raise ValueError(
            "--samples must be greater than zero."
        )

    if args.transaction_index < 0:
        raise ValueError(
            "--transaction-index cannot be negative."
        )

    print(
        "\n=========================================="
    )
    print(
        "========== SHAP ANALYSIS ================"
    )
    print(
        "=========================================="
    )

    # --------------------------------------------------------
    # Required files
    # --------------------------------------------------------

    require_file(DATA_PATH)
    require_file(PREPROCESSOR_PATH)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # MLflow
    # --------------------------------------------------------

    print(
        "\n========== CONFIGURING MLFLOW =========="
    )

    configure_mlflow()

    # --------------------------------------------------------
    # Production model
    # --------------------------------------------------------

    print(
        "\n========== LOADING PRODUCTION MODEL ====="
    )

    model, champion = get_champion_model()

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print(
        "\n========== LOADING DATA ================="
    )

    data = pd.read_csv(
        DATA_PATH
    )

    print(
        f"Dataset shape: {data.shape}"
    )

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = split_data(
        data
    )

    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    print(
        "\n========== PREPROCESSING ================"
    )

    (
        X_train_encoded,
        X_val_encoded,
        X_test_encoded,
        preprocessor,
    ) = process_data(
        X_train,
        X_val,
        X_test,
    )

    encoded_count = (
        X_test_encoded.shape[1]
    )

    print(
        f"Encoded features: {encoded_count}"
    )

    if encoded_count != EXPECTED_ENCODED_FEATURE_COUNT:
        raise ValueError(
            "Expected "
            f"{EXPECTED_ENCODED_FEATURE_COUNT} "
            "encoded features, but received "
            f"{encoded_count}."
        )

    feature_names = get_feature_names(
        preprocessor,
        encoded_count,
    )

    print(
        f"Feature names recovered: "
        f"{len(feature_names)}"
    )

    # IMPORTANT:
    # X_test_encoded now exists, so validation of the requested
    # transaction index must happen AFTER preprocessing.
    test_size = X_test_encoded.shape[0]

    if args.transaction_index >= test_size:
        raise IndexError(
            "transaction-index is outside the "
            f"test set. Test size: {test_size}"
        )

    # --------------------------------------------------------
    # Global SHAP sample
    # --------------------------------------------------------

    sample_size = min(
        args.samples,
        test_size
    )

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    sample_indices = rng.choice(
        test_size,
        size=sample_size,
        replace=False,
    )

    X_sample_sparse = (
        X_test_encoded[
            sample_indices
        ]
    )

    # Convert ONLY the selected sample to dense.
    X_sample = to_dense(
        X_sample_sparse
    )

    print(
        "\n========== SHAP EXPLAINER ==============="
    )

    explainer = make_shap_explainer(
        model
    )

    print(
        f"Calculating SHAP values for "
        f"{sample_size} transactions..."
    )

    shap_values = explainer.shap_values(
        X_sample
    )

    shap_values = normalize_shap_values(
        shap_values
    )

    print(
        f"SHAP matrix shape: "
        f"{shap_values.shape}"
    )

    # --------------------------------------------------------
    # Global plots
    # --------------------------------------------------------

    print(
        "\n========== GLOBAL EXPLANATION ==========="
    )

    summary_path = create_global_summary(
        shap_values,
        X_sample,
        feature_names,
    )

    bar_path = create_global_bar(
        shap_values,
        X_sample,
        feature_names,
    )

    print(
        f"Summary plot: {summary_path}"
    )

    print(
        f"Bar plot:     {bar_path}"
    )

    # --------------------------------------------------------
    # Local explanation
    # --------------------------------------------------------

    print(
        "\n========== LOCAL EXPLANATION ============"
    )

    X_row_sparse = (
        X_test_encoded[
            args.transaction_index:
            args.transaction_index + 1
        ]
    )

    # Convert ONLY the requested transaction to dense.
    X_row = to_dense(
        X_row_sparse
    )

    (
        local_plot_path,
        local_json_path,
        explanation,
    ) = create_local_explanation(
        model,
        explainer,
        X_row,
        feature_names,
        args.transaction_index,
    )

    print(
        f"Transaction index: "
        f"{args.transaction_index}"
    )

    print(
        f"Fraud probability: "
        f"{explanation['fraud_probability']:.6f}"
    )

    print(
        f"Prediction: "
        f"{explanation['prediction']}"
    )

    print(
        "\nTop risk factors:"
    )

    for feature in explanation[
        "top_features"
    ][:10]:

        print(
            f"  {feature['feature']}: "
            f"{feature['shap_value']:+.6f} "
            f"({feature['direction']})"
        )

    print(
        f"\nLocal plot: {local_plot_path}"
    )

    print(
        f"Local JSON:  {local_json_path}"
    )

    # --------------------------------------------------------
    # Global metadata
    # --------------------------------------------------------

    global_importance = (
        np.mean(
            np.abs(shap_values),
            axis=0,
        )
    )

    order = np.argsort(
        global_importance
    )[::-1]

    top_global_features = []

    for position in order[:50]:

        top_global_features.append({
            "feature": str(
                feature_names[position]
            ),
            "mean_absolute_shap": float(
                global_importance[position]
            ),
        })

    metadata = {
        "model_name":
            REGISTERED_MODEL_NAME,

        "model_alias":
            CHAMPION_ALIAS,

        "model_version":
            int(champion.version),

        "run_id":
            str(champion.run_id),

        "dataset":
            str(DATA_PATH),

        "encoded_feature_count":
            int(encoded_count),

        "global_sample_count":
            int(sample_size),

        "random_state":
            RANDOM_STATE,

        "production_threshold":
            PRODUCTION_THRESHOLD,

        "top_global_features":
            top_global_features,
    }

    metadata_path = (
        OUTPUT_DIR
        / "shap_metadata.json"
    )

    save_json(
        metadata_path,
        metadata
    )

    print(
        f"\nMetadata: {metadata_path}"
    )

    print(
        "\n=========================================="
    )
    print(
        "========== SHAP COMPLETE ================"
    )
    print(
        "=========================================="
    )


if __name__ == "__main__":
    main()