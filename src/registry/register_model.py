from pathlib import Path
import json

import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient
import xgboost as xgb

from src.utils.mlflow_config import configure_mlflow


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "tuned_xgboost_model.json"
)

PREPROCESSOR_PATH = (
    PROJECT_ROOT
    / "models"
    / "preprocessor.pkl"
)

ENSEMBLE_CONFIG_PATH = (
    PROJECT_ROOT
    / "models"
    / "best_ensemble_config.json"
)


# ============================================================
# MLflow CONFIGURATION
# ============================================================

REGISTERED_MODEL_NAME = (
    "AI-Risk-Manager-XGBoost"
)

EXPERIMENT_NAME = (
    "AI-Risk-Manager"
)


# ============================================================
# VALIDATION
# ============================================================

def validate_files():
    """Verify all required model files exist."""

    required_files = {
        "XGBoost model": MODEL_PATH,
        "Preprocessor": PREPROCESSOR_PATH,
        "Ensemble configuration": ENSEMBLE_CONFIG_PATH,
    }

    missing = []

    for name, path in required_files.items():

        if not path.exists():
            missing.append(
                f"{name}: {path}"
            )

    if missing:

        raise FileNotFoundError(
            "Required model files are missing:\n"
            + "\n".join(
                f"  - {item}"
                for item in missing
            )
        )


# ============================================================
# LOAD CONFIGURATION
# ============================================================

def load_ensemble_config():
    """Load the final ensemble/production configuration."""

    if not ENSEMBLE_CONFIG_PATH.exists():
        return {}

    with open(
        ENSEMBLE_CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


# ============================================================
# REGISTER MODEL
# ============================================================

def register_model():

    print("\n==========================================")
    print("========== MODEL REGISTRATION ===========")
    print("==========================================")

    # --------------------------------------------------------
    # VALIDATE FILES
    # --------------------------------------------------------

    validate_files()

    print("\nRequired model files found.")

    print(
        f"\nXGBoost model:\n{MODEL_PATH}"
    )

    print(
        f"\nPreprocessor:\n{PREPROCESSOR_PATH}"
    )

    print(
        f"\nEnsemble config:\n{ENSEMBLE_CONFIG_PATH}"
    )

    # --------------------------------------------------------
    # CONFIGURE MLFLOW
    # --------------------------------------------------------

    print("\n========== CONFIGURING MLFLOW ==========")

    tracking_uri = configure_mlflow()

    print(
        f"MLflow tracking URI configured."
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    # --------------------------------------------------------
    # LOAD XGBOOST MODEL
    # --------------------------------------------------------

    print("\n========== LOADING MODEL ==========")

    model = xgb.XGBClassifier()

    model.load_model(
        str(MODEL_PATH)
    )

    print(
        "Tuned XGBoost model loaded successfully."
    )

    # --------------------------------------------------------
    # LOAD ENSEMBLE CONFIG
    # --------------------------------------------------------

    ensemble_config = (
        load_ensemble_config()
    )

    # --------------------------------------------------------
    # START MLFLOW RUN
    # --------------------------------------------------------

    print("\n========== MLFLOW RUN ==========")

    with mlflow.start_run(
        run_name="register-tuned-xgboost"
    ) as run:

        run_id = run.info.run_id

        # ----------------------------------------------------
        # MODEL PARAMETERS
        # ----------------------------------------------------

        booster = model.get_booster()

        params = model.get_params()

        mlflow.log_params(
            {
                "model_type": "XGBoost",
                "n_estimators":
                    params.get(
                        "n_estimators"
                    ),
                "learning_rate":
                    params.get(
                        "learning_rate"
                    ),
                "max_depth":
                    params.get(
                        "max_depth"
                    ),
                "min_child_weight":
                    params.get(
                        "min_child_weight"
                    ),
                "subsample":
                    params.get(
                        "subsample"
                    ),
                "colsample_bytree":
                    params.get(
                        "colsample_bytree"
                    ),
                "gamma":
                    params.get(
                        "gamma"
                    ),
                "reg_alpha":
                    params.get(
                        "reg_alpha"
                    ),
                "reg_lambda":
                    params.get(
                        "reg_lambda"
                    ),
            }
        )

        # ----------------------------------------------------
        # METRICS FROM FINAL ENSEMBLE CONFIG
        # ----------------------------------------------------

        if ensemble_config:

            metric_mapping = {
                "validation_pr_auc":
                    "validation_pr_auc",

                "test_pr_auc":
                    "test_pr_auc",

                "test_f1":
                    "test_f1",

                "test_precision":
                    "test_precision",

                "test_recall":
                    "test_recall",
            }

            metrics = {}

            for config_key, metric_key in (
                metric_mapping.items()
            ):

                value = ensemble_config.get(
                    config_key
                )

                if value is not None:

                    try:
                        metrics[metric_key] = float(
                            value
                        )
                    except (
                        TypeError,
                        ValueError,
                    ):
                        pass

            if metrics:
                mlflow.log_metrics(
                    metrics
                )

        # ----------------------------------------------------
        # MODEL TAGS
        # ----------------------------------------------------

        mlflow.set_tags(
            {
                "project":
                    "AI-Risk-Manager",

                "model":
                    "Tuned XGBoost",

                "model_status":
                    "production-candidate",

                "feature_count":
                    "422",

                "raw_feature_count":
                    "117",

                "categorical_features":
                    "25",

                "numeric_features":
                    "92",

                "model_source":
                    "models/tuned_xgboost_model.json",
            }
        )

        # ----------------------------------------------------
        # LOG PREPROCESSOR
        # ----------------------------------------------------

        mlflow.log_artifact(
            str(PREPROCESSOR_PATH),
            artifact_path="preprocessing",
        )

        # ----------------------------------------------------
        # LOG ENSEMBLE CONFIG
        # ----------------------------------------------------

        mlflow.log_artifact(
            str(ENSEMBLE_CONFIG_PATH),
            artifact_path="configuration",
        )

        # ----------------------------------------------------
        # LOG XGBOOST MODEL
        # ----------------------------------------------------

        model_info = mlflow.xgboost.log_model(
            xgb_model=booster,
            artifact_path="model",
            registered_model_name=(
                REGISTERED_MODEL_NAME
            ),
        )

        print(
            "\nXGBoost model logged successfully."
        )

        print(
            f"Model URI: {model_info.model_uri}"
        )

    # --------------------------------------------------------
    # FIND NEW MODEL VERSION
    # --------------------------------------------------------

    client = MlflowClient(
        tracking_uri=tracking_uri
    )

    versions = client.search_model_versions(
        f"name='{REGISTERED_MODEL_NAME}'"
    )

    if not versions:

        raise RuntimeError(
            "Model was logged but no registered "
            "model version was found."
        )

    # Find the version associated with this run.
    matching_versions = [
        version
        for version in versions
        if version.run_id == run_id
    ]

    if matching_versions:

        latest_version = max(
            matching_versions,
            key=lambda version: int(
                version.version
            ),
        )

    else:

        latest_version = max(
            versions,
            key=lambda version: int(
                version.version
            ),
        )

    # --------------------------------------------------------
    # ADD MODEL VERSION TAGS
    # --------------------------------------------------------

    client.set_model_version_tag(
        REGISTERED_MODEL_NAME,
        latest_version.version,
        "model_status",
        "production-candidate",
    )

    client.set_model_version_tag(
        REGISTERED_MODEL_NAME,
        latest_version.version,
        "source",
        "tuned_xgboost_model",
    )

    client.set_model_version_tag(
        REGISTERED_MODEL_NAME,
        latest_version.version,
        "feature_count",
        "422",
    )

    print("\n==========================================")
    print("========== REGISTRATION COMPLETE =========")
    print("==========================================")

    print(
        f"\nRegistered model:"
        f" {REGISTERED_MODEL_NAME}"
    )

    print(
        f"Model version:"
        f" {latest_version.version}"
    )

    print(
        f"Run ID:"
        f" {run_id}"
    )

    print(
        "\nModel status:"
        " production-candidate"
    )

    return latest_version.version


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    register_model()