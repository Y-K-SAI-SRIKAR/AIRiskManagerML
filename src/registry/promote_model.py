
from pathlib import Path
import argparse
import json

import mlflow
from mlflow.tracking import MlflowClient

from src.utils.mlflow_config import configure_mlflow


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ENSEMBLE_CONFIG_PATH = (
    PROJECT_ROOT
    / "models"
    / "best_ensemble_config.json"
)


# ============================================================
# MLFLOW CONFIGURATION
# ============================================================

MODEL_NAME = "AI-Risk-Manager-XGBoost"

EXPERIMENT_NAME = "AI-Risk-Manager"

PRODUCTION_ALIAS = "champion"


# ============================================================
# QUALITY GATES
# ============================================================

MIN_VALIDATION_PR_AUC = 0.80

MIN_VALIDATION_F1 = 0.70


# ============================================================
# ARGUMENTS
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Promote a validated XGBoost model "
            "version to the MLflow champion alias."
        )
    )

    parser.add_argument(
        "--version",
        type=int,
        required=True,
        help="MLflow model version to promote.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Bypass quality gates. "
            "Use only when explicitly required."
        ),
    )

    return parser.parse_args()


# ============================================================
# LOAD ENSEMBLE CONFIGURATION
# ============================================================

def load_ensemble_config():

    if not ENSEMBLE_CONFIG_PATH.exists():

        raise FileNotFoundError(
            "Final ensemble configuration not found:\n"
            f"{ENSEMBLE_CONFIG_PATH}"
        )

    with open(
        ENSEMBLE_CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        config = json.load(file)

    return config


# ============================================================
# VALIDATE CONFIGURATION
# ============================================================

def validate_configuration(config):

    required_fields = [
        "model_type",
        "xgb_weight",
        "nn_weight",
        "production_threshold",
        "selection_metric",
        "threshold_selection_metric",
        "validation_pr_auc",
        "validation_precision",
        "validation_recall",
        "validation_f1",
        "test_pr_auc",
        "test_f1",
        "test_precision",
        "test_recall",
    ]

    missing = [
        field
        for field in required_fields
        if field not in config
    ]

    if missing:

        raise ValueError(
            "Missing required fields in "
            "best_ensemble_config.json:\n"
            + "\n".join(
                f"  - {field}"
                for field in missing
            )
        )


# ============================================================
# DISPLAY CONFIGURATION
# ============================================================

def display_configuration(config):

    print(
        "\n========== MODEL EVALUATION =========="
    )

    print(
        f"Model type: "
        f"{config['model_type']}"
    )

    print(
        f"XGBoost weight: "
        f"{config['xgb_weight']}"
    )

    print(
        f"Neural Network weight: "
        f"{config['nn_weight']}"
    )

    print(
        f"Production threshold: "
        f"{config['production_threshold']}"
    )

    print(
        "\nValidation metrics:"
    )

    print(
        f"  PR-AUC: "
        f"{config['validation_pr_auc']:.6f}"
    )

    print(
        f"  Precision: "
        f"{config['validation_precision']:.6f}"
    )

    print(
        f"  Recall: "
        f"{config['validation_recall']:.6f}"
    )

    print(
        f"  F1: "
        f"{config['validation_f1']:.6f}"
    )

    print(
        "\nTest metrics:"
    )

    print(
        f"  PR-AUC: "
        f"{config['test_pr_auc']:.6f}"
    )

    print(
        f"  Precision: "
        f"{config['test_precision']:.6f}"
    )

    print(
        f"  Recall: "
        f"{config['test_recall']:.6f}"
    )

    print(
        f"  F1: "
        f"{config['test_f1']:.6f}"
    )


# ============================================================
# QUALITY GATES
# ============================================================

def validate_quality_gates(
    config,
    force=False,
):

    validation_pr_auc = float(
        config["validation_pr_auc"]
    )

    validation_f1 = float(
        config["validation_f1"]
    )

    print(
        "\n========== QUALITY GATES =========="
    )

    print(
        f"Validation PR-AUC:"
        f" {validation_pr_auc:.6f}"
    )

    print(
        f"Required PR-AUC:"
        f" {MIN_VALIDATION_PR_AUC:.6f}"
    )

    print(
        f"Validation F1:"
        f" {validation_f1:.6f}"
    )

    print(
        f"Required F1:"
        f" {MIN_VALIDATION_F1:.6f}"
    )

    if force:

        print(
            "\nWARNING:"
            " --force supplied."
        )

        print(
            "Quality gates are being bypassed."
        )

        return

    if validation_pr_auc < (
        MIN_VALIDATION_PR_AUC
    ):

        raise RuntimeError(
            "\nPROMOTION REJECTED\n"
            "Validation PR-AUC quality gate failed.\n"
            f"Required: {MIN_VALIDATION_PR_AUC}\n"
            f"Actual:   {validation_pr_auc}"
        )

    if validation_f1 < (
        MIN_VALIDATION_F1
    ):

        raise RuntimeError(
            "\nPROMOTION REJECTED\n"
            "Validation F1 quality gate failed.\n"
            f"Required: {MIN_VALIDATION_F1}\n"
            f"Actual:   {validation_f1}"
        )

    print(
        "\nAll quality gates PASSED."
    )


# ============================================================
# MLFLOW RUN VERIFICATION
# ============================================================

def verify_registered_model(
    client,
    version,
    config,
):

    print(
        "\n========== MLFLOW MODEL VERIFICATION =========="
    )

    model_version = (
        client.get_model_version(
            name=MODEL_NAME,
            version=str(version),
        )
    )

    print(
        f"Registered model:"
        f" {model_version.name}"
    )

    print(
        f"Version:"
        f" {model_version.version}"
    )

    print(
        f"Run ID:"
        f" {model_version.run_id}"
    )

    print(
        f"Status:"
        f" {model_version.status}"
    )

    print(
        f"Aliases:"
        f" {model_version.aliases}"
    )

    if str(model_version.status).upper() != "READY":

        raise RuntimeError(
            "Model version is not READY."
        )

    if not model_version.run_id:

        raise RuntimeError(
            "Registered model version does not "
            "have an MLflow run ID."
        )

    # --------------------------------------------------------
    # GET ASSOCIATED RUN
    # --------------------------------------------------------

    run = client.get_run(
        model_version.run_id
    )

    print(
        "\nAssociated MLflow run found."
    )

    print(
        f"Run ID: {run.info.run_id}"
    )

    # --------------------------------------------------------
    # VERIFY MODEL TAG
    # --------------------------------------------------------

    model_status = (
        model_version.tags.get(
            "model_status"
        )
    )

    print(
        f"Model status tag:"
        f" {model_status}"
    )

    if model_status not in {
        "production-candidate",
        "candidate",
        "validated",
    }:

        raise RuntimeError(
            "Model version is not marked as "
            "a production candidate."
        )

    # --------------------------------------------------------
    # VERIFY CONFIGURATION VALUES
    # --------------------------------------------------------

    expected_feature_count = (
        model_version.tags.get(
            "feature_count"
        )
    )

    if expected_feature_count:

        print(
            f"Feature count:"
            f" {expected_feature_count}"
        )

        if expected_feature_count != "422":

            raise RuntimeError(
                "Feature count mismatch. "
                f"Expected 422, got "
                f"{expected_feature_count}."
            )

    # --------------------------------------------------------
    # CHECK RUN METRICS IF AVAILABLE
    # --------------------------------------------------------

    run_metrics = run.data.metrics

    print(
        "\nMLflow run metrics:"
    )

    for name, value in run_metrics.items():

        print(
            f"  {name}: {value}"
        )

    # If validation PR-AUC was logged into MLflow,
    # verify it agrees with the final configuration.
    mlflow_validation_pr_auc = (
        run_metrics.get(
            "validation_pr_auc"
        )
    )

    if mlflow_validation_pr_auc is not None:

        config_pr_auc = float(
            config["validation_pr_auc"]
        )

        if abs(
            float(
                mlflow_validation_pr_auc
            )
            - config_pr_auc
        ) > 1e-6:

            raise RuntimeError(
                "Validation PR-AUC mismatch between "
                "MLflow run and ensemble configuration."
            )

    return model_version


# ============================================================
# CURRENT CHAMPION
# ============================================================

def get_current_champion(client):

    try:

        champion = (
            client.get_model_version_by_alias(
                MODEL_NAME,
                PRODUCTION_ALIAS,
            )
        )

        return champion

    except Exception:

        return None


# ============================================================
# PROMOTION
# ============================================================

def promote_model(
    version,
    force=False,
):

    print(
        "\n=========================================="
    )

    print(
        "========== MODEL PROMOTION ============="
    )

    print(
        "=========================================="
    )

    print(
        f"\nModel:"
        f" {MODEL_NAME}"
    )

    print(
        f"Candidate version:"
        f" {version}"
    )

    print(
        f"Production alias:"
        f" @{PRODUCTION_ALIAS}"
    )

    # --------------------------------------------------------
    # LOAD FINAL CONFIG
    # --------------------------------------------------------

    config = load_ensemble_config()

    validate_configuration(
        config
    )

    display_configuration(
        config
    )

    # --------------------------------------------------------
    # QUALITY GATES
    # --------------------------------------------------------

    validate_quality_gates(
        config,
        force=force,
    )

    # --------------------------------------------------------
    # CONFIGURE MLFLOW
    # --------------------------------------------------------

    print(
        "\n========== CONFIGURING MLFLOW =========="
    )

    tracking_uri = configure_mlflow()

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    client = MlflowClient(
        tracking_uri=tracking_uri
    )

    print(
        "MLflow connection successful."
    )

    # --------------------------------------------------------
    # VERIFY MODEL
    # --------------------------------------------------------

    model_version = (
        verify_registered_model(
            client,
            version,
            config,
        )
    )

    # --------------------------------------------------------
    # CURRENT CHAMPION
    # --------------------------------------------------------

    print(
        "\n========== CURRENT CHAMPION =========="
    )

    previous_champion = (
        get_current_champion(
            client
        )
    )

    if previous_champion:

        print(
            f"Current champion:"
            f" version {previous_champion.version}"
        )

    else:

        print(
            "No current champion found."
        )

    # --------------------------------------------------------
    # PROMOTE
    # --------------------------------------------------------

    print(
        "\n========== PROMOTING MODEL =========="
    )

    client.set_registered_model_alias(
        MODEL_NAME,
        PRODUCTION_ALIAS,
        str(version),
    )

    # --------------------------------------------------------
    # UPDATE VERSION TAGS
    # --------------------------------------------------------

    client.set_model_version_tag(
        MODEL_NAME,
        str(version),
        "model_status",
        "production",
    )

    client.set_model_version_tag(
        MODEL_NAME,
        str(version),
        "deployment_alias",
        PRODUCTION_ALIAS,
    )

    client.set_model_version_tag(
        MODEL_NAME,
        str(version),
        "promotion_method",
        "quality-gate",
    )

    client.set_model_version_tag(
        MODEL_NAME,
        str(version),
        "production_threshold",
        str(
            config["production_threshold"]
        ),
    )

    client.set_model_version_tag(
        MODEL_NAME,
        str(version),
        "validation_pr_auc",
        str(
            config["validation_pr_auc"]
        ),
    )

    client.set_model_version_tag(
        MODEL_NAME,
        str(version),
        "validation_f1",
        str(
            config["validation_f1"]
        ),
    )

    # --------------------------------------------------------
    # REGISTERED MODEL TAGS
    # --------------------------------------------------------

    client.set_registered_model_tag(
        MODEL_NAME,
        "production_alias",
        PRODUCTION_ALIAS,
    )

    client.set_registered_model_tag(
        MODEL_NAME,
        "production_version",
        str(version),
    )

    client.set_registered_model_tag(
        MODEL_NAME,
        "production_threshold",
        str(
            config["production_threshold"]
        ),
    )

    # --------------------------------------------------------
    # VERIFY ALIAS
    # --------------------------------------------------------

    promoted = (
        client.get_model_version_by_alias(
            MODEL_NAME,
            PRODUCTION_ALIAS,
        )
    )

    if str(promoted.version) != str(
        version
    ):

        raise RuntimeError(
            "Production alias verification failed."
        )

    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    print(
        "\n=========================================="
    )

    print(
        "========= MODEL PROMOTION COMPLETE ======="
    )

    print(
        "=========================================="
    )

    print(
        f"\nModel:"
        f" {MODEL_NAME}"
    )

    print(
        f"Production version:"
        f" {version}"
    )

    print(
        f"Production alias:"
        f" @{PRODUCTION_ALIAS}"
    )

    print(
        "\nProduction model URI:"
    )

    print(
        f"models:/{MODEL_NAME}"
        f"@{PRODUCTION_ALIAS}"
    )

    print(
        "\nValidation PR-AUC:"
        f" {config['validation_pr_auc']:.6f}"
    )

    print(
        "Validation F1:"
        f" {config['validation_f1']:.6f}"
    )

    print(
        "Production threshold:"
        f" {config['production_threshold']}"
    )

    if previous_champion:

        print(
            "\nPrevious champion:"
            f" version {previous_champion.version}"
        )

    else:

        print(
            "\nPrevious champion: None"
        )

    print(
        "\nPromotion verified successfully."
    )

    return promoted


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_arguments()

    promote_model(
        version=args.version,
        force=args.force,
    )


if __name__ == "__main__":
    main()