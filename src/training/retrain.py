import argparse
import json
import os
import shutil
from datetime import datetime, timezone

import mlflow
import mlflow.xgboost
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data.split_data import split_data
from src.data.model_processing import process_data
from src.models.xgboost_model import train_xgboost
from src.utils.mlflow_config import configure_mlflow


# ============================================================
# PATHS
# ============================================================

DATA_PATH = "data/processed/feature_engineered.csv"

XGB_PARAMS_PATH = "models/best_xgboost_params.json"
ENSEMBLE_CONFIG_PATH = "models/best_ensemble_config.json"

XGB_MODEL_PATH = "models/retrained_xgboost_model.json"

PREPROCESSOR_PATH = "models/preprocessor.pkl"

REGISTERED_MODEL_NAME = "AI-Risk-Manager-XGBoost"
CHAMPION_ALIAS = "champion"

# These are the same quality gates used by promote_model.py.
MIN_VALIDATION_PR_AUC = 0.80
MIN_VALIDATION_F1 = 0.70

# Reporting only. Test data is NEVER used to decide promotion.
FALSE_POSITIVE_COST = 100
FALSE_NEGATIVE_COST = 5000


# ============================================================
# HELPERS
# ============================================================

def require_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )


def load_json(path):
    require_file(path)

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def calculate_metrics(y_true, probabilities, threshold=0.5):
    predictions = (
        probabilities >= threshold
    ).astype(int)

    return {
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y_true,
                probabilities
            )
        ),
    }


def find_best_validation_threshold(y_true, probabilities):
    best_threshold = 0.50
    best_metrics = None

    for threshold in [
        round(x, 2)
        for x in list(
            __import__("numpy").arange(
                0.10, 0.91, 0.05
            )
        )
    ]:
        metrics = calculate_metrics(
            y_true,
            probabilities,
            threshold
        )

        if (
            best_metrics is None
            or metrics["f1"] > best_metrics["f1"]
        ):
            best_threshold = threshold
            best_metrics = metrics

    return float(best_threshold), best_metrics


def calculate_costs(
    y_true,
    probabilities,
    threshold,
):
    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn = int(
        ((y_true == 0) & (predictions == 0)).sum()
    )
    fp = int(
        ((y_true == 0) & (predictions == 1)).sum()
    )
    fn = int(
        ((y_true == 1) & (predictions == 0)).sum()
    )
    tp = int(
        ((y_true == 1) & (predictions == 1)).sum()
    )

    total_cost = int(
        fp * FALSE_POSITIVE_COST
        + fn * FALSE_NEGATIVE_COST
    )

    return {
        "threshold": float(threshold),
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "total_cost": total_cost,
    }


def get_current_champion(client):
    try:
        versions = client.search_model_versions(
            f"name='{REGISTERED_MODEL_NAME}'"
        )
    except Exception:
        return None

    for version in versions:
        aliases = list(
            getattr(version, "aliases", []) or []
        )

        if CHAMPION_ALIAS in aliases:
            return version

    return None


def get_run_metrics(client, run_id):
    run = client.get_run(run_id)
    return run.data.metrics


def champion_is_better_or_equal(candidate, champion):
    """
    Candidate wins only if it is not worse on either
    validation PR-AUC or validation F1, and is strictly
    better on at least one.

    This prevents a retraining run from replacing a
    stronger production model.
    """

    candidate_pr = float(
        candidate["validation_pr_auc"]
    )
    candidate_f1 = float(
        candidate["validation_f1"]
    )

    champion_pr = float(
        champion["validation_pr_auc"]
    )
    champion_f1 = float(
        champion["validation_f1"]
    )

    not_worse = (
        candidate_pr >= champion_pr
        and candidate_f1 >= champion_f1
    )

    strictly_better = (
        candidate_pr > champion_pr
        or candidate_f1 > champion_f1
    )

    return not_worse and strictly_better


def safe_log_params(params):
    """
    MLflow parameters must be primitive values.
    """

    cleaned = {}

    for key, value in params.items():

        if value is None:
            continue

        if isinstance(value, (str, int, float, bool)):
            cleaned[key] = value

        else:
            cleaned[key] = str(value)

    if cleaned:
        mlflow.log_params(cleaned)


# ============================================================
# MAIN RETRAINING PIPELINE
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Retrain the production XGBoost model, "
            "evaluate it against the current champion, "
            "register it in MLflow, and promote only "
            "if quality gates and champion comparison pass."
        )
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Allow promotion when the candidate passes "
            "quality gates even if it does not beat the "
            "current champion."
        )
    )

    args = parser.parse_args()

    print(
        "\n=========================================="
    )
    print(
        "========== PRODUCTION RETRAIN ==========="
    )
    print(
        "=========================================="
    )

    # ========================================================
    # CHECK FILES
    # ========================================================

    print(
        "\n========== CHECKING REQUIRED FILES =========="
    )

    require_file(DATA_PATH)
    require_file(XGB_PARAMS_PATH)
    require_file(ENSEMBLE_CONFIG_PATH)

    print("Dataset found.")
    print("Best XGBoost parameters found.")
    print("Ensemble configuration found.")

    # ========================================================
    # MLFLOW
    # ========================================================

    print(
        "\n========== CONFIGURING MLFLOW =========="
    )

    tracking_uri = configure_mlflow()

    print(
        f"MLflow tracking URI configured."
    )

    # ========================================================
    # LOAD CONFIGURATION
    # ========================================================

    xgb_result = load_json(
        XGB_PARAMS_PATH
    )

    ensemble_config = load_json(
        ENSEMBLE_CONFIG_PATH
    )

    xgb_params = xgb_result["best_params"]

    production_threshold = float(
        ensemble_config.get(
            "production_threshold",
            0.70
        )
    )

    print(
        "\n========== RETRAIN CONFIGURATION =========="
    )

    print(
        json.dumps(
            xgb_params,
            indent=4
        )
    )

    print(
        f"Production threshold: "
        f"{production_threshold:.2f}"
    )

    # ========================================================
    # DATA
    # ========================================================

    print(
        "\n========== DATA LOADING =========="
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
        y_test
    ) = split_data(
        data
    )

    print(
        f"Train:      {X_train.shape}"
    )
    print(
        f"Validation: {X_val.shape}"
    )
    print(
        f"Test:       {X_test.shape}"
    )

    print(
        "\nFraud rates:"
    )

    print(
        f"Train:      {y_train.mean() * 100:.4f}%"
    )
    print(
        f"Validation: {y_val.mean() * 100:.4f}%"
    )
    print(
        f"Test:       {y_test.mean() * 100:.4f}%"
    )

    # ========================================================
    # PREPROCESSING
    # ========================================================

    print(
        "\n========== MODEL PREPROCESSING =========="
    )

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

    print(
        f"Encoded features: "
        f"{X_train_encoded.shape[1]}"
    )

    # ========================================================
    # CURRENT CHAMPION
    # ========================================================

    client = mlflow.MlflowClient()

    champion = get_current_champion(
        client
    )

    if champion is None:
        print(
            "\nCurrent champion: NONE"
        )
    else:
        print(
            "\n========== CURRENT CHAMPION =========="
        )
        print(
            f"Version: {champion.version}"
        )
        print(
            f"Run ID:  {champion.run_id}"
        )
        print(
            f"Aliases: {champion.aliases}"
        )

    # ========================================================
    # RETRAIN
    # ========================================================

    run_name = (
        "production_retrain_"
        + datetime.now(
            timezone.utc
        ).strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    print(
        "\n========== TRAINING RETRAINED MODEL =========="
    )

    with mlflow.start_run(
        run_name=run_name
    ) as run:

        mlflow.set_tags({
            "model_type": "XGBoost",
            "stage": "retraining",
            "retraining": "true",
            "test_set_used_for_selection": "false",
            "registered_model": REGISTERED_MODEL_NAME,
            "candidate_status": "production-candidate",
        })

        # ----------------------------------------------------
        # Dataset metadata
        # ----------------------------------------------------

        safe_log_params({
            "dataset": os.path.basename(DATA_PATH),
            "total_samples": len(data),
            "train_samples": len(X_train),
            "validation_samples": len(X_val),
            "test_samples": len(X_test),
            "original_features": X_train.shape[1],
            "encoded_features": X_train_encoded.shape[1],
            "train_fraud_rate": float(y_train.mean()),
            "validation_fraud_rate": float(y_val.mean()),
            "test_fraud_rate": float(y_test.mean()),
        })

        # ----------------------------------------------------
        # XGBoost parameters
        # ----------------------------------------------------

        safe_log_params({
            f"xgb_{key}": value
            for key, value in xgb_params.items()
        })

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        model = train_xgboost(
            X_train_encoded,
            y_train,
            X_val_encoded,
            y_val,
            model_path=XGB_MODEL_PATH,
            **xgb_params
        )

        # ----------------------------------------------------
        # Probabilities
        # ----------------------------------------------------

        validation_probabilities = (
            model.predict_proba(
                X_val_encoded
            )[:, 1]
        )

        test_probabilities = (
            model.predict_proba(
                X_test_encoded
            )[:, 1]
        )

        # ----------------------------------------------------
        # Validation metrics
        # ----------------------------------------------------

        validation_metrics = calculate_metrics(
            y_val,
            validation_probabilities,
            production_threshold
        )

        # ----------------------------------------------------
        # Validation F1 threshold
        # ----------------------------------------------------

        (
            best_validation_threshold,
            best_validation_threshold_metrics
        ) = find_best_validation_threshold(
            y_val,
            validation_probabilities
        )

        # ----------------------------------------------------
        # Test metrics
        #
        # Reporting only. Never used for promotion.
        # ----------------------------------------------------

        test_metrics = calculate_metrics(
            y_test,
            test_probabilities,
            production_threshold
        )

        # ----------------------------------------------------
        # Cost analysis
        #
        # Reporting only.
        # ----------------------------------------------------

        cost_analysis = calculate_costs(
            y_test,
            test_probabilities,
            production_threshold
        )

        # ----------------------------------------------------
        # MLflow metrics
        # ----------------------------------------------------

        mlflow.log_metrics({

            "validation_accuracy":
                validation_metrics["accuracy"],

            "validation_precision":
                validation_metrics["precision"],

            "validation_recall":
                validation_metrics["recall"],

            "validation_f1":
                validation_metrics["f1"],

            "validation_roc_auc":
                validation_metrics["roc_auc"],

            "validation_pr_auc":
                validation_metrics["pr_auc"],

            "validation_best_f1_threshold":
                best_validation_threshold,

            "validation_best_f1":
                best_validation_threshold_metrics["f1"],

            "test_accuracy":
                test_metrics["accuracy"],

            "test_precision":
                test_metrics["precision"],

            "test_recall":
                test_metrics["recall"],

            "test_f1":
                test_metrics["f1"],

            "test_roc_auc":
                test_metrics["roc_auc"],

            "test_pr_auc":
                test_metrics["pr_auc"],

            "test_false_positives":
                cost_analysis["false_positive"],

            "test_false_negatives":
                cost_analysis["false_negative"],

            "test_total_cost":
                cost_analysis["total_cost"],
        })

        safe_log_params({
            "production_threshold":
                production_threshold,
            "false_positive_cost":
                FALSE_POSITIVE_COST,
            "false_negative_cost":
                FALSE_NEGATIVE_COST,
        })

        # ----------------------------------------------------
        # Log artifacts
        # ----------------------------------------------------

        if os.path.exists(XGB_MODEL_PATH):
            mlflow.log_artifact(
                XGB_MODEL_PATH,
                artifact_path="xgboost_model"
            )

        if os.path.exists(PREPROCESSOR_PATH):
            mlflow.log_artifact(
                PREPROCESSOR_PATH,
                artifact_path="preprocessor"
            )

        if os.path.exists(ENSEMBLE_CONFIG_PATH):
            mlflow.log_artifact(
                ENSEMBLE_CONFIG_PATH,
                artifact_path="ensemble_config"
            )

        # ----------------------------------------------------
        # Log native XGBoost model
        # ----------------------------------------------------

        model_info = mlflow.xgboost.log_model(
            model,
            artifact_path="model"
        )

        run_id = run.info.run_id

        print(
            "\n========== RETRAIN RESULTS =========="
        )

        print(
            f"Validation PR-AUC: "
            f"{validation_metrics['pr_auc']:.6f}"
        )

        print(
            f"Validation F1: "
            f"{validation_metrics['f1']:.6f}"
        )

        print(
            f"Test PR-AUC: "
            f"{test_metrics['pr_auc']:.6f}"
        )

        print(
            f"Test F1: "
            f"{test_metrics['f1']:.6f}"
        )

        # ====================================================
        # QUALITY GATES
        # ====================================================

        pr_auc_passed = (
            validation_metrics["pr_auc"]
            >= MIN_VALIDATION_PR_AUC
        )

        f1_passed = (
            validation_metrics["f1"]
            >= MIN_VALIDATION_F1
        )

        print(
            "\n========== QUALITY GATES =========="
        )

        print(
            f"Validation PR-AUC: "
            f"{validation_metrics['pr_auc']:.6f} "
            f"(required >= {MIN_VALIDATION_PR_AUC:.2f}) "
            f"{'PASS' if pr_auc_passed else 'FAIL'}"
        )

        print(
            f"Validation F1: "
            f"{validation_metrics['f1']:.6f} "
            f"(required >= {MIN_VALIDATION_F1:.2f}) "
            f"{'PASS' if f1_passed else 'FAIL'}"
        )

        quality_gates_passed = (
            pr_auc_passed
            and f1_passed
        )

        mlflow.set_tag(
            "quality_gates",
            "passed"
            if quality_gates_passed
            else "failed"
        )

        if not quality_gates_passed:
            mlflow.set_tag(
                "candidate_status",
                "rejected_quality_gate"
            )

            print(
                "\nCandidate FAILED quality gates."
            )

            print(
                "Model will NOT be registered or promoted."
            )

            return

        print(
            "\nAll quality gates PASSED."
        )

        # ====================================================
        # CHAMPION COMPARISON
        # ====================================================

        candidate_metrics = {
            "validation_pr_auc":
                validation_metrics["pr_auc"],
            "validation_f1":
                validation_metrics["f1"],
        }

        should_promote = False

        if champion is None:

            should_promote = True

            print(
                "\nNo current champion exists."
            )

        else:

            champion_metrics_raw = get_run_metrics(
                client,
                champion.run_id
            )

            champion_pr_auc = champion_metrics_raw.get(
                "validation_pr_auc"
            )

            champion_f1 = champion_metrics_raw.get(
                "validation_f1"
            )

            if (
                champion_pr_auc is None
                or champion_f1 is None
            ):
                raise RuntimeError(
                    "Current champion does not contain "
                    "validation_pr_auc and validation_f1 "
                    "metrics. Refusing automatic promotion."
                )

            champion_metrics = {
                "validation_pr_auc":
                    float(champion_pr_auc),
                "validation_f1":
                    float(champion_f1),
            }

            print(
                "\n========== CHAMPION COMPARISON =========="
            )

            print(
                f"Current champion PR-AUC: "
                f"{champion_metrics['validation_pr_auc']:.6f}"
            )

            print(
                f"Candidate PR-AUC: "
                f"{candidate_metrics['validation_pr_auc']:.6f}"
            )

            print(
                f"Current champion F1: "
                f"{champion_metrics['validation_f1']:.6f}"
            )

            print(
                f"Candidate F1: "
                f"{candidate_metrics['validation_f1']:.6f}"
            )

            should_promote = (
                champion_is_better_or_equal(
                    candidate_metrics,
                    champion_metrics
                )
            )

            if should_promote:
                print(
                    "\nCandidate is better than "
                    "the current champion."
                )
            else:
                print(
                    "\nCandidate did not beat "
                    "the current champion."
                )

        # ====================================================
        # REGISTER CANDIDATE
        # ====================================================

        print(
            "\n========== REGISTERING CANDIDATE =========="
        )

        try:
            registered = mlflow.register_model(
                model_uri=model_info.model_uri,
                name=REGISTERED_MODEL_NAME
            )

        except Exception as exc:
            raise RuntimeError(
                "Model passed quality gates but "
                "registration failed."
            ) from exc

        candidate_version = int(
            registered.version
        )

        print(
            f"Registered model: "
            f"{REGISTERED_MODEL_NAME}"
        )

        print(
            f"Candidate version: "
            f"{candidate_version}"
        )

        # ----------------------------------------------------
        # Add candidate metadata to registry version
        # ----------------------------------------------------

        client.set_model_version_tag(
            REGISTERED_MODEL_NAME,
            str(candidate_version),
            "candidate_status",
            (
                "promotion-ready"
                if should_promote
                else "registered-not-promoted"
            )
        )

        client.set_model_version_tag(
            REGISTERED_MODEL_NAME,
            str(candidate_version),
            "validation_pr_auc",
            str(
                validation_metrics["pr_auc"]
            )
        )

        client.set_model_version_tag(
            REGISTERED_MODEL_NAME,
            str(candidate_version),
            "validation_f1",
            str(
                validation_metrics["f1"]
            )
        )

        client.set_model_version_tag(
            REGISTERED_MODEL_NAME,
            str(candidate_version),
            "production_threshold",
            str(
                production_threshold
            )
        )

        # ====================================================
        # PROMOTION
        # ====================================================

        if should_promote or args.force:

            if args.force and not should_promote:
                print(
                    "\nWARNING: --force supplied."
                )
                print(
                    "Candidate is being promoted "
                    "despite not beating champion."
                )

            print(
                "\n========== PROMOTING CANDIDATE =========="
            )

            client.set_registered_model_alias(
                REGISTERED_MODEL_NAME,
                CHAMPION_ALIAS,
                str(candidate_version)
            )

            # Verify alias after assignment.
            promoted = client.get_model_version_by_alias(
                REGISTERED_MODEL_NAME,
                CHAMPION_ALIAS
            )

            if str(promoted.version) != str(
                candidate_version
            ):
                raise RuntimeError(
                    "Promotion verification failed."
                )

            print(
                "\n=========================================="
            )
            print(
                "====== RETRAIN + PROMOTION COMPLETE ====="
            )
            print(
                "=========================================="
            )

            print(
                f"Production model: "
                f"{REGISTERED_MODEL_NAME}"
            )

            print(
                f"Production version: "
                f"{candidate_version}"
            )

            print(
                f"Production alias: "
                f"@{CHAMPION_ALIAS}"
            )

            print(
                f"Model URI: "
                f"models:/{REGISTERED_MODEL_NAME}"
                f"@{CHAMPION_ALIAS}"
            )

        else:

            print(
                "\n========== PROMOTION SKIPPED =========="
            )

            print(
                f"Candidate version {candidate_version} "
                "was registered."
            )

            print(
                "Current champion remains unchanged."
            )

            print(
                "\n=========================================="
            )
            print(
                "========= RETRAIN COMPLETE ============="
            )
            print(
                "=========================================="
            )

        print(
            f"\nMLflow run ID: {run_id}"
        )

        print(
            f"Validation PR-AUC: "
            f"{validation_metrics['pr_auc']:.6f}"
        )

        print(
            f"Validation F1: "
            f"{validation_metrics['f1']:.6f}"
        )

        print(
            f"Test PR-AUC: "
            f"{test_metrics['pr_auc']:.6f}"
        )

        print(
            f"Test F1: "
            f"{test_metrics['f1']:.6f}"
        )


if __name__ == "__main__":
    main()