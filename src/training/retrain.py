import argparse
import json
import os
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


# ============================================================
# QUALITY GATES
# ============================================================

MIN_VALIDATION_PR_AUC = 0.80

MIN_VALIDATION_PRECISION = 0.70

MIN_VALIDATION_F1 = 0.70

MIN_VALIDATION_RECALL = 0.70


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

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def calculate_metrics(
    y_true,
    probabilities,
    threshold
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return {

        "accuracy":
            float(
                accuracy_score(
                    y_true,
                    predictions
                )
            ),

        "precision":
            float(
                precision_score(
                    y_true,
                    predictions,
                    zero_division=0
                )
            ),

        "recall":
            float(
                recall_score(
                    y_true,
                    predictions,
                    zero_division=0
                )
            ),

        "f1":
            float(
                f1_score(
                    y_true,
                    predictions,
                    zero_division=0
                )
            ),

        "roc_auc":
            float(
                roc_auc_score(
                    y_true,
                    probabilities
                )
            ),

        "pr_auc":
            float(
                average_precision_score(
                    y_true,
                    probabilities
                )
            ),
    }


def get_current_champion(client):

    try:

        return client.get_model_version_by_alias(
            REGISTERED_MODEL_NAME,
            CHAMPION_ALIAS
        )

    except Exception:

        return None


def get_run_metrics(
    client,
    run_id
):

    run = client.get_run(
        run_id
    )

    return run.data.metrics


def champion_is_better_or_equal(
    candidate,
    champion
):
    """
    Recall-first production promotion policy.

    The candidate is eligible for promotion only after the
    quality gates have passed.

    Ranking priority:
        1. Validation recall (primary objective)
        2. Validation PR-AUC (tie-breaker)
        3. Validation F1 (second tie-breaker)

    A candidate with lower recall is never promoted merely because
    its PR-AUC or F1 is higher.

    This makes the production policy explicitly fraud-recall aware
    while the quality gates prevent unacceptable precision/F1/PR-AUC
    regressions.
    """

    candidate_recall = float(
        candidate["validation_recall"]
    )
    champion_recall = float(
        champion["validation_recall"]
    )

    if candidate_recall > champion_recall:
        return True

    if candidate_recall < champion_recall:
        return False

    candidate_pr = float(
        candidate["validation_pr_auc"]
    )
    champion_pr = float(
        champion["validation_pr_auc"]
    )

    if candidate_pr > champion_pr:
        return True

    if candidate_pr < champion_pr:
        return False

    candidate_f1 = float(
        candidate["validation_f1"]
    )
    champion_f1 = float(
        champion["validation_f1"]
    )

    return candidate_f1 > champion_f1


def safe_log_params(params):

    cleaned = {}

    for key, value in params.items():

        if value is None:

            continue

        if isinstance(
            value,
            (
                str,
                int,
                float,
                bool
            )
        ):

            cleaned[key] = value

        else:

            cleaned[key] = str(value)

    if cleaned:

        mlflow.log_params(
            cleaned
        )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Promote a quality-gate-passing candidate "
            "even if it does not beat the champion."
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
    # REQUIRED FILES
    # ========================================================

    print(
        "\n========== CHECKING REQUIRED FILES =========="
    )

    require_file(
        DATA_PATH
    )

    require_file(
        XGB_PARAMS_PATH
    )

    require_file(
        ENSEMBLE_CONFIG_PATH
    )

    print(
        "Dataset found."
    )

    print(
        "Best XGBoost parameters found."
    )

    print(
        "Ensemble configuration found."
    )

    # ========================================================
    # MLFLOW
    # ========================================================

    print(
        "\n========== CONFIGURING MLFLOW =========="
    )

    configure_mlflow()

    print(
        "MLflow tracking URI configured."
    )

    # ========================================================
    # CONFIGURATION
    # ========================================================

    xgb_result = load_json(
        XGB_PARAMS_PATH
    )

    ensemble_config = load_json(
        ENSEMBLE_CONFIG_PATH
    )

    xgb_params = (
        xgb_result["best_params"]
    )

    production_threshold = float(
        ensemble_config[
            "production_threshold"
        ]
    )

    threshold_selection_metric = (
        ensemble_config.get(
            "threshold_selection_metric",
            "unknown"
        )
    )

    minimum_precision = float(
        ensemble_config.get(
            "minimum_precision",
            0.0
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

    print(
        f"Threshold selection metric: "
        f"{threshold_selection_metric}"
    )

    print(
        f"Minimum precision constraint: "
        f"{minimum_precision:.4f}"
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
        f"Dataset shape: "
        f"{data.shape}"
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
        f"Train:      "
        f"{y_train.mean() * 100:.4f}%"
    )

    print(
        f"Validation: "
        f"{y_val.mean() * 100:.4f}%"
    )

    print(
        f"Test:       "
        f"{y_test.mean() * 100:.4f}%"
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
            f"Version: "
            f"{champion.version}"
        )

        print(
            f"Run ID: "
            f"{champion.run_id}"
        )

        print(
            f"Aliases: "
            f"{champion.aliases}"
        )

    # ========================================================
    # TRAIN
    # ========================================================

    run_name = (
        "production_retrain_"
        +
        datetime.now(
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

            "model_type":
                "XGBoost",

            "stage":
                "retraining",

            "retraining":
                "true",

            "test_set_used_for_selection":
                "false",

            "registered_model":
                REGISTERED_MODEL_NAME,

            "threshold_selection_metric":
                threshold_selection_metric
        })

        safe_log_params({

            "production_threshold":
                production_threshold,

            "minimum_precision":
                minimum_precision,

            "minimum_validation_recall":
                MIN_VALIDATION_RECALL,

            "minimum_validation_precision":
                MIN_VALIDATION_PRECISION,

            "minimum_validation_pr_auc":
                MIN_VALIDATION_PR_AUC,

            "minimum_validation_f1":
                MIN_VALIDATION_F1
        })

        safe_log_params({

            f"xgb_{key}":
                value

            for key, value
            in xgb_params.items()
        })

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        model = train_xgboost(

            X_train_encoded,

            y_train,

            X_val_encoded,

            y_val,

            model_path=XGB_MODEL_PATH,

            **xgb_params
        )

        model_info = mlflow.xgboost.log_model(
            model,
            artifact_path="model"
        )

        # ----------------------------------------------------
        # PROBABILITIES
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
        # METRICS
        # ----------------------------------------------------

        validation_metrics = calculate_metrics(

            y_val,

            validation_probabilities,

            production_threshold
        )

        test_metrics = calculate_metrics(

            y_test,

            test_probabilities,

            production_threshold
        )

        # ----------------------------------------------------
        # LOG METRICS
        # ----------------------------------------------------

        mlflow.log_metrics({

            "validation_accuracy":
                validation_metrics[
                    "accuracy"
                ],

            "validation_precision":
                validation_metrics[
                    "precision"
                ],

            "validation_recall":
                validation_metrics[
                    "recall"
                ],

            "validation_f1":
                validation_metrics[
                    "f1"
                ],

            "validation_roc_auc":
                validation_metrics[
                    "roc_auc"
                ],

            "validation_pr_auc":
                validation_metrics[
                    "pr_auc"
                ],

            "test_accuracy":
                test_metrics[
                    "accuracy"
                ],

            "test_precision":
                test_metrics[
                    "precision"
                ],

            "test_recall":
                test_metrics[
                    "recall"
                ],

            "test_f1":
                test_metrics[
                    "f1"
                ],

            "test_roc_auc":
                test_metrics[
                    "roc_auc"
                ],

            "test_pr_auc":
                test_metrics[
                    "pr_auc"
                ]
        })

        # ====================================================
        # RESULTS
        # ====================================================

        print(
            "\n========== RETRAIN RESULTS =========="
        )

        print(
            f"Validation PR-AUC: "
            f"{validation_metrics['pr_auc']:.6f}"
        )

        print(
            f"Validation Precision: "
            f"{validation_metrics['precision']:.6f}"
        )

        print(
            f"Validation Recall: "
            f"{validation_metrics['recall']:.6f}"
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
            f"Test Precision: "
            f"{test_metrics['precision']:.6f}"
        )

        print(
            f"Test Recall: "
            f"{test_metrics['recall']:.6f}"
        )

        print(
            f"Test F1: "
            f"{test_metrics['f1']:.6f}"
        )

        # ====================================================
        # QUALITY GATES
        # ====================================================

        precision_gate = max(
            minimum_precision,
            MIN_VALIDATION_PRECISION
        )

        precision_passed = (
            validation_metrics[
                "precision"
            ]
            >= precision_gate
        )

        pr_auc_passed = (
            validation_metrics[
                "pr_auc"
            ]
            >= MIN_VALIDATION_PR_AUC
        )

        f1_passed = (
            validation_metrics[
                "f1"
            ]
            >= MIN_VALIDATION_F1
        )

        recall_passed = (
            validation_metrics[
                "recall"
            ]
            >= MIN_VALIDATION_RECALL
        )

        print(
            "\n========== QUALITY GATES =========="
        )

        print(
            f"Validation Precision: "
            f"{validation_metrics['precision']:.6f} "
            f"(required >= "
            f"{precision_gate:.2f}) "
            f"{'PASS' if precision_passed else 'FAIL'}"
        )

        print(
            f"Validation PR-AUC: "
            f"{validation_metrics['pr_auc']:.6f} "
            f"(required >= "
            f"{MIN_VALIDATION_PR_AUC:.2f}) "
            f"{'PASS' if pr_auc_passed else 'FAIL'}"
        )

        print(
            f"Validation F1: "
            f"{validation_metrics['f1']:.6f} "
            f"(required >= "
            f"{MIN_VALIDATION_F1:.2f}) "
            f"{'PASS' if f1_passed else 'FAIL'}"
        )

        print(
            f"Validation Recall: "
            f"{validation_metrics['recall']:.6f} "
            f"(required >= "
            f"{MIN_VALIDATION_RECALL:.2f}) "
            f"{'PASS' if recall_passed else 'FAIL'}"
        )

        quality_gates_passed = (
            precision_passed
            and
            pr_auc_passed
            and
            f1_passed
            and
            recall_passed
        )

        if not quality_gates_passed:

            mlflow.set_tag(
                "quality_gates",
                "failed"
            )

            mlflow.set_tag(
                "candidate_status",
                "rejected_quality_gate"
            )

            print(
                "\nCandidate FAILED quality gates."
            )

            print(
                "Model will NOT be registered "
                "or promoted."
            )

            return

        print(
            "\nAll quality gates PASSED."
        )

        mlflow.set_tag(
            "quality_gates",
            "passed"
        )

        # ====================================================
        # CHAMPION COMPARISON
        # ====================================================

        candidate_metrics = {

            "validation_precision":
                validation_metrics[
                    "precision"
                ],

            "validation_pr_auc":
                validation_metrics[
                    "pr_auc"
                ],

            "validation_f1":
                validation_metrics[
                    "f1"
                ],

            "validation_recall":
                validation_metrics[
                    "recall"
                ]
        }

        should_promote = False

        if champion is None:

            should_promote = True

            print(
                "\nNo current champion exists."
            )

        else:

            champion_metrics_raw = (
                get_run_metrics(
                    client,
                    champion.run_id
                )
            )

            required_metrics = [
                "validation_precision",
                "validation_pr_auc",
                "validation_f1",
                "validation_recall"
            ]

            missing = [
                metric
                for metric
                in required_metrics
                if champion_metrics_raw.get(
                    metric
                ) is None
            ]

            if missing:

                raise RuntimeError(
                    "Current champion is missing "
                    f"required metrics: {missing}. "
                    "Automatic promotion refused."
                )

            champion_metrics = {

                "validation_precision":
                    float(
                        champion_metrics_raw[
                            "validation_precision"
                        ]
                    ),

                "validation_pr_auc":
                    float(
                        champion_metrics_raw[
                            "validation_pr_auc"
                        ]
                    ),

                "validation_f1":
                    float(
                        champion_metrics_raw[
                            "validation_f1"
                        ]
                    ),

                "validation_recall":
                    float(
                        champion_metrics_raw[
                            "validation_recall"
                        ]
                    )
            }

            print(
                "\n========== CHAMPION COMPARISON =========="
            )

            print(
                f"Champion PR-AUC: "
                f"{champion_metrics['validation_pr_auc']:.6f}"
            )

            print(
                f"Candidate PR-AUC: "
                f"{candidate_metrics['validation_pr_auc']:.6f}"
            )

            print(
                f"Champion F1: "
                f"{champion_metrics['validation_f1']:.6f}"
            )

            print(
                f"Candidate F1: "
                f"{candidate_metrics['validation_f1']:.6f}"
            )

            print(
                f"Champion Recall: "
                f"{champion_metrics['validation_recall']:.6f}"
            )

            print(
                f"Candidate Recall: "
                f"{candidate_metrics['validation_recall']:.6f}"
            )

            should_promote = (
                champion_is_better_or_equal(
                    candidate_metrics,
                    champion_metrics
                )
            )

            if should_promote:

                print(
                    "\nCandidate beats the current champion "
                    "under the recall-first policy."
                )

            else:

                print(
                    "\nCandidate did not beat "
                    "the current champion under "
                    "the recall-first policy."
                )

        # ====================================================
        # REGISTER
        # ====================================================

        print(
            "\n========== REGISTERING CANDIDATE =========="
        )

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

        registered = mlflow.register_model(
            model_uri=model_info.model_uri,
            name=REGISTERED_MODEL_NAME
        )

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

        # ====================================================
        # REGISTRY TAGS
        # ====================================================

        client.set_model_version_tag(

            REGISTERED_MODEL_NAME,

            str(candidate_version),

            "candidate_status",

            (
                "promotion-ready"
                if should_promote
                else
                "registered-not-promoted"
            )
        )

        client.set_model_version_tag(

            REGISTERED_MODEL_NAME,

            str(candidate_version),

            "threshold_selection_metric",

            threshold_selection_metric
        )

        client.set_model_version_tag(

            REGISTERED_MODEL_NAME,

            str(candidate_version),

            "production_threshold",

            str(
                production_threshold
            )
        )

        client.set_model_version_tag(

            REGISTERED_MODEL_NAME,

            str(candidate_version),

            "promotion_policy",

            "recall_first_with_quality_gates"
        )

        client.set_model_version_tag(

            REGISTERED_MODEL_NAME,

            str(candidate_version),

            "validation_precision",

            str(
                validation_metrics[
                    "precision"
                ]
            )
        )

        client.set_model_version_tag(

            REGISTERED_MODEL_NAME,

            str(candidate_version),

            "validation_pr_auc",

            str(
                validation_metrics[
                    "pr_auc"
                ]
            )
        )

        client.set_model_version_tag(

            REGISTERED_MODEL_NAME,

            str(candidate_version),

            "validation_f1",

            str(
                validation_metrics[
                    "f1"
                ]
            )
        )

        client.set_model_version_tag(

            REGISTERED_MODEL_NAME,

            str(candidate_version),

            "validation_recall",

            str(
                validation_metrics[
                    "recall"
                ]
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
                    "despite champion comparison."
                )

            print(
                "\n========== PROMOTING CANDIDATE =========="
            )

            client.set_registered_model_alias(

                REGISTERED_MODEL_NAME,

                CHAMPION_ALIAS,

                str(candidate_version)
            )

            promoted = (
                client.get_model_version_by_alias(
                    REGISTERED_MODEL_NAME,
                    CHAMPION_ALIAS
                )
            )

            if str(
                promoted.version
            ) != str(
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
                f"Candidate version "
                f"{candidate_version} "
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
            f"\nMLflow run ID: "
            f"{run.info.run_id}"
        )

        print(
            f"Validation PR-AUC: "
            f"{validation_metrics['pr_auc']:.6f}"
        )

        print(
            f"Validation Precision: "
            f"{validation_metrics['precision']:.6f}"
        )

        print(
            f"Validation Recall: "
            f"{validation_metrics['recall']:.6f}"
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
            f"Test Precision: "
            f"{test_metrics['precision']:.6f}"
        )

        print(
            f"Test Recall: "
            f"{test_metrics['recall']:.6f}"
        )

        print(
            f"Test F1: "
            f"{test_metrics['f1']:.6f}"
        )


if __name__ == "__main__":
    main()