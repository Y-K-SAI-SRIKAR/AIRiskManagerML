import os
import json
import itertools

import mlflow
import mlflow.xgboost
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)

from src.data.split_data import split_data
from src.data.model_processing import process_data
from src.utils.mlflow_config import configure_mlflow
from src.models.xgboost_model import train_xgboost


DATA_PATH = (
    "data/processed/"
    "feature_engineered.csv"
)

BEST_PARAMS_PATH = (
    "models/best_xgboost_params.json"
)


def evaluate_predictions(
    y_true,
    y_prob,
    threshold=0.5
):

    y_pred = (
        y_prob >= threshold
    ).astype(int)

    return {
        "pr_auc": average_precision_score(
            y_true,
            y_prob
        ),

        "roc_auc": roc_auc_score(
            y_true,
            y_prob
        ),

        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0
        )
    }


def main():

    configure_mlflow()

    print(
        "\n========== XGBOOST TUNING =========="
    )

    # ==========================================
    # LOAD DATA
    # ==========================================

    print(
        "Loading dataset..."
    )

    data = pd.read_csv(
        DATA_PATH
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

    # ==========================================
    # PREPROCESS
    # ==========================================

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

    # ==========================================
    # SEARCH SPACE
    # ==========================================

    param_grid = {

        "learning_rate": [
            0.03,
            0.05,
            0.08
        ],

        "max_depth": [
            4,
            6,
            8
        ],

        "min_child_weight": [
            1,
            5,
            10
        ],

        "subsample": [
            0.8,
            1.0
        ],

        "colsample_bytree": [
            0.8,
            1.0
        ],

        "gamma": [
            0,
            0.2
        ],

        "reg_alpha": [
            0,
            0.1
        ],

        "reg_lambda": [
            1,
            5
        ]
    }

    # ==========================================
    # MANUAL TRIAL SELECTION
    # ==========================================
    #
    # Full Cartesian product would be:
    #
    # 3*3*3*2*2*2*2*2 = 1728 trials
    #
    # That is far too expensive.
    #
    # Instead, use selected combinations.
    # ==========================================

    trials = [

        {
            "learning_rate": 0.05,
            "max_depth": 6,
            "min_child_weight": 1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 0,
            "reg_alpha": 0,
            "reg_lambda": 1
        },

        {
            "learning_rate": 0.03,
            "max_depth": 6,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 0,
            "reg_alpha": 0.1,
            "reg_lambda": 5
        },

        {
            "learning_rate": 0.05,
            "max_depth": 8,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 0.2,
            "reg_alpha": 0.1,
            "reg_lambda": 5
        },

        {
            "learning_rate": 0.03,
            "max_depth": 8,
            "min_child_weight": 10,
            "subsample": 0.8,
            "colsample_bytree": 1.0,
            "gamma": 0.2,
            "reg_alpha": 0.1,
            "reg_lambda": 5
        },

        {
            "learning_rate": 0.08,
            "max_depth": 4,
            "min_child_weight": 1,
            "subsample": 1.0,
            "colsample_bytree": 0.8,
            "gamma": 0,
            "reg_alpha": 0,
            "reg_lambda": 1
        },

        {
            "learning_rate": 0.05,
            "max_depth": 4,
            "min_child_weight": 5,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "gamma": 0,
            "reg_alpha": 0.1,
            "reg_lambda": 5
        },

        {
            "learning_rate": 0.03,
            "max_depth": 6,
            "min_child_weight": 10,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "gamma": 0.2,
            "reg_alpha": 0.1,
            "reg_lambda": 5
        },

        {
            "learning_rate": 0.08,
            "max_depth": 8,
            "min_child_weight": 1,
            "subsample": 0.8,
            "colsample_bytree": 1.0,
            "gamma": 0,
            "reg_alpha": 0,
            "reg_lambda": 1
        }
    ]

    print(
        f"\nTotal XGBoost trials: "
        f"{len(trials)}"
    )

    # ==========================================
    # BASELINE SCALE POS WEIGHT
    # ==========================================

    negative = (
        y_train == 0
    ).sum()

    positive = (
        y_train == 1
    ).sum()

    scale_pos_weight = (
        negative / positive
    )

    print(
        f"Scale pos weight: "
        f"{scale_pos_weight:.4f}"
    )

    # ==========================================
    # BEST TRACKING
    # ==========================================

    best_pr_auc = -1.0
    best_params = None
    best_metrics = None

    os.makedirs(
        "models",
        exist_ok=True
    )

    # ==========================================
    # TRIALS
    # ==========================================

    for trial_number, params in enumerate(
        trials,
        start=1
    ):

        print(
            "\n=========================================="
        )

        print(
            f"XGBOOST TRIAL {trial_number}/"
            f"{len(trials)}"
        )

        print(
            "=========================================="
        )

        print(
            json.dumps(
                params,
                indent=2
            )
        )

        # ======================================
        # START MLFLOW RUN
        # ======================================

        with mlflow.start_run(
            run_name=(
                f"xgboost_tuning_"
                f"{trial_number:02d}"
            )
        ):

            mlflow.set_tags({
                "model_type": "XGBoost",
                "stage": "tuning",
                "tuning_trial": str(
                    trial_number
                )
            })

            mlflow.log_params(
                params
            )

            mlflow.log_param(
                "scale_pos_weight",
                scale_pos_weight
            )

            # ==================================
            # TRAIN
            # ==================================

            model = train_xgboost(
                X_train_encoded,
                y_train,
                X_val_encoded,
                y_val,
                model_path=(
                    "models/"
                    f"xgb_trial_{trial_number}.json"
                ),

                learning_rate=params[
                    "learning_rate"
                ],

                max_depth=params[
                    "max_depth"
                ],

                min_child_weight=params[
                    "min_child_weight"
                ],

                subsample=params[
                    "subsample"
                ],

                colsample_bytree=params[
                    "colsample_bytree"
                ],

                gamma=params[
                    "gamma"
                ],

                reg_alpha=params[
                    "reg_alpha"
                ],

                reg_lambda=params[
                    "reg_lambda"
                ],

                scale_pos_weight=(
                    scale_pos_weight
                )
            )

            # ==================================
            # VALIDATION
            # ==================================

            val_prob = (
                model.predict_proba(
                    X_val_encoded
                )[:, 1]
            )

            metrics = evaluate_predictions(
                y_val,
                val_prob
            )

            mlflow.log_metrics({

                "validation_pr_auc":
                    metrics["pr_auc"],

                "validation_roc_auc":
                    metrics["roc_auc"],

                "validation_precision":
                    metrics["precision"],

                "validation_recall":
                    metrics["recall"],

                "validation_f1":
                    metrics["f1"]
            })

            print(
                f"Validation PR-AUC: "
                f"{metrics['pr_auc']:.4f}"
            )

            print(
                f"Validation ROC-AUC: "
                f"{metrics['roc_auc']:.4f}"
            )

            print(
                f"Validation F1: "
                f"{metrics['f1']:.4f}"
            )

            # ==================================
            # BEST MODEL
            # ==================================

            if metrics["pr_auc"] > best_pr_auc:

                best_pr_auc = (
                    metrics["pr_auc"]
                )

                best_params = params.copy()

                best_metrics = metrics.copy()

                print(
                    "\n*** NEW BEST XGBOOST "
                    "MODEL ***"
                )

    # ==========================================
    # SAVE BEST PARAMETERS
    # ==========================================

    result = {

        "model": "XGBoost",

        "selection_metric": (
            "validation_pr_auc"
        ),

        "best_validation_pr_auc": (
            best_pr_auc
        ),

        "best_validation_roc_auc": (
            best_metrics["roc_auc"]
        ),

        "best_validation_f1": (
            best_metrics["f1"]
        ),

        "best_params": best_params,

        "scale_pos_weight": (
            scale_pos_weight
        )
    }

    with open(
        BEST_PARAMS_PATH,
        "w"
    ) as file:

        json.dump(
            result,
            file,
            indent=4
        )

    print(
        "\n=========================================="
    )

    print(
        "====== BEST XGBOOST CONFIGURATION ======"
    )

    print(
        "=========================================="
    )

    print(
        json.dumps(
            result,
            indent=4
        )
    )

    print(
        f"\nSaved to: "
        f"{BEST_PARAMS_PATH}"
    )


if __name__ == "__main__":
    main()