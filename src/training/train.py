import os

import pandas as pd
import mlflow
import mlflow.xgboost
import mlflow.pytorch
import torch

from src.data.split_data import split_data
from src.utils.mlflow_config import configure_mlflow
from src.data.model_processing import process_data

from src.models.xgboost_model import train_xgboost

from src.models.neural_network import (
    train_neural_network,
    predict_neural_network
)

from src.models.ensemble import (
    find_best_weight,
    ensemble_predict
)

from src.evaluation.metrics import (
    evaluate_model,
    evaluate_probabilities
)

from src.evaluation.confusion_matrix import (
    evaluate_confusion_matrix
)

from src.evaluation.threshold_analysis import (
    analyze_thresholds
)

from src.evaluation.cost_analysis import (
    analyze_costs
)


# ==========================================
# PATHS
# ==========================================

DATA_PATH = (
    "data/processed/"
    "feature_engineered.csv"
)

MODEL_PATH = (
    "models/xgboost_model.json"
)

PREPROCESSOR_PATH = (
    "models/preprocessor.pkl"
)

NN_MODEL_PATH = (
    "models/neural_network.pt"
)


# ==========================================
# MLFLOW NAMES
# ==========================================

XGB_REGISTERED_MODEL = (
    "AI-Risk-Manager-XGBoost"
)

NN_REGISTERED_MODEL = (
    "AI-Risk-Manager-Neural-Network"
)


def main():

    # ==========================================
    # MLFLOW CONFIGURATION
    # ==========================================

    configure_mlflow()

    # ==========================================
    # LOAD DATASET
    # ==========================================

    print(
        "\n========== DATA LOADING =========="
    )

    print(
        "Loading dataset..."
    )

    data = pd.read_csv(
        DATA_PATH
    )

    print(
        f"Dataset shape: {data.shape}"
    )

    # ==========================================
    # DATA SPLIT
    # ==========================================

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
        "\n========== DATA SPLIT =========="
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

    # ==========================================
    # PREPROCESSING
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

    categorical_features = len(
        X_train.select_dtypes(
            include=["object"]
        ).columns
    )

    numeric_features = len(
        X_train.select_dtypes(
            exclude=["object"]
        ).columns
    )

    encoded_features = (
        X_train_encoded.shape[1]
    )

    print(
        "\n========== MODEL PREPROCESSING =========="
    )

    print(
        f"Categorical features: "
        f"{categorical_features}"
    )

    print(
        f"Numerical features: "
        f"{numeric_features}"
    )

    print(
        f"Encoded features: "
        f"{encoded_features}"
    )

    print(
        f"Train shape: "
        f"{X_train_encoded.shape}"
    )

    print(
        f"Validation shape: "
        f"{X_val_encoded.shape}"
    )

    print(
        f"Test shape: "
        f"{X_test_encoded.shape}"
    )

    # ==========================================
    # COMMON MLFLOW PARAMETERS
    # ==========================================

    common_params = {

        "dataset": (
            "feature_engineered.csv"
        ),

        "total_samples": len(data),

        "train_samples": len(
            X_train
        ),

        "validation_samples": len(
            X_val
        ),

        "test_samples": len(
            X_test
        ),

        "original_features": (
            X_train.shape[1]
        ),

        "categorical_features": (
            categorical_features
        ),

        "numeric_features": (
            numeric_features
        ),

        "encoded_features": (
            encoded_features
        ),

        "train_fraud_rate": float(
            y_train.mean()
        ),

        "validation_fraud_rate": float(
            y_val.mean()
        ),

        "test_fraud_rate": float(
            y_test.mean()
        )
    }

    # ==========================================================
    # ==========================================================
    # XGBOOST RUN
    # ==========================================================
    # ==========================================================

    print(
        "\n\n=========================================="
    )

    print(
        "========== XGBOOST MLFLOW RUN =========="
    )

    print(
        "=========================================="
    )

    with mlflow.start_run(
        run_name="xgboost_baseline"
    ):

        mlflow.set_tags({
            "model_type": "XGBoost",
            "stage": "baseline"
        })

        mlflow.log_params(
            common_params
        )

        # ======================================
        # TRAIN XGBOOST
        # ======================================

        model = train_xgboost(
            X_train_encoded,
            y_train,
            X_val_encoded,
            y_val,
            model_path=MODEL_PATH
        )

        # ======================================
        # XGBOOST PROBABILITIES
        # ======================================

        xgb_val_prob = (
            model.predict_proba(
                X_val_encoded
            )[:, 1]
        )

        xgb_test_prob = (
            model.predict_proba(
                X_test_encoded
            )[:, 1]
        )

        # ======================================
        # XGBOOST PARAMETERS
        # ======================================

        model_params = (
            model.get_params()
        )

        xgb_params = {

            "objective": model_params.get(
                "objective"
            ),

            "n_estimators": model_params.get(
                "n_estimators"
            ),

            "learning_rate": model_params.get(
                "learning_rate"
            ),

            "max_depth": model_params.get(
                "max_depth"
            ),

            "min_child_weight": model_params.get(
                "min_child_weight"
            ),

            "subsample": model_params.get(
                "subsample"
            ),

            "colsample_bytree": model_params.get(
                "colsample_bytree"
            ),

            "reg_alpha": model_params.get(
                "reg_alpha"
            ),

            "reg_lambda": model_params.get(
                "reg_lambda"
            ),

            "gamma": model_params.get(
                "gamma"
            ),

            "random_state": model_params.get(
                "random_state"
            ),

            "eval_metric": model_params.get(
                "eval_metric"
            ),

            "scale_pos_weight": model_params.get(
                "scale_pos_weight"
            )
        }

        # Remove None values
        xgb_params = {
            key: value
            for key, value
            in xgb_params.items()
            if value is not None
        }

        mlflow.log_params(
            xgb_params
        )

        # ======================================
        # BEST XGBOOST ITERATION
        # ======================================

        mlflow.log_metrics({

            "best_iteration": float(
                model.best_iteration
            ),

            "best_validation_aucpr": float(
                model.best_score
            )
        })

        # ======================================
        # DEFAULT TEST EVALUATION
        # ======================================

        xgb_metrics = evaluate_model(
            model,
            X_test_encoded,
            y_test
        )

        mlflow.log_metrics({

            "test_accuracy": (
                xgb_metrics["accuracy"]
            ),

            "test_precision": (
                xgb_metrics["precision"]
            ),

            "test_recall": (
                xgb_metrics["recall"]
            ),

            "test_f1": (
                xgb_metrics["f1"]
            ),

            "test_roc_auc": (
                xgb_metrics["roc_auc"]
            ),

            "test_pr_auc": (
                xgb_metrics["pr_auc"]
            )
        })

        # ======================================
        # VALIDATION THRESHOLD ANALYSIS
        # ======================================

        print(
            "\n========== XGBOOST "
            "VALIDATION THRESHOLD ANALYSIS =========="
        )

        xgb_threshold_results = (
            analyze_thresholds(
                y_val,
                xgb_val_prob
            )
        )

        best_xgb_threshold = max(
            xgb_threshold_results,
            key=lambda result: result["f1"]
        )

        print(
            "\nBest XGBoost validation "
            "F1 threshold:"
        )

        print(
            f"Threshold: "
            f"{best_xgb_threshold['threshold']}"
        )

        print(
            f"F1: "
            f"{best_xgb_threshold['f1']:.4f}"
        )

        mlflow.log_metrics({

            "validation_best_f1_threshold": (
                best_xgb_threshold[
                    "threshold"
                ]
            ),

            "validation_best_threshold_precision": (
                best_xgb_threshold[
                    "precision"
                ]
            ),

            "validation_best_threshold_recall": (
                best_xgb_threshold[
                    "recall"
                ]
            ),

            "validation_best_threshold_f1": (
                best_xgb_threshold[
                    "f1"
                ]
            )
        })

        # ======================================
        # VALIDATION COST ANALYSIS
        # ======================================

        false_positive_cost = 100
        false_negative_cost = 5000

        xgb_cost_results = (
            analyze_costs(
                y_val,
                xgb_val_prob,
                false_positive_cost=(
                    false_positive_cost
                ),
                false_negative_cost=(
                    false_negative_cost
                )
            )
        )

        best_xgb_cost = min(
            xgb_cost_results,
            key=lambda result:
                result["total_cost"]
        )

        mlflow.log_params({

            "false_positive_cost": (
                false_positive_cost
            ),

            "false_negative_cost": (
                false_negative_cost
            )
        })

        mlflow.log_metrics({

            "validation_cost_optimal_threshold": (
                best_xgb_cost[
                    "threshold"
                ]
            ),

            "validation_minimum_total_cost": (
                best_xgb_cost[
                    "total_cost"
                ]
            ),

            "validation_cost_optimal_fp": (
                best_xgb_cost[
                    "fp"
                ]
            ),

            "validation_cost_optimal_fn": (
                best_xgb_cost[
                    "fn"
                ]
            )
        })

        # ======================================
        # CONFUSION MATRIX
        # ======================================

        confusion_results = (
            evaluate_confusion_matrix(
                model,
                X_test_encoded,
                y_test,
                threshold=0.5
            )
        )

        mlflow.log_metrics({

            "test_true_negatives": (
                confusion_results["tn"]
            ),

            "test_false_positives": (
                confusion_results["fp"]
            ),

            "test_false_negatives": (
                confusion_results["fn"]
            ),

            "test_true_positives": (
                confusion_results["tp"]
            )
        })

        # ======================================
        # LOG XGBOOST ARTIFACT
        # ======================================

        if os.path.exists(
            MODEL_PATH
        ):

            mlflow.log_artifact(
                MODEL_PATH,
                artifact_path="model"
            )

        # ======================================
        # LOG PREPROCESSOR
        # ======================================

        if os.path.exists(
            PREPROCESSOR_PATH
        ):

            mlflow.log_artifact(
                PREPROCESSOR_PATH,
                artifact_path="preprocessor"
            )

        # ======================================
        # REGISTER XGBOOST
        # ======================================

        print(
            "\n========== MLFLOW "
            "XGBOOST REGISTRATION =========="
        )

        mlflow.xgboost.log_model(
            xgb_model=model,
            name="xgboost_model",
            registered_model_name=(
                XGB_REGISTERED_MODEL
            ),
            model_format="json"
        )

        print(
            f"Registered model: "
            f"{XGB_REGISTERED_MODEL}"
        )

    # ==========================================================
    # ==========================================================
    # NEURAL NETWORK RUN
    # ==========================================================
    # ==========================================================

    print(
        "\n\n=========================================="
    )

    print(
        "====== NEURAL NETWORK MLFLOW RUN ======="
    )

    print(
        "=========================================="
    )

    with mlflow.start_run(
        run_name="neural_network_baseline"
    ):

        mlflow.set_tags({
            "model_type": "Neural Network",
            "stage": "baseline"
        })

        mlflow.log_params(
            common_params
        )

        # ======================================
        # NN CONFIG
        # ======================================

        nn_epochs = 10
        nn_batch_size = 1024
        nn_learning_rate = 0.001

        mlflow.log_params({

            "nn_epochs": nn_epochs,

            "nn_batch_size": (
                nn_batch_size
            ),

            "nn_learning_rate": (
                nn_learning_rate
            ),

            "nn_input_size": (
                encoded_features
            ),

            "nn_architecture": (
                f"{encoded_features}-256-128-64-1"
            ),

            "nn_dropout": (
                "0.30-0.30-0.20"
            ),

            "nn_optimizer": "Adam",

            "nn_loss": (
                "BCEWithLogitsLoss"
            )
        })

        # ======================================
        # TRAIN NN
        # ======================================

        nn_model = train_neural_network(
            X_train_encoded,
            y_train,
            X_val_encoded,
            y_val,
            epochs=nn_epochs,
            batch_size=nn_batch_size,
            learning_rate=nn_learning_rate
        )

        # ======================================
        # NN PROBABILITIES
        # ======================================

        nn_val_prob = (
            predict_neural_network(
                nn_model,
                X_val_encoded
            )
        )

        nn_test_prob = (
            predict_neural_network(
                nn_model,
                X_test_encoded
            )
        )

        # ======================================
        # NN TEST EVALUATION
        # ======================================

        nn_metrics = (
            evaluate_probabilities(
                y_test,
                nn_test_prob,
                threshold=0.5,
                title=(
                    "NEURAL NETWORK "
                    "TEST SET EVALUATION"
                )
            )
        )

        mlflow.log_metrics({

            "test_accuracy": (
                nn_metrics["accuracy"]
            ),

            "test_precision": (
                nn_metrics["precision"]
            ),

            "test_recall": (
                nn_metrics["recall"]
            ),

            "test_f1": (
                nn_metrics["f1"]
            ),

            "test_roc_auc": (
                nn_metrics["roc_auc"]
            ),

            "test_pr_auc": (
                nn_metrics["pr_auc"]
            )
        })

        # ======================================
        # SAVE NN
        # ======================================

        os.makedirs(
            "models",
            exist_ok=True
        )

        torch.save(
            nn_model.state_dict(),
            NN_MODEL_PATH
        )

        mlflow.log_artifact(
            NN_MODEL_PATH,
            artifact_path="model"
        )

        # ======================================
        # REGISTER NN WITH MLFLOW
        # ======================================

        print(
            "\n========== MLFLOW "
            "NEURAL NETWORK REGISTRATION =========="
        )

        # ======================================
        # REGISTER NEURAL NETWORK WITH MLFLOW
        # ======================================

        print(
            "\n========== MLFLOW "
            "NEURAL NETWORK REGISTRATION =========="
        )

        nn_model_cpu = nn_model.to("cpu")

        input_example = torch.zeros(
            (1, encoded_features),
            dtype=torch.float32
        )

        nn_model_cpu = nn_model.to("cpu")
        input_example = torch.zeros(
            (1, encoded_features),
            dtype=torch.float32
        )
        mlflow.pytorch.log_model(
            nn_model_cpu,
            name="neural_network_model",
            registered_model_name=NN_REGISTERED_MODEL,
            input_example=input_example,
            serialization_format="pickle"
        )

        print(
            f"Registered model: "
            f"{NN_REGISTERED_MODEL}"
        )

        print(
            f"Registered model: "
            f"{NN_REGISTERED_MODEL}"
        )

    # ==========================================================
    # ==========================================================
    # ENSEMBLE RUN
    # ==========================================================
    # ==========================================================

    print(
        "\n\n=========================================="
    )

    print(
        "========== ENSEMBLE MLFLOW RUN =========="
    )

    print(
        "=========================================="
    )

    with mlflow.start_run(
        run_name="xgb_nn_ensemble"
    ):

        mlflow.set_tags({
            "model_type": (
                "XGBoost + Neural Network"
            ),

            "stage": "ensemble"
        })

        mlflow.log_params(
            common_params
        )

        # ======================================
        # FIND BEST CONFIGURATION
        # ======================================

        (
            xgb_weight,
            nn_weight,
            ensemble_threshold,
            validation_f1
        ) = find_best_weight(
            y_val,
            xgb_val_prob,
            nn_val_prob
        )

        # ======================================
        # LOG ENSEMBLE CONFIGURATION
        # ======================================

        mlflow.log_params({

            "ensemble_xgb_weight": (
                xgb_weight
            ),

            "ensemble_nn_weight": (
                nn_weight
            ),

            "ensemble_threshold": (
                ensemble_threshold
            )
        })

        mlflow.log_metric(
            "validation_f1",
            float(validation_f1)
        )

        # ======================================
        # VALIDATION ENSEMBLE
        # ======================================

        ensemble_val_prob = (
            ensemble_predict(
                xgb_val_prob,
                nn_val_prob,
                xgb_weight,
                nn_weight
            )
        )

        # ======================================
        # TEST ENSEMBLE
        # ======================================

        ensemble_test_prob = (
            ensemble_predict(
                xgb_test_prob,
                nn_test_prob,
                xgb_weight,
                nn_weight
            )
        )

        # ======================================
        # FINAL TEST EVALUATION
        # ======================================

        ensemble_metrics = (
            evaluate_probabilities(
                y_test,
                ensemble_test_prob,
                threshold=(
                    ensemble_threshold
                ),
                title=(
                    "ENSEMBLE "
                    "TEST SET EVALUATION"
                )
            )
        )

        # ======================================
        # LOG ENSEMBLE METRICS
        # ======================================

        mlflow.log_metrics({

            "test_accuracy": (
                ensemble_metrics[
                    "accuracy"
                ]
            ),

            "test_precision": (
                ensemble_metrics[
                    "precision"
                ]
            ),

            "test_recall": (
                ensemble_metrics[
                    "recall"
                ]
            ),

            "test_f1": (
                ensemble_metrics[
                    "f1"
                ]
            ),

            "test_roc_auc": (
                ensemble_metrics[
                    "roc_auc"
                ]
            ),

            "test_pr_auc": (
                ensemble_metrics[
                    "pr_auc"
                ]
            )
        })

        # ======================================
        # LOG ENSEMBLE INFORMATION
        # ======================================

        print(
            "\n========== ENSEMBLE RESULTS =========="
        )

        print(
            f"XGBoost weight: "
            f"{xgb_weight:.2f}"
        )

        print(
            f"Neural Network weight: "
            f"{nn_weight:.2f}"
        )

        print(
            f"Threshold: "
            f"{ensemble_threshold:.2f}"
        )

        print(
            f"Validation F1: "
            f"{validation_f1:.4f}"
        )

        print(
            f"Test F1: "
            f"{ensemble_metrics['f1']:.4f}"
        )

    # ==========================================================
    # FINAL
    # ==========================================================

    print(
        "\n=========================================="
    )

    print(
        "====== ALL TRAINING COMPLETE ======"
    )

    print(
        "XGBoost training: COMPLETE"
    )

    print(
        "Neural Network training: COMPLETE"
    )

    print(
        "Ensemble evaluation: COMPLETE"
    )

    print(
        "MLflow logging: COMPLETE"
    )

    print(
        "=========================================="
    )


if __name__ == "__main__":
    main()