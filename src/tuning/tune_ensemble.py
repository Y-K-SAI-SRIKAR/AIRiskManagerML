import json
import os

import mlflow
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)

from src.data.split_data import split_data
from src.data.model_processing import process_data
from src.utils.mlflow_config import configure_mlflow

from src.models.xgboost_model import train_xgboost

from src.models.neural_network import (
    FraudNeuralNetwork,
    predict_neural_network
)


# ============================================================
# PATHS
# ============================================================

DATA_PATH = "data/processed/feature_engineered.csv"

XGB_PARAMS_PATH = "models/best_xgboost_params.json"

NN_PARAMS_PATH = "models/best_neural_network_params.json"

NN_MODEL_PATH = "models/best_neural_network.pt"

XGB_MODEL_PATH = "models/tuned_xgboost_model.json"

ENSEMBLE_RESULTS_PATH = "models/best_ensemble_config.json"


# ============================================================
# RISK POLICY
# ============================================================
#
# These values control the production operating point.
# They are configuration, not model-specific hardcoded
# decisions.
#
# Primary threshold objective:
#   maximize recall
#
# Constraint:
#   precision must remain >= MINIMUM_PRECISION
#
# Model selection:
#   validation PR-AUC
# ============================================================

MINIMUM_PRECISION = 0.70

THRESHOLD_MIN = 0.10
THRESHOLD_MAX = 0.90
THRESHOLD_STEP = 0.01


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    probabilities,
    threshold=0.5
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
            )
    }


# ============================================================
# RECALL-AWARE THRESHOLD SEARCH
# ============================================================

def find_best_threshold(
    y_true,
    probabilities,
    minimum_precision=MINIMUM_PRECISION
):

    thresholds = np.arange(
        THRESHOLD_MIN,
        THRESHOLD_MAX + THRESHOLD_STEP / 2,
        THRESHOLD_STEP
    )

    eligible = []

    print(
        "\n========== RECALL-AWARE "
        "VALIDATION THRESHOLD ANALYSIS =========="
    )

    print(
        f"Minimum precision constraint: "
        f"{minimum_precision:.4f}"
    )

    print(
        f"\n{'Threshold':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1':<12}"
        f"{'Fraud Flags':<15}"
        f"{'Eligible':<10}"
    )

    for threshold in thresholds:

        threshold = round(
            float(threshold),
            2
        )

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = float(
            precision_score(
                y_true,
                predictions,
                zero_division=0
            )
        )

        recall = float(
            recall_score(
                y_true,
                predictions,
                zero_division=0
            )
        )

        f1 = float(
            f1_score(
                y_true,
                predictions,
                zero_division=0
            )
        )

        fraud_flags = int(
            predictions.sum()
        )

        is_eligible = (
            precision >= minimum_precision
        )

        if is_eligible:
            eligible.append({
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "fraud_flags": fraud_flags
            })

        print(
            f"{threshold:<12.2f}"
            f"{precision:<12.4f}"
            f"{recall:<12.4f}"
            f"{f1:<12.4f}"
            f"{fraud_flags:<15}"
            f"{'YES' if is_eligible else 'NO':<10}"
        )

    if not eligible:
        raise RuntimeError(
            "No validation threshold satisfies "
            f"minimum precision={minimum_precision:.4f}"
        )

    # --------------------------------------------------------
    # PRIMARY:
    #   maximize recall
    #
    # TIE BREAKER:
    #   maximize F1
    #
    # FINAL TIE BREAKER:
    #   maximize precision
    # --------------------------------------------------------

    best = max(
        eligible,
        key=lambda result: (
            result["recall"],
            result["f1"],
            result["precision"]
        )
    )

    print(
        "\n========== SELECTED "
        "RECALL-AWARE THRESHOLD =========="
    )

    print(
        f"Threshold: "
        f"{best['threshold']:.2f}"
    )

    print(
        f"Precision: "
        f"{best['precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{best['recall']:.4f}"
    )

    print(
        f"F1: "
        f"{best['f1']:.4f}"
    )

    return (
        float(best["threshold"]),
        {
            "precision":
                float(best["precision"]),

            "recall":
                float(best["recall"]),

            "f1":
                float(best["f1"])
        }
    )


# ============================================================
# COST ANALYSIS
# ============================================================

def analyze_costs(
    y_true,
    probabilities,
    false_positive_cost=100,
    false_negative_cost=5000
):

    thresholds = np.arange(
        THRESHOLD_MIN,
        THRESHOLD_MAX + THRESHOLD_STEP / 2,
        THRESHOLD_STEP
    )

    best_threshold = 0.50
    best_cost = float("inf")

    best_fp = 0
    best_fn = 0

    print(
        "\n========== ENSEMBLE "
        "COST ANALYSIS =========="
    )

    print(
        f"{'Threshold':<12}"
        f"{'FP':<10}"
        f"{'FN':<10}"
        f"{'Total Cost':<15}"
    )

    for threshold in thresholds:

        threshold = round(
            float(threshold),
            2
        )

        predictions = (
            probabilities >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1]
        ).ravel()

        fp = int(fp)
        fn = int(fn)

        total_cost = int(
            fp * false_positive_cost
            +
            fn * false_negative_cost
        )

        print(
            f"{threshold:<12.2f}"
            f"{fp:<10}"
            f"{fn:<10}"
            f"₹{total_cost:<14,}"
        )

        if total_cost < best_cost:

            best_cost = total_cost

            best_threshold = threshold

            best_fp = fp
            best_fn = fn

    print(
        "\n========== TEST COST ANALYSIS =========="
    )

    print(
        f"Minimum-cost threshold: "
        f"{best_threshold:.2f}"
    )

    print(
        f"False Positives: {best_fp}"
    )

    print(
        f"False Negatives: {best_fn}"
    )

    print(
        f"Total Cost: ₹{best_cost:,}"
    )

    return {
        "threshold":
            float(best_threshold),

        "false_positives":
            int(best_fp),

        "false_negatives":
            int(best_fn),

        "total_cost":
            int(best_cost)
    }


# ============================================================
# MAIN
# ============================================================

def main():

    configure_mlflow()

    print(
        "\n=========================================="
    )

    print(
        "========== ENSEMBLE TUNING ============="
    )

    print(
        "=========================================="
    )

    # ========================================================
    # DATA LOADING
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

    # ========================================================
    # PREPROCESSING
    # ========================================================

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
    # LOAD PARAMETERS
    # ========================================================

    with open(
        XGB_PARAMS_PATH,
        "r"
    ) as file:

        xgb_result = json.load(
            file
        )

    with open(
        NN_PARAMS_PATH,
        "r"
    ) as file:

        nn_result = json.load(
            file
        )

    xgb_params = (
        xgb_result["best_params"]
    )

    nn_params = (
        nn_result["best_params"]
    )

    print(
        "\n========== BEST XGBOOST PARAMETERS =========="
    )

    print(
        json.dumps(
            xgb_params,
            indent=4
        )
    )

    print(
        "\n========== BEST NN PARAMETERS =========="
    )

    print(
        json.dumps(
            nn_params,
            indent=4
        )
    )

    # ========================================================
    # TRAIN XGBOOST
    # ========================================================

    print(
        "\n=========================================="
    )

    print(
        "====== TRAINING BEST XGBOOST ==========="
    )

    print(
        "=========================================="
    )

    with mlflow.start_run(
        run_name="final_tuned_xgboost"
    ):

        mlflow.set_tags({
            "model_type": "XGBoost",
            "stage": "final_tuned_model"
        })

        model_xgb = train_xgboost(
            X_train_encoded,
            y_train,
            X_val_encoded,
            y_val,
            model_path=XGB_MODEL_PATH,
            **xgb_params
        )

        xgb_val_prob = (
            model_xgb.predict_proba(
                X_val_encoded
            )[:, 1]
        )

        xgb_test_prob = (
            model_xgb.predict_proba(
                X_test_encoded
            )[:, 1]
        )

        xgb_val_metrics = calculate_metrics(
            y_val,
            xgb_val_prob
        )

        xgb_test_metrics = calculate_metrics(
            y_test,
            xgb_test_prob
        )

        print(
            "\nXGBoost validation PR-AUC: "
            f"{xgb_val_metrics['pr_auc']:.4f}"
        )

        print(
            "XGBoost test PR-AUC: "
            f"{xgb_test_metrics['pr_auc']:.4f}"
        )

        mlflow.log_metrics({
            "validation_pr_auc":
                xgb_val_metrics["pr_auc"],

            "validation_roc_auc":
                xgb_val_metrics["roc_auc"],

            "validation_f1":
                xgb_val_metrics["f1"],

            "test_pr_auc":
                xgb_test_metrics["pr_auc"],

            "test_roc_auc":
                xgb_test_metrics["roc_auc"],

            "test_f1":
                xgb_test_metrics["f1"]
        })

    # ========================================================
    # LOAD NEURAL NETWORK
    # ========================================================

    print(
        "\n=========================================="
    )

    print(
        "====== LOADING BEST NEURAL NETWORK ====="
    )

    print(
        "=========================================="
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Neural Network device: "
        f"{device}"
    )

    if device.type == "cuda":
        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    model_nn = FraudNeuralNetwork(
        input_size=X_train_encoded.shape[1],
        hidden_1=nn_params["hidden_1"],
        hidden_2=nn_params["hidden_2"],
        hidden_3=nn_params["hidden_3"],
        dropout_1=nn_params["dropout_1"],
        dropout_2=nn_params["dropout_2"],
        dropout_3=nn_params["dropout_3"]
    )

    checkpoint = torch.load(
        NN_MODEL_PATH,
        map_location=device
    )

    model_nn.load_state_dict(
        checkpoint
    )

    model_nn = model_nn.to(
        device
    )

    model_nn.eval()

    print(
        "Neural Network checkpoint "
        "loaded successfully."
    )

    # ========================================================
    # NN PREDICTIONS
    # ========================================================

    nn_val_prob = predict_neural_network(
        model_nn,
        X_val_encoded,
        batch_size=nn_params["batch_size"]
    )

    nn_test_prob = predict_neural_network(
        model_nn,
        X_test_encoded,
        batch_size=nn_params["batch_size"]
    )

    nn_val_metrics = calculate_metrics(
        y_val,
        nn_val_prob
    )

    nn_test_metrics = calculate_metrics(
        y_test,
        nn_test_prob
    )

    print(
        "\n========== NEURAL NETWORK RESULTS =========="
    )

    print(
        f"Validation PR-AUC: "
        f"{nn_val_metrics['pr_auc']:.4f}"
    )

    print(
        f"Validation ROC-AUC: "
        f"{nn_val_metrics['roc_auc']:.4f}"
    )

    print(
        f"Test PR-AUC: "
        f"{nn_test_metrics['pr_auc']:.4f}"
    )

    print(
        f"Test ROC-AUC: "
        f"{nn_test_metrics['roc_auc']:.4f}"
    )

    # ========================================================
    # ENSEMBLE WEIGHT SEARCH
    # ========================================================

    print(
        "\n=========================================="
    )

    print(
        "======= ENSEMBLE WEIGHT SEARCH ========="
    )

    print(
        "=========================================="
    )

    best_xgb_weight = 1.0
    best_nn_weight = 0.0

    best_pr_auc = -1.0

    best_val_prob = None

    print(
        f"\n{'XGB Weight':<15}"
        f"{'NN Weight':<15}"
        f"{'Validation PR-AUC':<20}"
        f"{'Validation F1':<15}"
    )

    for weight_index in range(
        0,
        21
    ):

        xgb_weight = (
            weight_index / 20
        )

        nn_weight = (
            1.0 - xgb_weight
        )

        ensemble_val_prob = (
            xgb_weight * xgb_val_prob
            +
            nn_weight * nn_val_prob
        )

        metrics = calculate_metrics(
            y_val,
            ensemble_val_prob,
            threshold=0.5
        )

        print(
            f"{xgb_weight:<15.2f}"
            f"{nn_weight:<15.2f}"
            f"{metrics['pr_auc']:<20.4f}"
            f"{metrics['f1']:<15.4f}"
        )

        if metrics["pr_auc"] > best_pr_auc:

            best_pr_auc = (
                metrics["pr_auc"]
            )

            best_xgb_weight = (
                xgb_weight
            )

            best_nn_weight = (
                nn_weight
            )

            best_val_prob = (
                ensemble_val_prob.copy()
            )

    print(
        "\n========== BEST ENSEMBLE WEIGHTS =========="
    )

    print(
        f"XGBoost weight: "
        f"{best_xgb_weight:.2f}"
    )

    print(
        f"Neural Network weight: "
        f"{best_nn_weight:.2f}"
    )

    print(
        f"Validation PR-AUC: "
        f"{best_pr_auc:.4f}"
    )

    # ========================================================
    # RECALL-AWARE VALIDATION THRESHOLD
    # ========================================================

    (
        best_threshold,
        threshold_metrics
    ) = find_best_threshold(
        y_val,
        best_val_prob,
        minimum_precision=MINIMUM_PRECISION
    )

    # ========================================================
    # FINAL TEST ENSEMBLE
    # ========================================================

    ensemble_test_prob = (
        best_xgb_weight * xgb_test_prob
        +
        best_nn_weight * nn_test_prob
    )

    # ========================================================
    # TEST @ 0.5
    # ========================================================

    test_metrics_05 = calculate_metrics(
        y_test,
        ensemble_test_prob,
        threshold=0.5
    )

    print(
        "\n=========================================="
    )

    print(
        "========== ENSEMBLE TEST @ 0.5 ========="
    )

    print(
        f"Accuracy: "
        f"{test_metrics_05['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{test_metrics_05['precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{test_metrics_05['recall']:.4f}"
    )

    print(
        f"F1: "
        f"{test_metrics_05['f1']:.4f}"
    )

    print(
        f"ROC-AUC: "
        f"{test_metrics_05['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC: "
        f"{test_metrics_05['pr_auc']:.4f}"
    )

    # ========================================================
    # TEST @ VALIDATION-SELECTED THRESHOLD
    # ========================================================

    final_test_metrics = calculate_metrics(
        y_test,
        ensemble_test_prob,
        threshold=best_threshold
    )

    print(
        "\n=========================================="
    )

    print(
        "====== ENSEMBLE TEST FINAL ============="
    )

    print(
        f"Threshold: "
        f"{best_threshold:.2f}"
    )

    print(
        f"Accuracy: "
        f"{final_test_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{final_test_metrics['precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{final_test_metrics['recall']:.4f}"
    )

    print(
        f"F1: "
        f"{final_test_metrics['f1']:.4f}"
    )

    print(
        f"ROC-AUC: "
        f"{final_test_metrics['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC: "
        f"{final_test_metrics['pr_auc']:.4f}"
    )

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    test_predictions = (
        ensemble_test_prob >= best_threshold
    ).astype(int)

    print(
        "\n========== CLASSIFICATION REPORT =========="
    )

    print(
        classification_report(
            y_test,
            test_predictions,
            target_names=[
                "Legitimate",
                "Fraud"
            ],
            zero_division=0
        )
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        test_predictions,
        labels=[0, 1]
    ).ravel()

    tn = int(tn)
    fp = int(fp)
    fn = int(fn)
    tp = int(tp)

    print(
        "\n========== CONFUSION MATRIX =========="
    )

    print(
        f"True Negative: {tn}"
    )

    print(
        f"False Positive: {fp}"
    )

    print(
        f"False Negative: {fn}"
    )

    print(
        f"True Positive: {tp}"
    )

    # ========================================================
    # FRAUD ANALYSIS
    # ========================================================

    print(
        "\n========== FRAUD ANALYSIS =========="
    )

    print(
        f"Frauds detected: {tp}"
    )

    print(
        f"Frauds missed: {fn}"
    )

    print(
        f"Legitimate transactions "
        f"incorrectly flagged: {fp}"
    )

    # ========================================================
    # TEST COST ANALYSIS
    # ========================================================

    cost_results = analyze_costs(
        y_test,
        ensemble_test_prob,
        false_positive_cost=100,
        false_negative_cost=5000
    )

    # ========================================================
    # SAVE CONFIGURATION
    # ========================================================

    ensemble_config = {

        "model_type":
            "XGBoost + Neural Network",

        "xgb_weight":
            float(best_xgb_weight),

        "nn_weight":
            float(best_nn_weight),

        "production_threshold":
            float(best_threshold),

        "selection_metric":
            "validation_pr_auc",

        "threshold_selection_metric":
            "recall_at_minimum_precision",

        "minimum_precision":
            float(MINIMUM_PRECISION),

        "validation_pr_auc":
            float(best_pr_auc),

        "validation_precision":
            float(
                threshold_metrics["precision"]
            ),

        "validation_recall":
            float(
                threshold_metrics["recall"]
            ),

        "validation_f1":
            float(
                threshold_metrics["f1"]
            ),

        "test_accuracy":
            float(
                final_test_metrics["accuracy"]
            ),

        "test_precision":
            float(
                final_test_metrics["precision"]
            ),

        "test_recall":
            float(
                final_test_metrics["recall"]
            ),

        "test_f1":
            float(
                final_test_metrics["f1"]
            ),

        "test_roc_auc":
            float(
                final_test_metrics["roc_auc"]
            ),

        "test_pr_auc":
            float(
                final_test_metrics["pr_auc"]
            ),

        "confusion_matrix": {

            "true_negative":
                tn,

            "false_positive":
                fp,

            "false_negative":
                fn,

            "true_positive":
                tp
        },

        "test_cost_analysis":
            cost_results
    }

    os.makedirs(
        os.path.dirname(
            ENSEMBLE_RESULTS_PATH
        ),
        exist_ok=True
    )

    with open(
        ENSEMBLE_RESULTS_PATH,
        "w"
    ) as file:

        json.dump(
            ensemble_config,
            file,
            indent=4
        )

    print(
        "\nEnsemble configuration saved to:"
    )

    print(
        ENSEMBLE_RESULTS_PATH
    )

    # ========================================================
    # MLFLOW ENSEMBLE RUN
    # ========================================================

    print(
        "\n=========================================="
    )

    print(
        "========= MLFLOW ENSEMBLE RUN =========="
    )

    print(
        "=========================================="
    )

    with mlflow.start_run(
        run_name="final_tuned_ensemble"
    ):

        mlflow.set_tags({

            "model_type":
                "XGBoost + Neural Network",

            "stage":
                "final_ensemble",

            "selection_metric":
                "validation_pr_auc",

            "threshold_selection":
                "recall_at_minimum_precision",

            "test_set_used_for_selection":
                "false"
        })

        mlflow.log_params({

            "xgb_weight":
                float(best_xgb_weight),

            "nn_weight":
                float(best_nn_weight),

            "production_threshold":
                float(best_threshold),

            "minimum_precision":
                float(MINIMUM_PRECISION),

            "false_positive_cost":
                100,

            "false_negative_cost":
                5000
        })

        mlflow.log_metrics({

            "validation_pr_auc":
                float(best_pr_auc),

            "validation_precision":
                float(
                    threshold_metrics["precision"]
                ),

            "validation_recall":
                float(
                    threshold_metrics["recall"]
                ),

            "validation_f1":
                float(
                    threshold_metrics["f1"]
                ),

            "test_accuracy":
                float(
                    final_test_metrics["accuracy"]
                ),

            "test_precision":
                float(
                    final_test_metrics["precision"]
                ),

            "test_recall":
                float(
                    final_test_metrics["recall"]
                ),

            "test_f1":
                float(
                    final_test_metrics["f1"]
                ),

            "test_roc_auc":
                float(
                    final_test_metrics["roc_auc"]
                ),

            "test_pr_auc":
                float(
                    final_test_metrics["pr_auc"]
                )
        })

        mlflow.log_artifact(
            ENSEMBLE_RESULTS_PATH
        )

        mlflow.log_artifact(
            XGB_PARAMS_PATH
        )

        mlflow.log_artifact(
            NN_PARAMS_PATH
        )

        if os.path.exists(
            XGB_MODEL_PATH
        ):

            mlflow.log_artifact(
                XGB_MODEL_PATH
            )

        if os.path.exists(
            NN_MODEL_PATH
        ):

            mlflow.log_artifact(
                NN_MODEL_PATH
            )

        print(
            "MLflow ensemble logging complete."
        )

        print(
            f"Run ID: "
            f"{mlflow.active_run().info.run_id}"
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print(
        "\n=========================================="
    )

    print(
        "========= ENSEMBLE COMPLETE ============="
    )

    print(
        "=========================================="
    )

    print(
        f"XGBoost weight: "
        f"{best_xgb_weight:.2f}"
    )

    print(
        f"Neural Network weight: "
        f"{best_nn_weight:.2f}"
    )

    print(
        f"Production threshold: "
        f"{best_threshold:.2f}"
    )

    print(
        f"Threshold minimum precision: "
        f"{MINIMUM_PRECISION:.4f}"
    )

    print(
        f"Validation PR-AUC: "
        f"{best_pr_auc:.4f}"
    )

    print(
        f"Validation Precision: "
        f"{threshold_metrics['precision']:.4f}"
    )

    print(
        f"Validation Recall: "
        f"{threshold_metrics['recall']:.4f}"
    )

    print(
        f"Validation F1: "
        f"{threshold_metrics['f1']:.4f}"
    )

    print(
        f"Test PR-AUC: "
        f"{final_test_metrics['pr_auc']:.4f}"
    )

    print(
        f"Test Precision: "
        f"{final_test_metrics['precision']:.4f}"
    )

    print(
        f"Test Recall: "
        f"{final_test_metrics['recall']:.4f}"
    )

    print(
        f"Test F1: "
        f"{final_test_metrics['f1']:.4f}"
    )

    print(
        "\n========== ENSEMBLE TUNING COMPLETE =========="
    )


if __name__ == "__main__":
    main()