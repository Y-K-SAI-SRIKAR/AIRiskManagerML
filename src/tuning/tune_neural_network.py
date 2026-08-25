import os
import json

import mlflow
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)

from torch.utils.data import DataLoader

from src.data.split_data import split_data
from src.data.model_processing import process_data
from src.utils.mlflow_config import configure_mlflow

from src.models.neural_network import (
    FraudNeuralNetwork,
    SparseDataset,
    sparse_collate
)


DATA_PATH = (
    "data/processed/"
    "feature_engineered.csv"
)

BEST_PARAMS_PATH = (
    "models/best_neural_network_params.json"
)


# ==========================================
# TUNABLE NETWORK
# ==========================================

class TunableFraudNeuralNetwork(
    nn.Module
):

    def __init__(
        self,
        input_size,
        hidden_1,
        hidden_2,
        hidden_3,
        dropout_1,
        dropout_2,
        dropout_3
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_size,
                hidden_1
            ),

            nn.ReLU(),

            nn.BatchNorm1d(
                hidden_1
            ),

            nn.Dropout(
                dropout_1
            ),

            nn.Linear(
                hidden_1,
                hidden_2
            ),

            nn.ReLU(),

            nn.BatchNorm1d(
                hidden_2
            ),

            nn.Dropout(
                dropout_2
            ),

            nn.Linear(
                hidden_2,
                hidden_3
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout_3
            ),

            nn.Linear(
                hidden_3,
                1
            )
        )

    def forward(
        self,
        x
    ):

        return self.network(
            x
        ).squeeze(1)


def predict_model(
    model,
    X,
    batch_size=1024
):

    device = next(
        model.parameters()
    ).device

    dummy_y = torch.zeros(
        X.shape[0],
        dtype=torch.float32
    )

    dataset = SparseDataset(
        X,
        dummy_y.numpy()
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=sparse_collate
    )

    model.eval()

    probabilities = []

    with torch.no_grad():

        for X_batch, _ in loader:

            X_batch = X_batch.to(
                device,
                non_blocking=True
            )

            logits = model(
                X_batch
            )

            probs = torch.sigmoid(
                logits
            )

            probabilities.append(
                probs.cpu().numpy()
            )

    return (
        __import__("numpy")
        .concatenate(probabilities)
    )


def evaluate(
    y_true,
    probabilities
):

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    return {

        "pr_auc":
            average_precision_score(
                y_true,
                probabilities
            ),

        "roc_auc":
            roc_auc_score(
                y_true,
                probabilities
            ),

        "precision":
            precision_score(
                y_true,
                predictions,
                zero_division=0
            ),

        "recall":
            recall_score(
                y_true,
                predictions,
                zero_division=0
            ),

        "f1":
            f1_score(
                y_true,
                predictions,
                zero_division=0
            )
    }


def train_trial(
    X_train,
    y_train,
    X_val,
    y_val,
    params
):

    device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu")

    print(
        f"Neural Network device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = TunableFraudNeuralNetwork(
        input_size=X_train.shape[1],

        hidden_1=params[
            "hidden_1"
        ],

        hidden_2=params[
            "hidden_2"
        ],

        hidden_3=params[
            "hidden_3"
        ],

        dropout_1=params[
            "dropout_1"
        ],

        dropout_2=params[
            "dropout_2"
        ],

        dropout_3=params[
            "dropout_3"
        ]
    ).to(device)

    train_dataset = SparseDataset(
        X_train,
        y_train
    )


    train_loader = DataLoader(
    train_dataset,
    batch_size=params["batch_size"],
    shuffle=True,
    collate_fn=sparse_collate,
    pin_memory=(device.type == "cuda"))

    positive = (
        y_train == 1
    ).sum()

    negative = (
        y_train == 0
    ).sum()

    pos_weight_value = (
        negative / positive
    )

    pos_weight = torch.tensor(
        [pos_weight_value],
        dtype=torch.float32,
        device=device
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params[
            "learning_rate"
        ],
        weight_decay=params[
            "weight_decay"
        ]
    )

    best_pr_auc = -1.0
    best_state = None
    patience_counter = 0

    for epoch in range(
        params["epochs"]
    ):

        # ======================================
        # TRAIN
        # ======================================

        model.train()

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(
            device,
            non_blocking=True
            )

            y_batch = y_batch.to(
            device,
            non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                X_batch
            )

            loss = criterion(
                logits,
                y_batch
            )

            loss.backward()

            optimizer.step()

        # ======================================
        # VALIDATION
        # ======================================

        val_prob = predict_model(
            model,
            X_val,
            params["batch_size"]
        )

        val_metrics = evaluate(
            y_val,
            val_prob
        )

        print(
            f"Epoch "
            f"{epoch + 1}/"
            f"{params['epochs']} | "
            f"PR-AUC: "
            f"{val_metrics['pr_auc']:.4f} | "
            f"ROC-AUC: "
            f"{val_metrics['roc_auc']:.4f} | "
            f"F1: "
            f"{val_metrics['f1']:.4f}"
        )

        # ======================================
        # EARLY STOPPING
        # ======================================

        if (
            val_metrics["pr_auc"]
            > best_pr_auc
        ):

            best_pr_auc = (
                val_metrics["pr_auc"]
            )

            best_state = {
                key: value.detach()
                .cpu()
                .clone()

                for key, value
                in model.state_dict().items()
            }

            patience_counter = 0

        else:

            patience_counter += 1

        if (
            patience_counter
            >= params["patience"]
        ):

            print(
                "Early stopping."
            )

            break

    model.load_state_dict(
        best_state
    )

    final_prob = predict_model(
        model,
        X_val,
        params["batch_size"]
    )

    final_metrics = evaluate(
        y_val,
        final_prob
    )

    return (
        model,
        final_metrics
    )


def main():

    configure_mlflow()

    print(
        "\n========== NEURAL NETWORK "
        "TUNING =========="
    )

    # ==========================================
    # LOAD DATA
    # ==========================================

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

    input_size = (
        X_train_encoded.shape[1]
    )

    print(
        f"NN input size: "
        f"{input_size}"
    )

    # ==========================================
    # SEARCH SPACE
    # ==========================================

    trials = [

        {
            "hidden_1": 256,
            "hidden_2": 128,
            "hidden_3": 64,
            "dropout_1": 0.30,
            "dropout_2": 0.30,
            "dropout_3": 0.20,
            "learning_rate": 0.001,
            "batch_size": 1024,
            "weight_decay": 1e-4,
            "epochs": 15,
            "patience": 3
        },

        {
            "hidden_1": 512,
            "hidden_2": 256,
            "hidden_3": 128,
            "dropout_1": 0.30,
            "dropout_2": 0.30,
            "dropout_3": 0.20,
            "learning_rate": 0.0005,
            "batch_size": 1024,
            "weight_decay": 1e-4,
            "epochs": 15,
            "patience": 3
        },

        {
            "hidden_1": 256,
            "hidden_2": 128,
            "hidden_3": 64,
            "dropout_1": 0.20,
            "dropout_2": 0.20,
            "dropout_3": 0.10,
            "learning_rate": 0.0005,
            "batch_size": 1024,
            "weight_decay": 1e-5,
            "epochs": 15,
            "patience": 3
        },

        {
            "hidden_1": 512,
            "hidden_2": 256,
            "hidden_3": 64,
            "dropout_1": 0.40,
            "dropout_2": 0.30,
            "dropout_3": 0.20,
            "learning_rate": 0.0005,
            "batch_size": 2048,
            "weight_decay": 1e-4,
            "epochs": 15,
            "patience": 3
        },

        {
            "hidden_1": 256,
            "hidden_2": 64,
            "hidden_3": 32,
            "dropout_1": 0.20,
            "dropout_2": 0.20,
            "dropout_3": 0.10,
            "learning_rate": 0.001,
            "batch_size": 2048,
            "weight_decay": 1e-4,
            "epochs": 15,
            "patience": 3
        }
    ]

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
            f"NEURAL NETWORK TRIAL "
            f"{trial_number}/"
            f"{len(trials)}"
        )

        print(
            "=========================================="
        )

        with mlflow.start_run(
            run_name=(
                f"nn_tuning_"
                f"{trial_number:02d}"
            )
        ):

            mlflow.set_tags({

                "model_type":
                    "Neural Network",

                "stage":
                    "tuning",

                "tuning_trial":
                    str(trial_number)
            })

            mlflow.log_params(
                params
            )

            mlflow.log_param(
                "input_size",
                input_size
            )

            model, metrics = (
                train_trial(
                    X_train_encoded,
                    y_train,
                    X_val_encoded,
                    y_val,
                    params
                )
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
                f"\nValidation PR-AUC: "
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
            # BEST
            # ==================================

            if (
                metrics["pr_auc"]
                > best_pr_auc
            ):

                best_pr_auc = (
                    metrics["pr_auc"]
                )

                best_params = (
                    params.copy()
                )

                best_metrics = (
                    metrics.copy()
                )

                torch.save(
                    model.state_dict(),
                    "models/"
                    "best_neural_network.pt"
                )

                print(
                    "\n*** NEW BEST "
                    "NEURAL NETWORK ***"
                )

    # ==========================================
    # SAVE RESULTS
    # ==========================================

    result = {

        "model":
            "Neural Network",

        "selection_metric":
            "validation_pr_auc",

        "best_validation_pr_auc":
            best_pr_auc,

        "best_validation_roc_auc":
            best_metrics["roc_auc"],

        "best_validation_f1":
            best_metrics["f1"],

        "best_params":
            best_params
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
        "===== BEST NEURAL NETWORK CONFIG ====="
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

    print(
        "\nBest model saved to:"
    )

    print(
        "models/best_neural_network.pt"
    )


if __name__ == "__main__":
    main()