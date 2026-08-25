import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader


# ============================================================
# SPARSE DATASET
# ============================================================

class SparseDataset(Dataset):
    """
    Dataset wrapper for scipy sparse matrices.

    Sparse rows are kept sparse until a batch is created.
    The batch is then converted to a dense PyTorch tensor.
    """

    def __init__(self, X, y):

        self.X = X

        self.y = np.asarray(
            y,
            dtype=np.float32
        )

    def __len__(self):

        return len(self.y)

    def __getitem__(self, index):

        return (
            self.X[index],
            self.y[index]
        )


# ============================================================
# SPARSE COLLATE
# ============================================================

def sparse_collate(batch):
    """
    Convert a batch of sparse rows into a dense PyTorch tensor.
    """

    X_batch = np.vstack([
        row.toarray()
        for row, _ in batch
    ]).astype(
        np.float32,
        copy=False
    )

    y_batch = np.asarray(
        [
            y
            for _, y in batch
        ],
        dtype=np.float32
    )

    return (
        torch.from_numpy(X_batch),
        torch.from_numpy(y_batch)
    )


# ============================================================
# FRAUD NEURAL NETWORK
# ============================================================

class FraudNeuralNetwork(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_1=512,
        hidden_2=256,
        hidden_3=64,
        dropout_1=0.4,
        dropout_2=0.3,
        dropout_3=0.2
    ):

        super().__init__()

        self.network = nn.Sequential(

            # ----------------------------------------------
            # Layer 1
            # ----------------------------------------------

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

            # ----------------------------------------------
            # Layer 2
            # ----------------------------------------------

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

            # ----------------------------------------------
            # Layer 3
            # ----------------------------------------------

            nn.Linear(
                hidden_2,
                hidden_3
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout_3
            ),

            # ----------------------------------------------
            # Output
            # ----------------------------------------------

            nn.Linear(
                hidden_3,
                1
            )
        )

    def forward(self, x):

        return self.network(
            x
        ).squeeze(1)


# ============================================================
# TRAIN NEURAL NETWORK
# ============================================================

def train_neural_network(
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=10,
    batch_size=1024,
    learning_rate=0.001,
    hidden_1=512,
    hidden_2=256,
    hidden_3=64,
    dropout_1=0.4,
    dropout_2=0.3,
    dropout_3=0.2,
    weight_decay=0.0001,
    patience=3
):

    # ========================================================
    # DEVICE
    # ========================================================

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

    # ========================================================
    # DATASETS
    # ========================================================

    train_dataset = SparseDataset(
        X_train,
        y_train
    )

    val_dataset = SparseDataset(
        X_val,
        y_val
    )

    use_pin_memory = (
        device.type == "cuda"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=sparse_collate,
        pin_memory=use_pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=sparse_collate,
        pin_memory=use_pin_memory
    )

    # ========================================================
    # MODEL
    # ========================================================

    input_size = X_train.shape[1]

    model = FraudNeuralNetwork(
        input_size=input_size,
        hidden_1=hidden_1,
        hidden_2=hidden_2,
        hidden_3=hidden_3,
        dropout_1=dropout_1,
        dropout_2=dropout_2,
        dropout_3=dropout_3
    ).to(device)

    # ========================================================
    # CLASS IMBALANCE
    # ========================================================

    positive = np.sum(
        np.asarray(y_train) == 1
    )

    negative = np.sum(
        np.asarray(y_train) == 0
    )

    if positive == 0:

        raise ValueError(
            "Training data contains "
            "no positive fraud samples."
        )

    pos_weight_value = (
        negative / positive
    )

    pos_weight = torch.tensor(
        [pos_weight_value],
        dtype=torch.float32,
        device=device
    )

    print(
        f"NN positive samples: "
        f"{positive}"
    )

    print(
        f"NN negative samples: "
        f"{negative}"
    )

    print(
        f"NN pos_weight: "
        f"{pos_weight_value:.4f}"
    )

    # ========================================================
    # LOSS
    # ========================================================

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # ========================================================
    # EARLY STOPPING
    # ========================================================

    best_val_loss = float(
        "inf"
    )

    best_state = None

    epochs_without_improvement = 0

    # ========================================================
    # TRAINING
    # ========================================================

    for epoch in range(
        epochs
    ):

        model.train()

        train_loss = 0.0

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

            train_loss += (
                loss.item()
                * len(y_batch)
            )

        train_loss /= len(
            train_dataset
        )

        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()

        val_loss = 0.0

        with torch.no_grad():

            for X_batch, y_batch in val_loader:

                X_batch = X_batch.to(
                    device,
                    non_blocking=True
                )

                y_batch = y_batch.to(
                    device,
                    non_blocking=True
                )

                logits = model(
                    X_batch
                )

                loss = criterion(
                    logits,
                    y_batch
                )

                val_loss += (
                    loss.item()
                    * len(y_batch)
                )

        val_loss /= len(
            val_dataset
        )

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss:.5f} "
            f"Val Loss: {val_loss:.5f}"
        )

        # ====================================================
        # BEST MODEL
        # ====================================================

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_state = {
                key: value.detach().cpu().clone()
                for key, value
                in model.state_dict().items()
            }

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        # ====================================================
        # EARLY STOPPING
        # ====================================================

        if (
            epochs_without_improvement
            >= patience
        ):

            print(
                f"Early stopping triggered "
                f"after {epoch + 1} epochs."
            )

            break

    # ========================================================
    # RESTORE BEST MODEL
    # ========================================================

    if best_state is None:

        raise RuntimeError(
            "Neural Network training did not "
            "produce a valid model state."
        )

    model.load_state_dict(
        best_state
    )

    model = model.to(
        device
    )

    print(
        "\nNeural Network training completed."
    )

    print(
        f"Best validation loss: "
        f"{best_val_loss:.6f}"
    )

    return model


# ============================================================
# PREDICTION
# ============================================================

def predict_neural_network(
    model,
    X,
    batch_size=1024
):
    """
    Return fraud probabilities.
    """

    device = next(
        model.parameters()
    ).device

    dummy_y = np.zeros(
        X.shape[0],
        dtype=np.float32
    )

    dataset = SparseDataset(
        X,
        dummy_y
    )

    use_pin_memory = (
        device.type == "cuda"
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=sparse_collate,
        pin_memory=use_pin_memory
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

    if not probabilities:

        return np.array(
            [],
            dtype=np.float32
        )

    return np.concatenate(
        probabilities
    )