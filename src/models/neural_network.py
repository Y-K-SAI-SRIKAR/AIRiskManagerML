import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader


class SparseDataset(Dataset):
    """
    Dataset wrapper for scipy sparse matrices.

    Sparse rows are kept sparse until a batch is
    created. The batch is then converted to dense.
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


def sparse_collate(batch):
    """
    Convert a batch of sparse rows into a
    single dense PyTorch tensor.

    This is considerably more efficient than
    converting every individual row separately.
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


class FraudNeuralNetwork(nn.Module):

    def __init__(
        self,
        input_size
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_size,
                256
            ),

            nn.ReLU(),

            nn.BatchNorm1d(
                256
            ),

            nn.Dropout(
                0.30
            ),

            nn.Linear(
                256,
                128
            ),

            nn.ReLU(),

            nn.BatchNorm1d(
                128
            ),

            nn.Dropout(
                0.30
            ),

            nn.Linear(
                128,
                64
            ),

            nn.ReLU(),

            nn.Dropout(
                0.20
            ),

            nn.Linear(
                64,
                1
            )
        )

    def forward(self, x):

        return self.network(
            x
        ).squeeze(1)


def train_neural_network(
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=10,
    batch_size=1024,
    learning_rate=0.001
):

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Neural Network device: "
        f"{device}"
    )

    # ==========================================
    # DATASETS
    # ==========================================

    train_dataset = SparseDataset(
        X_train,
        y_train
    )

    val_dataset = SparseDataset(
        X_val,
        y_val
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=sparse_collate,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=sparse_collate,
        pin_memory=torch.cuda.is_available()
    )

    # ==========================================
    # MODEL
    # ==========================================

    input_size = X_train.shape[1]

    model = FraudNeuralNetwork(
        input_size=input_size
    ).to(device)

    # ==========================================
    # CLASS IMBALANCE
    # ==========================================

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

    # ==========================================
    # LOSS
    # ==========================================

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    # ==========================================
    # OPTIMIZER
    # ==========================================

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    # ==========================================
    # BEST MODEL TRACKING
    # ==========================================

    best_val_loss = float(
        "inf"
    )

    best_state = None

    # ==========================================
    # TRAINING
    # ==========================================

    for epoch in range(epochs):

        model.train()

        train_loss = 0.0

        for X_batch, y_batch in train_loader:

            if torch.cuda.is_available():
                X_batch = X_batch.pin_memory()
                y_batch = y_batch.pin_memory()

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

        # ======================================
        # VALIDATION
        # ======================================

        model.eval()

        val_loss = 0.0

        with torch.no_grad():

            for X_batch, y_batch in val_loader:

                if torch.cuda.is_available():
                    X_batch = X_batch.pin_memory()
                    y_batch = y_batch.pin_memory()

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

        # ======================================
        # SAVE BEST MODEL
        # ======================================

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_state = {
                key: value.detach().cpu().clone()
                for key, value
                in model.state_dict().items()
            }

    # ==========================================
    # RESTORE BEST MODEL
    # ==========================================

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

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=sparse_collate,
        pin_memory=torch.cuda.is_available()
    )

    model.eval()

    probabilities = []

    with torch.no_grad():

        for X_batch, _ in loader:

            if torch.cuda.is_available():
                X_batch = X_batch.pin_memory()

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