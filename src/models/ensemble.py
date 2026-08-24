import numpy as np
from sklearn.metrics import f1_score


THRESHOLDS = np.arange(
    0.10,
    0.91,
    0.05
)


def ensemble_predict(
    xgb_prob,
    nn_prob,
    xgb_weight,
    nn_weight
):
    """
    Combine XGBoost and Neural Network
    fraud probabilities using weighted averaging.
    """

    xgb_prob = np.asarray(
        xgb_prob,
        dtype=np.float32
    )

    nn_prob = np.asarray(
        nn_prob,
        dtype=np.float32
    )

    if len(xgb_prob) != len(nn_prob):
        raise ValueError(
            "XGBoost and Neural Network "
            "probability arrays must have "
            "the same length."
        )

    if not np.isclose(
        xgb_weight + nn_weight,
        1.0
    ):
        raise ValueError(
            "Ensemble weights must sum to 1.0."
        )

    return (
        xgb_weight * xgb_prob
        + nn_weight * nn_prob
    )


def find_best_weight(
    y_val,
    xgb_prob,
    nn_prob
):
    """
    Find the best XGBoost/Neural Network
    weights using validation data.

    For every weight combination, multiple
    thresholds are tested.

    The combination producing the highest
    validation F1 is selected.

    IMPORTANT:
    Only validation data is used here.
    The test set is never used for selection.
    """

    y_val = np.asarray(
        y_val,
        dtype=int
    )

    xgb_prob = np.asarray(
        xgb_prob,
        dtype=np.float32
    )

    nn_prob = np.asarray(
        nn_prob,
        dtype=np.float32
    )

    best_xgb_weight = 0.5
    best_nn_weight = 0.5
    best_threshold = 0.5
    best_f1 = -1.0

    print(
        "\n========== ENSEMBLE "
        "WEIGHT + THRESHOLD SEARCH =========="
    )

    for xgb_weight in np.arange(
        0.0,
        1.01,
        0.05
    ):

        nn_weight = 1.0 - xgb_weight

        ensemble_prob = ensemble_predict(
            xgb_prob,
            nn_prob,
            xgb_weight,
            nn_weight
        )

        for threshold in THRESHOLDS:

            predictions = (
                ensemble_prob >= threshold
            ).astype(int)

            f1 = f1_score(
                y_val,
                predictions,
                zero_division=0
            )

            if f1 > best_f1:

                best_f1 = f1
                best_xgb_weight = (
                    float(xgb_weight)
                )
                best_nn_weight = (
                    float(nn_weight)
                )
                best_threshold = (
                    float(threshold)
                )

    print(
        "\n========== BEST ENSEMBLE CONFIGURATION =========="
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
        f"Threshold: "
        f"{best_threshold:.2f}"
    )

    print(
        f"Validation F1: "
        f"{best_f1:.4f}"
    )

    return (
        best_xgb_weight,
        best_nn_weight,
        best_threshold,
        best_f1
    )


def find_best_threshold(
    y_val,
    probabilities
):
    """
    Find the threshold producing the best
    F1 score on validation data.

    This function is useful when the ensemble
    weights have already been selected.
    """

    y_val = np.asarray(
        y_val,
        dtype=int
    )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float32
    )

    best_threshold = 0.5
    best_f1 = -1.0

    print(
        "\n========== ENSEMBLE "
        "THRESHOLD SEARCH =========="
    )

    for threshold in THRESHOLDS:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        f1 = f1_score(
            y_val,
            predictions,
            zero_division=0
        )

        print(
            f"Threshold: {threshold:.2f} | "
            f"F1: {f1:.4f}"
        )

        if f1 > best_f1:

            best_f1 = f1
            best_threshold = float(
                threshold
            )

    print(
        "\n========== BEST ENSEMBLE THRESHOLD =========="
    )

    print(
        f"Threshold: "
        f"{best_threshold:.2f}"
    )

    print(
        f"Validation F1: "
        f"{best_f1:.4f}"
    )

    return (
        best_threshold,
        best_f1
    )