import numpy as np

from src.evaluation.metrics import (
    evaluate_probabilities
)


def test_metrics_return_required_keys():

    y_true = np.array([
        0, 1, 0, 1,
        1, 0
    ])

    y_prob = np.array([
        0.10,
        0.90,
        0.20,
        0.80,
        0.70,
        0.30
    ])

    metrics = evaluate_probabilities(
        y_true,
        y_prob,
        threshold=0.5,
        title="UNIT TEST"
    )

    required = {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc"
    }

    assert required.issubset(metrics.keys())


def test_metrics_range():

    y_true = np.array([
        0, 1, 0, 1
    ])

    y_prob = np.array([
        0.05,
        0.95,
        0.20,
        0.80
    ])

    metrics = evaluate_probabilities(
        y_true,
        y_prob,
        threshold=0.5
    )

    for key in [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc"
    ]:

        assert 0 <= metrics[key] <= 1


def test_threshold_changes_predictions():

    y_true = np.array([
        0, 1, 0, 1
    ])

    y_prob = np.array([
        0.45,
        0.55,
        0.60,
        0.40
    ])

    metrics_low = evaluate_probabilities(
        y_true,
        y_prob,
        threshold=0.4
    )

    metrics_high = evaluate_probabilities(
        y_true,
        y_prob,
        threshold=0.6
    )

    assert metrics_low["recall"] >= metrics_high["recall"]