import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


def analyze_thresholds(y_test, y_prob):

    thresholds = np.arange(0.10, 0.91, 0.05)

    results = []

    print("\n========== THRESHOLD ANALYSIS ==========")
    print(
        f"{'Threshold':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1':<12}"
        f"{'Fraud Flags':<15}"
    )

    for threshold in thresholds:

        y_pred = (y_prob >= threshold).astype(int)

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        fraud_flags = y_pred.sum()

        results.append({
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
        )

    # Best threshold based on F1
    best_result = max(
        results,
        key=lambda x: x["f1"]
    )

    print("\n========== BEST F1 THRESHOLD ==========")
    print(f"Threshold: {best_result['threshold']:.2f}")
    print(f"Precision: {best_result['precision']:.4f}")
    print(f"Recall:    {best_result['recall']:.4f}")
    print(f"F1 Score:  {best_result['f1']:.4f}")

    return results