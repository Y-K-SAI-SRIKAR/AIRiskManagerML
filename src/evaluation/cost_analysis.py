import numpy as np

def analyze_costs(
    y_test,
    y_prob,
    false_positive_cost=100,
    false_negative_cost=5000):

    thresholds = np.arange(0.10, 0.91, 0.05)

    results = []

    print("\n========== COST ANALYSIS ==========")

    print(f"False Positive Cost: ₹{false_positive_cost}")
    print(f"False Negative Cost: ₹{false_negative_cost}")

    print(
        f"\n{'Threshold':<12}"
        f"{'FP':<10}"
        f"{'FN':<10}"
        f"{'Total Cost':<15}")

    for threshold in thresholds:

        y_pred = (y_prob >= threshold).astype(int)

        # Confusion matrix values without needing another prediction call
        tn = np.sum((y_test == 0) & (y_pred == 0))
        fp = np.sum((y_test == 0) & (y_pred == 1))
        fn = np.sum((y_test == 1) & (y_pred == 0))
        tp = np.sum((y_test == 1) & (y_pred == 1))

        total_cost = (
            fp * false_positive_cost
            + fn * false_negative_cost
        )

        results.append({
            "threshold": threshold,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "total_cost": total_cost
        })

        print(
            f"{threshold:<12.2f}"
            f"{fp:<10}"
            f"{fn:<10}"
            f"₹{total_cost:<14,.0f}"
        )

    # Find minimum-cost threshold
    best_result = min(
        results,
        key=lambda x: x["total_cost"]
    )

    print("\n========== MINIMUM COST THRESHOLD ==========")

    print(f"Threshold:   {best_result['threshold']:.2f}")

    print(f"False Positives: {best_result['fp']}")

    print(f"False Negatives: {best_result['fn']}")

    print(f"Total Cost: ₹{best_result['total_cost']:,.0f}")

    return results