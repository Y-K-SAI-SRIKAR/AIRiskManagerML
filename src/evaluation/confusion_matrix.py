from sklearn.metrics import confusion_matrix


def evaluate_confusion_matrix(model, X_test, y_test, threshold=0.5):

    # Fraud probabilities
    y_prob = model.predict_proba(X_test)[:, 1]

    # Convert probabilities to predictions
    y_pred = (y_prob >= threshold).astype(int)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()

    print("\n========== CONFUSION MATRIX ==========")

    print(f"Threshold:      {threshold}")
    print(f"True Negative:  {tn}")
    print(f"False Positive: {fp}")
    print(f"False Negative: {fn}")
    print(f"True Positive:  {tp}")

    print("\n========== FRAUD ANALYSIS ==========")

    print(f"Frauds detected: {tp}")
    print(f"Frauds missed:   {fn}")
    print(f"Legitimate transactions incorrectly flagged: {fp}")

    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }