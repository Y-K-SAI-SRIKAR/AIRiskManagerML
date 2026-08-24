import os
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report
)

def evaluate_model(model,X_test,y_test):

    """
    Evaluate a trained binary classification model
    on the untouched test dataset.
    """

    # Probability of fraud
    y_prob = model.predict_proba(X_test)[:, 1]

    # Default classification threshold
    threshold = 0.5
    y_pred = (y_prob >= threshold).astype(int)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    print("\n========== TEST SET EVALUATION ==========")

    print(f"Threshold:  {threshold}")
    print(f"Accuracy:   {accuracy:.4f}")
    print(f"Precision:  {precision:.4f}")
    print(f"Recall:     {recall:.4f}")
    print(f"F1 Score:   {f1:.4f}")
    print(f"ROC-AUC:    {roc_auc:.4f}")
    print(f"PR-AUC:     {pr_auc:.4f}")

    print("\n========== CLASSIFICATION REPORT ==========")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Legitimate", "Fraud"],
            zero_division=0
        )
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    }