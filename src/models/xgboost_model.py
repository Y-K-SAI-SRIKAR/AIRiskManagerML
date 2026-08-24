import os
import joblib

from xgboost import XGBClassifier


def train_xgboost(
    X_train,
    y_train,
    X_val,
    y_val,
    model_path="models/xgboost_model.json"):

    # Calculate class imbalance from training data
    negative = (y_train == 0).sum()
    positive = (y_train == 1).sum()

    scale_pos_weight = negative / positive

    print("\n========== XGBOOST ==========")
    print(f"Negative samples: {negative}")
    print(f"Positive samples: {positive}")
    print(f"Scale pos weight: {scale_pos_weight:.4f}")

    model = XGBClassifier(
        objective="binary:logistic",

        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,

        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,

        reg_alpha=0.1,
        reg_lambda=1.0,

        scale_pos_weight=scale_pos_weight,

        tree_method="hist",

        eval_metric="aucpr",

        random_state=42,
        n_jobs=-1,

        early_stopping_rounds=50
    )

    print("\nTraining XGBoost...")

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=50
    )

    os.makedirs(
        os.path.dirname(model_path),
        exist_ok=True
    )

    model.save_model(model_path)

    print("\nXGBoost training completed.")
    print(f"Best iteration: {model.best_iteration}")
    print(f"Best score: {model.best_score}")
    print(f"Model saved to: {model_path}")

    return model