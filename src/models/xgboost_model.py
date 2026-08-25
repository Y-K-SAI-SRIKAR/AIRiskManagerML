import os

import xgboost as xgb


def train_xgboost(
    X_train,
    y_train,
    X_val,
    y_val,
    model_path="models/xgboost_model.json",

    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0,
    reg_alpha=0,
    reg_lambda=1,
    scale_pos_weight=None,
    random_state=42,
    early_stopping_rounds=50
):

    # ==========================================
    # CLASS IMBALANCE
    # ==========================================

    negative = (
        y_train == 0
    ).sum()

    positive = (
        y_train == 1
    ).sum()

    if positive == 0:
        raise ValueError(
            "Training data contains "
            "no positive fraud samples."
        )

    calculated_scale_pos_weight = (
        negative / positive
    )

    if scale_pos_weight is None:
        scale_pos_weight = (
            calculated_scale_pos_weight
        )

    print(
        "\n========== XGBOOST =========="
    )

    print(
        f"Negative samples: {negative}"
    )

    print(
        f"Positive samples: {positive}"
    )

    print(
        f"Scale pos weight: "
        f"{scale_pos_weight:.4f}"
    )

    # ==========================================
    # MODEL
    # ==========================================

    model = xgb.XGBClassifier(

        n_estimators=n_estimators,

        learning_rate=learning_rate,

        max_depth=max_depth,

        min_child_weight=min_child_weight,

        subsample=subsample,

        colsample_bytree=colsample_bytree,

        gamma=gamma,

        reg_alpha=reg_alpha,

        reg_lambda=reg_lambda,

        scale_pos_weight=scale_pos_weight,

        objective="binary:logistic",

        eval_metric="aucpr",

        tree_method="hist",

        random_state=random_state,

        n_jobs=-1
    )

    # ==========================================
    # TRAIN
    # ==========================================

    print(
        "\nTraining XGBoost..."
    )

    model.fit(

        X_train,

        y_train,

        eval_set=[
            (
                X_val,
                y_val
            )
        ],

        verbose=50
    )

    # ==========================================
    # BEST ITERATION
    # ==========================================
    #
    # Your installed XGBoost version is not
    # exposing best_iteration because early
    # stopping is not active here.
    #
    # Therefore don't access best_iteration
    # blindly.
    # ==========================================

    try:

        best_iteration = (
            model.best_iteration
        )

    except AttributeError:

        best_iteration = (
            n_estimators - 1
        )

    try:

        best_score = (
            model.best_score
        )

    except AttributeError:

        # Get the final validation score
        evaluation_results = (
            model.evals_result()
        )

        validation_scores = (
            evaluation_results[
                "validation_0"
            ][
                "aucpr"
            ]
        )

        best_score = max(
            validation_scores
        )

    # ==========================================
    # SAVE MODEL
    # ==========================================

    os.makedirs(
        os.path.dirname(
            model_path
        ),
        exist_ok=True
    )

    model.save_model(
        model_path
    )

    print(
        "\nXGBoost training completed."
    )

    print(
        f"Best iteration: "
        f"{best_iteration}"
    )

    print(
        f"Best score: "
        f"{best_score}"
    )

    print(
        f"Model saved to: "
        f"{model_path}"
    )

    return model