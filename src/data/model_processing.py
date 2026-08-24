import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def create_preprocessor(X_train):

    categorical_cols = X_train.select_dtypes(
        include=["object"]
    ).columns.tolist()

    numerical_cols = X_train.select_dtypes(
        exclude=["object"]
    ).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True
                ),
                categorical_cols
            ),
            (
                "numerical",
                "passthrough",
                numerical_cols
            )
        ]
    )

    return preprocessor, categorical_cols, numerical_cols


def process_data(
    X_train,
    X_val,
    X_test,
    save_path="models/preprocessor.pkl"
):

    # Create preprocessing pipeline
    preprocessor, categorical_cols, numerical_cols = create_preprocessor(
        X_train
    )

    # Fit ONLY on training data
    X_train_encoded = preprocessor.fit_transform(X_train)

    # Transform validation and test using training-fitted encoder
    X_val_encoded = preprocessor.transform(X_val)
    X_test_encoded = preprocessor.transform(X_test)

    # Save fitted preprocessor
    joblib.dump(preprocessor, save_path)

    print("\n========== MODEL PREPROCESSING ==========")
    print(f"Categorical features: {len(categorical_cols)}")
    print(f"Numerical features: {len(numerical_cols)}")
    print(f"Encoded features: {X_train_encoded.shape[1]}")
    print(f"Train shape: {X_train_encoded.shape}")
    print(f"Validation shape: {X_val_encoded.shape}")
    print(f"Test shape: {X_test_encoded.shape}")
    print(f"Preprocessor saved to: {save_path}")

    return (
        X_train_encoded,
        X_val_encoded,
        X_test_encoded,
        preprocessor
    )