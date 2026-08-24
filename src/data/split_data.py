from sklearn.model_selection import train_test_split
import pandas as pd

path = "data/processed/feature_engineered.csv"
data = pd.read_csv(path)

def split_data(data):

    if "TransactionAmt_Bin" in data.columns:
        data = data.drop(columns=["TransactionAmt_Bin"])

    X = data.drop(columns=["isFraud"])
    Y = data["isFraud"]

    # 70% train, 30% temporary
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        Y,
        test_size=0.30,
        random_state=42,
        stratify=Y
    )

    # Split remaining 30% into 15% validation and 15% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    print("Splitting of Data Completed Successfully")

    print(f"Train:      {X_train.shape}")
    print(f"Validation: {X_val.shape}")
    print(f"Test:       {X_test.shape}")

    print("\nFraud rates:")
    print(f"Train:      {y_train.mean():.4%}")
    print(f"Validation: {y_val.mean():.4%}")
    print(f"Test:       {y_test.mean():.4%}")


    categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X_train.select_dtypes(exclude=["object"]).columns.tolist()

    print(f"Categorical features: {len(categorical_cols)}")
    print(f"Numeric features: {len(numeric_cols)}")
    print("\nCategorical columns:")
    print(categorical_cols)

    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(data)

