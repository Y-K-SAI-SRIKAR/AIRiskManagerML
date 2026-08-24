import pandas as pd
from src.data.split_data import split_data
from src.data.model_processing import process_data
from src.models.xgboost_model import train_xgboost
from src.evaluation.metrics import evaluate_model
from src.evaluation.confusion_matrix import evaluate_confusion_matrix

DATA_PATH = "data/processed/feature_engineered.csv"

def main():

    print("Loading dataset...")
    data = pd.read_csv(DATA_PATH)

    # ==============================
    # DATA SPLIT
    # ==============================

    (X_train,X_val,X_test,y_train,y_val,y_test) = split_data(data)

    # ==============================
    # ENCODING
    # ==============================

    (X_train_encoded,X_val_encoded,X_test_encoded,preprocessor) = process_data(X_train,X_val,X_test)

    # ==============================
    # XGBOOST
    # ==============================

    model = train_xgboost(X_train_encoded,y_train,X_val_encoded,y_val)
    evaluate_model(model,X_test_encoded,y_test)
    evaluate_confusion_matrix(model,X_test_encoded,y_test)

    print("\n========== TRAINING COMPLETE + EVALUATION COMPLETE ==========")


if __name__ == "__main__":
    main()