from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "merged.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned.csv"


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess(
    input_path=DEFAULT_INPUT_PATH,
    output_path=DEFAULT_OUTPUT_PATH,
    missing_threshold=80.0,
    correlation_threshold=0.65,
):
    """
    Clean the merged transaction dataset.

    This stage performs:
        1. Dataset loading
        2. Missing-value analysis
        3. High-missing-column removal
        4. Missing-value imputation
        5. Duplicate removal
        6. TransactionID duplicate removal
        7. Removal of non-model identifiers
        8. Correlation-based feature reduction
        9. Saving cleaned dataset

    IMPORTANT:
    Model-specific preprocessing such as:
        - OneHotEncoding
        - Scaling
        - train/validation/test fitting

    should remain in the model preprocessing pipeline.
    """

    input_path = Path(input_path)
    output_path = Path(output_path)

    print("\n==========================================")
    print("========== DATA PREPROCESSING ===========")
    print("==========================================")

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input dataset not found:\n{input_path}"
        )

    print(f"\nLoading dataset from:\n{input_path}")

    data = pd.read_csv(input_path)

    print("\nMerged Dataset Loaded Successfully!")
    print(f"Dataset shape: {data.shape}")

    # --------------------------------------------------------
    # BASIC INFORMATION
    # --------------------------------------------------------

    print("\n========== DATA TYPES ==========")
    print(data.dtypes)

    print("\n========== MISSING VALUE ANALYSIS ==========")

    missing_count = data.isnull().sum()

    missing_percent = (
        missing_count / len(data)
    ) * 100

    missing_info = pd.DataFrame(
        {
            "missing_count": missing_count,
            "missing_percent": missing_percent,
        }
    )

    missing_info = (
        missing_info[
            missing_info["missing_count"] > 0
        ]
        .sort_values(
            by="missing_percent",
            ascending=False,
        )
    )

    print(missing_info)

    # --------------------------------------------------------
    # REMOVE HIGH-MISSING FEATURES
    # --------------------------------------------------------

    high_missing_cols = missing_info[
        missing_info["missing_percent"] >= missing_threshold
    ].index.tolist()

    print("\n========== HIGH MISSING COLUMNS ==========")
    print(
        f"Threshold: {missing_threshold}%"
    )
    print(
        f"Columns to remove: {len(high_missing_cols)}"
    )

    if high_missing_cols:
        print(high_missing_cols)

        data = data.drop(
            columns=high_missing_cols,
            errors="ignore",
        )
    else:
        print("No columns exceed the missing-value threshold.")

    # --------------------------------------------------------
    # HANDLE NUMERICAL MISSING VALUES
    # --------------------------------------------------------

    numeric_cols = data.select_dtypes(
        include=[np.number]
    ).columns

    if len(numeric_cols) > 0:

        medians = data[numeric_cols].median()

        data[numeric_cols] = data[numeric_cols].fillna(
            medians
        )

    # --------------------------------------------------------
    # HANDLE CATEGORICAL MISSING VALUES
    # --------------------------------------------------------

    categorical_cols = data.select_dtypes(
        include=[
            "object",
            "category",
            "string",
        ]
    ).columns

    if len(categorical_cols) > 0:

        data[categorical_cols] = (
            data[categorical_cols]
            .fillna("Missing")
        )

    print("\n========== AFTER IMPUTATION ==========")

    remaining_missing = int(
        data.isnull().sum().sum()
    )

    print(
        f"Remaining missing values: {remaining_missing}"
    )

    # --------------------------------------------------------
    # REMOVE EXACT DUPLICATES
    # --------------------------------------------------------

    before_duplicates = len(data)

    cleaned = data.drop_duplicates(
        keep="first"
    ).copy()

    removed_duplicates = (
        before_duplicates - len(cleaned)
    )

    print("\n========== DUPLICATE REMOVAL ==========")
    print(
        f"Duplicate rows removed: {removed_duplicates}"
    )
    print(
        f"Shape after duplicate removal: {cleaned.shape}"
    )

    # --------------------------------------------------------
    # REMOVE DUPLICATE TRANSACTIONS
    # --------------------------------------------------------

    if "TransactionID" in cleaned.columns:

        before_transactions = len(cleaned)

        cleaned = cleaned.drop_duplicates(
            subset=["TransactionID"],
            keep="first",
        ).copy()

        removed_transactions = (
            before_transactions - len(cleaned)
        )

        print(
            f"Duplicate TransactionIDs removed: "
            f"{removed_transactions}"
        )

    # --------------------------------------------------------
    # REMOVE NON-MODEL IDENTIFIERS
    # --------------------------------------------------------

    columns_to_remove = [
        "TransactionID",
        "DeviceType",
        "DeviceInfo",
    ]

    existing_columns = [
        column
        for column in columns_to_remove
        if column in cleaned.columns
    ]

    if existing_columns:

        cleaned = cleaned.drop(
            columns=existing_columns,
            errors="ignore",
        )

    print("\n========== NON-MODEL FEATURES ==========")
    print(
        f"Removed columns: {existing_columns}"
    )

    # --------------------------------------------------------
    # CORRELATION FEATURE REDUCTION
    # --------------------------------------------------------

    def remove_correlated_features(
        dataframe,
        feature_cols,
        threshold=0.65,
    ):
        """
        Remove highly correlated features.

        This function intentionally does NOT use the target
        variable to decide which feature survives.

        That avoids target leakage during the dataset-level
        preprocessing stage.
        """

        feature_cols = [
            column
            for column in feature_cols
            if column in dataframe.columns
        ]

        if len(feature_cols) < 2:
            return dataframe, []

        numeric_features = dataframe[
            feature_cols
        ].select_dtypes(
            include=[np.number]
        ).columns.tolist()

        if len(numeric_features) < 2:
            return dataframe, []

        corr_matrix = (
            dataframe[numeric_features]
            .corr()
            .abs()
        )

        upper_triangle = np.triu(
            np.ones(
                corr_matrix.shape,
                dtype=bool,
            ),
            k=1,
        )

        upper = corr_matrix.where(
            upper_triangle
        )

        columns_to_drop = [
            column
            for column in upper.columns
            if any(
                upper[column] >= threshold
            )
        ]

        dataframe = dataframe.drop(
            columns=columns_to_drop,
            errors="ignore",
        )

        print("\n========== FEATURE REDUCTION ==========")
        print(
            f"Threshold: {threshold}"
        )
        print(
            f"Features before: "
            f"{len(numeric_features)}"
        )
        print(
            f"Features dropped: "
            f"{len(columns_to_drop)}"
        )
        print(
            f"Features remaining: "
            f"{len(dataframe.columns)}"
        )

        return dataframe, columns_to_drop

    # --------------------------------------------------------
    # D FEATURES
    # --------------------------------------------------------

    d_cols = [
        column
        for column in cleaned.columns
        if column.startswith("D")
        and pd.api.types.is_numeric_dtype(
            cleaned[column]
        )
    ]

    cleaned, dropped_d = remove_correlated_features(
        cleaned,
        d_cols,
        correlation_threshold,
    )

    # --------------------------------------------------------
    # C FEATURES
    # --------------------------------------------------------

    c_cols = [
        column
        for column in cleaned.columns
        if column.startswith("C")
        and pd.api.types.is_numeric_dtype(
            cleaned[column]
        )
    ]

    cleaned, dropped_c = remove_correlated_features(
        cleaned,
        c_cols,
        correlation_threshold,
    )

    # --------------------------------------------------------
    # V FEATURES
    # --------------------------------------------------------

    v_cols = [
        column
        for column in cleaned.columns
        if column.startswith("V")
        and pd.api.types.is_numeric_dtype(
            cleaned[column]
        )
    ]

    cleaned, dropped_v = remove_correlated_features(
        cleaned,
        v_cols,
        correlation_threshold,
    )

    # --------------------------------------------------------
    # FINAL VALIDATION
    # --------------------------------------------------------

    print("\n========== FINAL DATASET ==========")

    print(
        f"Final shape: {cleaned.shape}"
    )

    print(
        f"Remaining missing values: "
        f"{cleaned.isnull().sum().sum()}"
    )

    if "isFraud" in cleaned.columns:

        print(
            f"Fraud rate: "
            f"{cleaned['isFraud'].mean() * 100:.4f}%"
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cleaned.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nCleaned dataset saved to:\n"
        f"{output_path}"
    )

    print("\n========== PREPROCESSING COMPLETE ==========")

    return cleaned


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":

    preprocess()