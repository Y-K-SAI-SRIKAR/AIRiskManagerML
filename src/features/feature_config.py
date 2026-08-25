from typing import List


# ============================================================
# TARGET
# ============================================================

TARGET_COLUMN = "isFraud"


# ============================================================
# IDENTIFIER COLUMNS
# ============================================================

ID_COLUMNS: List[str] = [
    "TransactionID",
]


# ============================================================
# CATEGORICAL FEATURES
# ============================================================

CATEGORICAL_FEATURES: List[str] = [
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
    "M7",
    "M8",
    "M9",
    "id_12",
    "id_15",
    "id_16",
    "id_28",
    "id_29",
    "id_31",
    "id_35",
    "id_36",
    "id_37",
    "id_38",
    "CardType",
]


# ============================================================
# CURRENT MODEL SCHEMA
# ============================================================

EXPECTED_CATEGORICAL_COUNT = 25

EXPECTED_NUMERIC_COUNT = 92

EXPECTED_RAW_FEATURE_COUNT = (
    EXPECTED_CATEGORICAL_COUNT
    + EXPECTED_NUMERIC_COUNT
)

EXPECTED_ENCODED_FEATURE_COUNT = 422


# ============================================================
# FEATURE ACCESSORS
# ============================================================

def get_categorical_features() -> List[str]:
    """
    Return a copy of the categorical feature list.
    """

    return CATEGORICAL_FEATURES.copy()


def get_numeric_features(df) -> List[str]:
    """
    Determine numerical features from the dataframe.

    Numerical features are all columns that are not:

    - target
    - identifiers
    - known categorical features
    """

    excluded_columns = (
        set(CATEGORICAL_FEATURES)
        | set(ID_COLUMNS)
        | {TARGET_COLUMN}
    )

    numeric_features = [
        column
        for column in df.columns
        if column not in excluded_columns
    ]

    return numeric_features


def get_model_features(df) -> List[str]:
    """
    Return all raw model features.

    Categorical features are returned first, followed by
    numerical features.
    """

    numeric_features = get_numeric_features(df)

    return (
        CATEGORICAL_FEATURES.copy()
        + numeric_features
    )


# ============================================================
# FEATURE VALIDATION
# ============================================================

def validate_required_columns(df) -> None:
    """
    Validate that the dataframe contains all required columns.
    """

    required_columns = (
        set(CATEGORICAL_FEATURES)
        | set(ID_COLUMNS)
        | {TARGET_COLUMN}
    )

    missing_columns = sorted(
        required_columns
        - set(df.columns)
    )

    if missing_columns:

        raise ValueError(
            "Missing required feature columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing_columns
            )
        )


def validate_feature_schema(df) -> None:
    """
    Validate the current dataframe against the known
    production training schema.

    This prevents accidental changes to the feature set
    from silently reaching the trained model.
    """

    validate_required_columns(df)

    numeric_features = get_numeric_features(df)

    categorical_count = len(
        CATEGORICAL_FEATURES
    )

    numeric_count = len(
        numeric_features
    )

    raw_feature_count = (
        categorical_count
        + numeric_count
    )

    if (
        categorical_count
        != EXPECTED_CATEGORICAL_COUNT
    ):

        raise ValueError(
            "Categorical feature count mismatch. "
            f"Expected "
            f"{EXPECTED_CATEGORICAL_COUNT}, "
            f"got {categorical_count}."
        )

    if (
        numeric_count
        != EXPECTED_NUMERIC_COUNT
    ):

        raise ValueError(
            "Numerical feature count mismatch. "
            f"Expected "
            f"{EXPECTED_NUMERIC_COUNT}, "
            f"got {numeric_count}."
        )

    if (
        raw_feature_count
        != EXPECTED_RAW_FEATURE_COUNT
    ):

        raise ValueError(
            "Raw feature count mismatch. "
            f"Expected "
            f"{EXPECTED_RAW_FEATURE_COUNT}, "
            f"got {raw_feature_count}."
        )


# ============================================================
# FEATURE SUMMARY
# ============================================================

def get_feature_summary(df) -> dict:
    """
    Return the complete feature configuration summary.
    """

    validate_feature_schema(df)

    numeric_features = (
        get_numeric_features(df)
    )

    return {

        "target_column":
            TARGET_COLUMN,

        "id_columns":
            ID_COLUMNS.copy(),

        "categorical_features":
            CATEGORICAL_FEATURES.copy(),

        "numeric_features":
            numeric_features,

        "categorical_count":
            len(CATEGORICAL_FEATURES),

        "numeric_count":
            len(numeric_features),

        "raw_feature_count":
            (
                len(CATEGORICAL_FEATURES)
                + len(numeric_features)
            ),

        "expected_encoded_feature_count":
            EXPECTED_ENCODED_FEATURE_COUNT,
    }


# ============================================================
# ENCODED FEATURE VALIDATION
# ============================================================

def validate_encoded_feature_count(
    encoded_data
) -> None:
    """
    Validate the encoded matrix before it is passed
    to the trained model.
    """

    if len(encoded_data.shape) != 2:

        raise ValueError(
            "Encoded feature matrix must be 2-dimensional."
        )

    encoded_feature_count = (
        encoded_data.shape[1]
    )

    if (
        encoded_feature_count
        != EXPECTED_ENCODED_FEATURE_COUNT
    ):

        raise ValueError(
            "Encoded feature count mismatch. "
            f"Expected "
            f"{EXPECTED_ENCODED_FEATURE_COUNT}, "
            f"got {encoded_feature_count}."
        )