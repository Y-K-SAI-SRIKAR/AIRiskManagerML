import pandas as pd
path = 'data/processed/merged.csv'

def preprocess(path):

    data = pd.read_csv(path)
    print("Merged DataSet Loaded Successfully!")

    #Shape and Colums in the dataset:
    print(f"Shape of DataSet:{data.shape}\n")
    cols = list(data.columns)
    print(f"List of Columns:\n{cols}\n")

    #Data Types:
    print(f"DataType:{data.dtypes}\n")

    #Identifying missing values:
    print(f"Missing values in DataSet of first 15 cols and 50 rows:\n{data.iloc[0:50,0:15].isna()}\n")

    missing_count = data.isnull().sum()
    missing_percent = (missing_count / len(data)) * 100

    missing_info = pd.DataFrame({"missing_count": missing_count,"missing_percent": missing_percent})

    missing_info = missing_info[
    missing_info["missing_count"] > 0].sort_values(by="missing_percent",ascending=False)

    print("\n========== MISSING VALUE ANALYSIS ==========")
    print(missing_info)

    missing_threshold = 80
    high_missing_cols = missing_info[missing_info["missing_percent"] >= missing_threshold].index.tolist()

    print("\n========== HIGH MISSING COLUMNS ==========")
    print(f"Threshold: {missing_threshold}%")
    print(f"Columns to remove: {len(high_missing_cols)}")
    print(high_missing_cols)
    data = data.drop(columns=high_missing_cols)

    #Handling missing values with median imputation:
    numeric_cols = data.select_dtypes(include=["int64", "float64"]).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())


    #HAndling Missing Catagorical colums with Missing imputation:
    categorical_cols = data.select_dtypes(include=["object"]).columns
    data[categorical_cols] = data[categorical_cols].fillna("Missing")
    print("\n========== AFTER IMPUTATION ==========")
    print(f"Remaining missing values: {data.isnull().sum().sum()}")

    #Removing redundant entries:
    cleaned = data.drop_duplicates(keep='first')
    print(f"Shape of Cleaned DataSet:{cleaned.shape}")

    #Removing redundant transactions:
    cleaned = cleaned.drop_duplicates(subset=["TransactionID"],keep='first')
    print(f"Shaped Transaction cleaned Dataset:{cleaned.shape}")

    #Reduction of data: (Removing non required features)
    cleaned = cleaned.drop(columns=['TransactionID','DeviceType','DeviceInfo'],errors='ignore')


    def remove_correlated_features(cleaned, feature_cols, target="isFraud", threshold=0.65):

        # Correlation between selected features
        corr_matrix = cleaned[feature_cols].corr().abs()

        # Correlation of features with target
        target_corr = (
            cleaned[feature_cols + [target]]
            .corr()[target]
            .drop(target)
            .abs()
        )

        columns_to_drop = set()
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):

                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]

                feature_corr = corr_matrix.iloc[i, j]
                if pd.isna(feature_corr):
                    continue

                if feature_corr >= threshold:
                    if target_corr[col1] >= target_corr[col2]:
                        columns_to_drop.add(col2)
                    else:
                        columns_to_drop.add(col1)

        cleaned = cleaned.drop(columns=list(columns_to_drop))

        print("\n========== FEATURE REDUCTION ==========")
        print(f"Threshold: {threshold}")
        print(f"Features before: {len(feature_cols)}")
        print(f"Features dropped: {len(columns_to_drop)}")
        print(f"Features after: {len(cleaned.columns)}")

        return cleaned

    d_cols = [col for col in cleaned.columns if col.startswith("D")]
    cleaned = remove_correlated_features(cleaned,d_cols,threshold=0.65)
    c_cols = [col for col in cleaned.columns if col.startswith("C")]
    cleaned = remove_correlated_features(cleaned,c_cols,threshold=0.65)
    v_cols = [col for col in cleaned.columns if col.startswith("V")]
    cleaned = remove_correlated_features(cleaned,v_cols,threshold=0.65)

    output_path = "data/processed/cleaned.csv"
    cleaned.to_csv(output_path, index=False)
    print(f"\nCleaned dataset saved to: {output_path}")

    return cleaned
preprocess(path)