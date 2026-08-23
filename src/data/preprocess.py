import pandas as pd
path = 'data/processed/merged.csv'

def preprocess(path):

    data = pd.read_csv(path)
    print("Merged DataSet Loaded Successfully!")

    #Shape and Colums in the dataset:
    print(f"Shape of DataSet:{data.shape}\n")
    print(f"List of Columns:\n{[data.columns]}\n")
    cols = [data.columns]

    #Data Types:
    print(f"DataType:{data.dtypes}\n")

    #Identifying missing values:
    print(f"Missing values in DataSet of first 15 cols and 50 rows:\n{data.iloc[0:50,0:15].isna()}\n")

    #Handling missing values with Zero imputation:
    data = data.fillna(0)
    print(f"Handled Data:\n{data.isnull()}")

    #Removing redundant entries:
    cleaned = data.drop_duplicates(keep=True)
    print(f"Shape of Cleaned DataSet:{cleaned.shape}")

    #Removing redundant transactions:
    cleaned = cleaned.drop_duplicates(subset=["TransactionID"])
    print(f"Shaped Transaction cleaned Dataset:{cleaned.shape}")

    #Reduction of data: (Removing non required features)
    cleaned.drop(columns=['TransactionID','TransactionDT','DeviceType','DeviceInfo'])


preprocess(path)