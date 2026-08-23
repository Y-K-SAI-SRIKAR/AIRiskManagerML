import pandas as pd
print(pd.__version__)

path1 = 'data/raw/train_identity.csv'
path2 = 'data/raw/train_transaction.csv'

def load_data(path1, path2):

    identity = pd.read_csv(path1)
    transaction = pd.read_csv(path2)

    print("Shape of Identity:", identity.shape)
    print("Shape of Transaction:", transaction.shape)

    merged = transaction.merge(identity,on="TransactionID",how="left")

    print("Shape of Merged:", merged.shape)
    print("Initiating export...")

    merged.to_csv("data/processed/merged.csv",index=False)
    print("Export completed successfully!")


load_data(path1, path2)