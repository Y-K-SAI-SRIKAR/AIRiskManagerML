import os
import subprocess
from dotenv import load_dotenv

load_dotenv()

user = os.getenv("MLFLOW_DB_USER")
password = os.getenv("MLFLOW_DB_PASSWORD")
host = os.getenv("MLFLOW_DB_HOST")
port = os.getenv("MLFLOW_DB_PORT")
database = os.getenv("MLFLOW_DB_NAME")

tracking_uri = (
    f"mysql+pymysql://"
    f"{user}:{password}@"
    f"{host}:{port}/"
    f"{database}"
)

print("Starting MLflow UI...")
print(f"Backend: MySQL database '{database}'")
print(f"Host: {host}")

subprocess.run([
    "mlflow",
    "ui",
    "--backend-store-uri",
    tracking_uri
])