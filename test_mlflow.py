import mlflow
from src.utils.mlflow_config import configure_mlflow


configure_mlflow()

with mlflow.start_run(run_name="connection_test"):

    mlflow.log_param("test", "mysql_connection")
    mlflow.log_metric("test_metric", 1.0)

    print("MLflow connection successful!")
    print("Run ID:", mlflow.active_run().info.run_id)