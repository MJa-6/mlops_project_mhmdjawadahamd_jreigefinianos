import argparse
import mlflow
from prometheus_client import start_http_server, Counter, Gauge, Summary
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from ml_data_pipeline.config import load_config
from ml_data_pipeline.data_loader import get_data_loader
from loguru import logger
import time

# Configure logging
logger.add("debug.log", format="{time} {level} {message}", level="DEBUG")

# Prometheus Metrics
inference_requests = Counter('inference_requests_total', 'Total number of inference requests made')
model_accuracy = Gauge('model_accuracy', 'Model accuracy')
data_processing_time = Summary('data_processing_seconds', 'Time spent processing data')

# Configure MLflow
mlflow.set_tracking_uri('file:///C:/Masters AI/Semester 3/Software Engineering for AI/Project/Solution/mlops_project_mhmdjawadahamd_jreigefinianos/ml_data_pipeline/mlruns')
mlflow.set_experiment("Heart_Disease_Prediction")

@data_processing_time.time()  # Measure data processing time
def preprocess_data(data):
    """Simulate data preprocessing"""
    logger.info("Preprocessing data...")
    time.sleep(2)  # Simulate some processing delay
    return data

def main(config_path):
    # Start the Prometheus metrics server
    start_http_server(8000)  # Exposes metrics at http://localhost:8000/metrics
    logger.info("Prometheus metrics server started on port 8000")

    # Load configuration using the given config path
    config = load_config(config_path)

    # Get the appropriate data loader based on the file type
    loader = get_data_loader(config.data_loader.file_type)
    data = loader.load_data(config.data_loader.file_path)
    logger.info("Data was loaded")
    print("Loaded Data:")
    print(data.head())

    # Preprocess data
    data = preprocess_data(data)

    # Prepare data for training
    X = data.drop("HeartDisease", axis=1)
    y = data["HeartDisease"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.model.random_state
    )

    # Initialize and train model within an MLflow run
    with mlflow.start_run():
        model = LogisticRegression(random_state=config.model.random_state, max_iter=3000)
        model.fit(X_train, y_train)

        # Evaluate the model
        score = model.score(X_test, y_test)
        print(f"Model accuracy: {score:.2f}")
        logger.info(f"Model accuracy: {score:.2f}")

        # Update Prometheus metrics
        inference_requests.inc()  # Increment inference request counter
        model_accuracy.set(score)  # Update accuracy metric in Prometheus

        # Log parameters, metrics, and the model to MLflow
        mlflow.log_params({
            "random_state": config.model.random_state,
            "max_iter": 3000,
            "model_type": "LogisticRegression"
        })
        mlflow.log_metric("accuracy", score)
        mlflow.sklearn.log_model(model, "model", signature=mlflow.models.infer_signature(X_train, model.predict(X_train)), input_example=X_train.iloc[:1])

    # Keep the application running for Prometheus scraping
    while True:
        time.sleep(10)  # Simulate periodic pipeline runs

if __name__ == "__main__":
    # Setup argparse to read the path to the configuration file
    parser = argparse.ArgumentParser(description="Run the ML pipeline with specified configuration")
    parser.add_argument("config_path", type=str, help="Path to the configuration file")
    args = parser.parse_args()

    main(args.config_path)
