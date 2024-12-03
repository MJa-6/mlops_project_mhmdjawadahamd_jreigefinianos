# ML Pipeline with MLOps Integration

This project demonstrates a machine learning pipeline with integrated MLOps best practices. It includes configuration management, CI/CD workflows, monitoring, and deployment, following industry standards.

## Features
- **End-to-End ML Pipeline**: Data loading, preprocessing, model training, and evaluation.
- **Experiment Tracking**: Integrated with MLflow for tracking experiments, metrics, and model artifacts.
- **Monitoring**: Prometheus metrics for monitoring pipeline performance and exposing custom metrics.
- **Containerization**: Dockerized for portability and consistent deployment.
- **CI/CD**: Automated workflows using GitHub Actions.

## Prerequisites
To run this project, you need:
- Python 3.11+
- Poetry (Python dependency manager)
- Docker and Docker Compose (for containerized deployment)

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/MJa-6/mlops_project_mhmdjawadahamd_jreigefinianos.git
   cd mlops_project_mhmdjawadahamd_jreigefinianos
2. **Install Dependencies**:
   poetry install
3. **Configure the Project**:
   Modify the configuration file located at config/config.yaml to adjust dataset paths, model parameters, and other settings.

## Usage

### 1. Run Locally

Use Poetry to run the pipeline with the specified configuration:

`poetry run train config/config.yaml`

### 2. Run in Docker

Build and run the pipeline in a containerized environment:

`docker-compose up`

### 3. Access Metrics

> Prometheus Metrics: http://localhost:8000/metrics
> Prometheus UI: http://localhost:9090

### 4. Experiment Tracking with MLflow

Start the MLflow UI locally to view experiment results:

`poetry run mlflow ui`

Access the MLflow UI at: http://localhost:5000

## Testing

To run tests locally:

`poetry run pytest`

## Monitoring

The pipeline exposes custom metrics (e.g., inference request count, model accuracy) to Prometheus, which can be visualized in the Prometheus web interface.