
# LossLess Engine (AI Risk Manager) : ML Layer

A production-grade fraud detection system built on MLOps principles. 
Provides end-to-end machine learning capabilities from data preparation through model deployment, governance, and inference.

## Overview

The AI Risk Manager ML Layer transforms transaction data into fraud risk assessments via a supervised classification pipeline. The system integrates data engineering, feature engineering, model development, experiment tracking, model governance, and scalable inference into a unified workflow.

**Core responsibility:** Convert transaction-level signals into calibrated fraud probability scores and binary risk decisions with full traceability and explainability.

## System Architecture


````mermaid
flowchart TD
    A["Raw Transaction Data"] --> B["Data Preprocessing<br/>& Feature Engineering"]
    B --> C["Train / Validate / Test"]
    C --> D["XGBoost"]
    C --> E["Neural Network"]
    D --> F["Ensemble / Threshold<br/>Optimization"]
    E --> F
    F --> G["Model Evaluation<br/>PR-AUC / Precision<br/>Recall / F1 / ROC"]
    G --> H["MLflow<br/>Tracking + Registry"]
    H --> I["S3 Artifacts"]
    I --> J["Docker + FastAPI"]
    J --> K["Render<br/>Production API"]
    
    style A fill:#ffffff,stroke:#000000,stroke-width:2px,color:#000000
    style B fill:#ffffff,stroke:#000000,stroke-width:2px,color:#000000
    style C fill:#ffffff,stroke:#000000,stroke-width:2px,color:#000000
    style D fill:#ffffff,stroke:#000000,stroke-width:2px,color:#000000
    style E fill:#ffffff,stroke:#000000,stroke-width:2px,color:#000000
    style F fill:#ffffff,stroke:#000000,stroke-width:2px,color:#000000
    style G fill:#ffffff,stroke:#000000,stroke-width:2px,color:#000000
    style H fill:#ffffff,stroke:#000000,stroke-width:2px,color:#000000
    style I fill:#ffffff,stroke:#000000,stroke-width:2px,color:#000000
    style J fill:#ffffff,stroke:#000000,stroke-width:2px,color:#000000
    style K fill:#ffffff,stroke:#000000,stroke-width:2px,color:#000000
````


The ML Layer operates between raw transaction data and application-level risk workflows:

```
Transaction Input
    ↓ 
Data Pipeline 
    ↓ 
Feature Layer
    ↓ 
XGBoost + Neural Network Ensemble
    ↓ 
Fraud Risk Score [0, 1]
    ↓ 
Binary Classification (Fraud / Legitimate)
    ↓
Risk Report + Application Layer
```

----------

## Technology Stack

Category

Technologies

**ML & Data**

Python 3.11 · XGBoost · PyTorch · Scikit-learn · Pandas · NumPy · SHAP 

**MLOps**

MLflow 3.15.2 · Model Registry · Experiment Tracking · Model Versioning · DVC

**Storage & Compute**

AWS S3 · AWS RDS MySQL · CSV-based transactions

**API & Serving**

FastAPI · Uvicorn · Batch inference pipelines

**Infrastructure**

Docker · Docker Hub · Render (cloud deployment)

**Config & Reproducibility**

YAML parameters · Environment variables · Artifact versioning

----------

## Machine Learning Approach

### Model Selection

**XGBoost (Primary)**

-   Chosen for structured transaction data containing non-linear relationships, mixed feature types, and class imbalance
-   Handles missing values natively and captures complex feature interactions

**Neural Network (Alternative)**

-   PyTorch-based architecture for comparative analysis and learned non-linear representations
-   Enables ensemble diversity

**Ensemble Configuration**

-   Configurable weighted combination of both models
-   Production default: XGBoost + Neural Network with equal weighting

### Feature Engineering

The system maintains 117 production features across multiple categories:

-   **Temporal:** Transaction hour, day, week, weekday
-   **Amount transformations:** Log scaling, statistical aggregations
-   **Card-based:** Frequency features, card type, historical patterns
-   **Identity signals:** Email domain matching, device fingerprinting
-   **Behavioral:** Missing value indicators, categorical attributes
-   **Relational:** Cross-feature interaction signals

Feature definitions are configuration-driven and decoupled from model logic, ensuring reproducibility across training and inference.

----------

## Data Processing Pipeline

### 1. **Data Loading & Cleaning**

-   Handles CSV-based transaction datasets
-   Validates schema, detects anomalies
-   Manages missing values strategically

### 2. **Preprocessing**

-   Numerical feature normalization and transformation
-   Categorical encoding
-   Missing value imputation with tracking
-   Preprocessor persisted as `models/preprocessor.pkl` to ensure train-serving consistency

### 3. **Train/Validation/Test Split**

-   **Training Set:** Parameter learning
-   **Validation Set:** Hyperparameter tuning, threshold selection, model comparison
-   **Test Set:** Independent final evaluation to estimate generalization

Strict separation prevents test data leakage into model selection.

### 4. **Feature Transformation**

-   Identical pipeline applied to all subsets
-   Enforced consistency between training and production inference

----------

## Model Development & Evaluation

### Training & Hyperparameter Tuning

-   Cross-validation with stratified splits
-   Grid/random search for hyperparameter optimization
-   Experiment tracking via MLflow for reproducibility

### Model Evaluation

-   **Metrics:** ROC-AUC, PR-AUC, F1 (at configurable thresholds), precision, recall
-   **Threshold Optimization:** Validated against business cost functions
-   **Imbalance Handling:** Class weights, SMOTE, threshold adjustment

### Explainability

-   SHAP (SHapley Additive exPlanations) analysis for model transparency
-   Feature importance rankings
-   Decision explanations for high-value transactions
-   Auditability and regulatory compliance support

----------

## MLOps & Model Governance

### Experiment Tracking (MLflow)

-   All training runs tracked with parameters, metrics, and artifacts
-   Enables reproducibility and comparison across experiments
-   Automatic logging of model performance, hyperparameters, and feature sets

### Model Registry

-   Centralized model versioning and lifecycle management
-   Promotion workflow: Candidate → Staging → Production
-   Quality gate evaluation before promotion
-   **Champion Model:** Single designated production model with versioning

### Quality Gates

-   Automated performance thresholds before production deployment
-   Validation metrics compared against baseline
-   Prevents regression in production

----------

## Production Deployment

### Containerization

```dockerfile
Docker image with:
- Python 3.11 runtime
- Serialized preprocessor
- Registered champion model
- FastAPI inference server

```

### Deployment Pipeline

1.  Model Registry → Docker image build
2.  Push to Docker Hub
3.  Deploy to Render (or target cloud platform)
4.  Champion model available via REST API

### Inference Modes

**Real-time (API)**

-   FastAPI endpoint accepts single transactions
-   Returns fraud probability and binary decision
-   Sub-100ms latency target

**Batch Processing**

-   Large-scale transaction prediction
-   Chunk-based processing for memory efficiency
-   Results written to S3 for reporting and audit

### Stateless Design

-   Deployed model never modifies or retrains
-   All state tracked via MLflow
-   Enables horizontal scaling

----------

## Configuration & Reproducibility

### Parameters (`params.yaml`)

```yaml
model:
  type: ensemble
  xgboost_weight: 0.5
  nn_weight: 0.5
  
threshold:
  production: 0.50  
  
training:
  train_size: 0.70
  val_size: 0.15
  test_size: 0.15

```

### Environment Management

-   `.env` provides template for secrets
-   Credentials (AWS keys, database URLs) via environment variables
-   Never committed to source control

### Artifact Management

-   Model files (>100MB) stored in MLflow/S3, not in Git through DVC
-   Preprocessor serialized and versioned
-   Configuration remains version-controlled

----------

## Project Structure

```
AIRiskManagerML/
├── src/
│   ├── api/
│   │   └── app.py                    # FastAPI inference server
│   │
│   ├── data/
│   │   ├── load_data.py              # CSV/S3 loading
│   │   ├── preprocess.py             # Preprocessing pipeline
│   │   ├── split_data.py             # Train/val/test split
│   │   └── model_processing.py       # Data validation
│   │
│   ├── features/
│   │   └── feature_config.py         # 117 feature definitions
│   │
│   ├── models/
│   │   ├── xgboost_model.py          # XGBoost training
│   │   ├── neural_network.py         # PyTorch model
│   │   └── ensemble.py               # Ensemble aggregation
│   │
│   ├── training/
│   │   ├── train.py                  # Main training loop
│   │   └── retrain.py                # Retraining pipeline
│   │
│   ├── tuning/
│   │   └── hyperparameter_tuning.py  # HPO & cross-validation
│   │
│   ├── evaluation/
│   │   └── evaluate.py               # Metric computation
│   │
│   ├── registry/
│   │   ├── register_model.py         # Model logging to MLflow
│   │   └── promote_model.py          # Promotion workflow
│   │
│   ├── inference/
│   │   ├── predict.py                # Single inference
│   │   └── batch_predict.py          # Batch processing
│   │
│   ├── explainability/
│   │   └── shap_analysis.py          # SHAP feature importance
│   │
│   └── utils/
│       └── mlflow_config.py          # MLflow initialization
│
├── models/
│   ├── preprocessor.pkl              # Serialized preprocessing
│   └── best_ensemble_config.json     # Ensemble weights
│
├── params.yaml                       # Training parameters
├── Dockerfile                        # Container specification
├── requirements.txt                  # Dependencies
├── .env.example                      # Environment template
└── .gitignore

```

----------

## Engineering Principles

1.  **Reproducibility:** Complete configuration and artifact versioning
2.  **Separation of Concerns:** Data, features, modeling, evaluation, serving isolated into modules
3.  **Model Governance:** Gated promotion via MLflow Registry
4.  **Configuration-Driven:** Thresholds and weights configurable, not hard-coded
5.  **Artifact Management:** Large models stored externally (MLflow/S3)
6.  **Stateless Inference:** No training during prediction; enables scaling
7.  **Train-Serving Consistency:** Identical preprocessing in training and production
8.  **Explainability:** SHAP-based decision transparency
9.  **Environment Isolation:** Docker ensures dev/prod parity
10.  **Secret Management:** Credentials via environment, never version-controlled

----------

## End-to-End Lifecycle

```
Historical Data
    ↓
Data Preparation & Cleaning
    ↓
Feature Engineering (117 features)
    ↓
Train/Tune/Validate
    ↓
MLflow Experiment Tracking
    ↓
Model Registry Submission
    ↓
Quality Gate Evaluation
    ↓
Champion Model Selection
    ↓
Docker Image Build
    ↓
Cloud Deployment
    ↓
Production Inference (Real-time + Batch)
    ↓
Risk Reports & Explainability
    ↓
Ground Truth Collection
    ↓
[Cycle: Retraining when justified by drift, performance degradation, or scheduled intervals]

```

----------

## Quick Start

### Prerequisites

-   Python 3.11+
-   Docker
-   AWS credentials (S3/RDS access)
-   MLflow tracking server

### Setup

```bash
# Clone and install
git clone https://github.com/Y-K-SAI-SRIKAR/AIRiskManagerML
cd AIRiskManagerML
pip install -r requirements.txt

# Configure environment
cp .env
# Edit .env with your AWS keys, database URL, MLflow URI

# Train a model
python -m src.training.train --config params.yaml

# Serve locally
python -m src.api.app
# API available at http://localhost:8000/docs

```

### Production Deployment

```bash
# Build image
docker build -t air-risk-manager:ml .

# Push to registry
docker push <registry>/air-risk-manager:ml
```

----------

## Monitoring & Maintenance

### Key Metrics to Track

-   **Model Performance:** Real-time vs. baseline ROC-AUC, calibration
-   **Data Quality:** Missing rates, distribution shifts, anomalies
-   **Inference Latency:** API response times, batch throughput
-   **Prediction Distribution:** Changes in fraud probability output

### Retraining Triggers

-   Scheduled retraining 
-   Performance degradation detected (>5% AUC drop)
-   Data distribution drift signals
-   Sufficient new ground-truth labels accumulated

### Model Rollback

-   Previous champion model versions retained in MLflow
-   Fast rollback via model alias update if new champion underperforms

----------

## Security & Compliance

-   No transaction PII in logs or artifacts
-   Model predictions are explainable for regulatory audit
-   All model versions and training data tracked in MLflow
-   Environment isolation via Docker
-   Cloud deployment with encrypted S3 storage

----------

## License

This project is licensed under the MIT License. see the [LICENSE](LICENSE) file for details.

----------

**Maintained by:** YERRAGUNTLA KAMESWARA SAI SRIKAR
**Last Updated:** September 03, 2026.