from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.inference.predict import (
    predict_transaction,
    load_production_config,
    load_production_model,
    REGISTERED_MODEL_NAME,
    CHAMPION_ALIAS,
)
from src.utils.mlflow_config import configure_mlflow


app = FastAPI(
    title="AI Risk Manager",
    description=(
        "Production fraud-risk prediction, "
        "model benchmarking, and recall/precision trade-off API"
    ),
    version="1.3.0",
)


# ============================================================
# REQUEST MODELS
# ============================================================

class TransactionRequest(BaseModel):
    transaction: dict


# ============================================================
# MLFLOW HELPERS
# ============================================================

def get_mlflow_client():
    """
    Create an MLflow client using the project's
    configured tracking/database environment.
    """

    configure_mlflow()

    from mlflow.tracking import MlflowClient

    return MlflowClient()


def get_model_versions():
    """
    Retrieve every registered version of the production model.
    """

    client = get_mlflow_client()

    versions = client.search_model_versions(
        f"name='{REGISTERED_MODEL_NAME}'"
    )

    return sorted(
        versions,
        key=lambda version: int(version.version),
    )


def get_version_metrics(version):
    """
    Retrieve metrics and metadata directly from
    the MLflow run associated with a model version.
    """

    client = get_mlflow_client()

    run = client.get_run(
        version.run_id
    )

    metrics = run.data.metrics
    tags = run.data.tags

    return {
        "version": int(version.version),
        "run_id": version.run_id,
        "status": version.status,
        "aliases": list(version.aliases),
        "model_status": tags.get(
            "model_status"
        ),

        # Validation metrics
        "validation_pr_auc": metrics.get(
            "validation_pr_auc"
        ),
        "validation_precision": metrics.get(
            "validation_precision"
        ),
        "validation_recall": metrics.get(
            "validation_recall"
        ),
        "validation_f1": metrics.get(
            "validation_f1"
        ),

        # Test metrics
        "test_accuracy": metrics.get(
            "test_accuracy"
        ),
        "test_precision": metrics.get(
            "test_precision"
        ),
        "test_recall": metrics.get(
            "test_recall"
        ),
        "test_f1": metrics.get(
            "test_f1"
        ),
        "test_roc_auc": metrics.get(
            "test_roc_auc"
        ),
        "test_pr_auc": metrics.get(
            "test_pr_auc"
        ),
    }


def safe_change(current, baseline):
    """
    Calculate absolute and percentage change.

    Returns None when either metric is unavailable.
    """

    if current is None or baseline is None:
        return {
            "absolute_change": None,
            "percentage_change": None,
        }

    current = float(current)
    baseline = float(baseline)

    absolute_change = current - baseline

    if baseline == 0:
        percentage_change = None
    else:
        percentage_change = (
            absolute_change
            / abs(baseline)
            * 100
        )

    return {
        "absolute_change": absolute_change,
        "percentage_change": percentage_change,
    }


def build_metric_improvement(baseline, current):
    """
    Compare all available metrics between the
    baseline and current champion.
    """

    metric_names = [
        "validation_pr_auc",
        "validation_precision",
        "validation_recall",
        "validation_f1",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_roc_auc",
        "test_pr_auc",
    ]

    improvement = {}

    for metric in metric_names:

        improvement[metric] = safe_change(
            current.get(metric),
            baseline.get(metric),
        )

    return improvement


def build_recall_tradeoff(baseline, current):
    """
    Explicitly describe the precision/recall trade-off
    between the baseline and current model.

    No threshold or metric is invented here.
    """

    baseline_precision = baseline.get(
        "test_precision"
    )
    current_precision = current.get(
        "test_precision"
    )

    baseline_recall = baseline.get(
        "test_recall"
    )
    current_recall = current.get(
        "test_recall"
    )

    baseline_f1 = baseline.get(
        "test_f1"
    )
    current_f1 = current.get(
        "test_f1"
    )

    precision_change = safe_change(
        current_precision,
        baseline_precision,
    )

    recall_change = safe_change(
        current_recall,
        baseline_recall,
    )

    f1_change = safe_change(
        current_f1,
        baseline_f1,
    )

    if (
        baseline_precision is not None
        and current_precision is not None
        and baseline_recall is not None
        and current_recall is not None
    ):

        precision_improved = (
            current_precision
            > baseline_precision
        )

        recall_decreased = (
            current_recall
            < baseline_recall
        )

        if (
            precision_improved
            and recall_decreased
        ):
            interpretation = (
                "The current model improves precision "
                "while sacrificing recall. This means "
                "fewer legitimate transactions are likely "
                "to be flagged, but more fraudulent "
                "transactions may be missed."
            )

        elif (
            current_recall > baseline_recall
            and current_precision < baseline_precision
        ):
            interpretation = (
                "The current model improves recall "
                "while sacrificing precision. This means "
                "more fraudulent transactions are detected, "
                "but more legitimate transactions may be flagged."
            )

        elif (
            current_recall > baseline_recall
            and current_precision > baseline_precision
        ):
            interpretation = (
                "The current model improves both precision "
                "and recall relative to the baseline."
            )

        elif (
            current_recall < baseline_recall
            and current_precision < baseline_precision
        ):
            interpretation = (
                "The current model decreases both precision "
                "and recall relative to the baseline."
            )

        else:
            interpretation = (
                "The current model shows no clear "
                "precision/recall directional change."
            )

    else:
        interpretation = (
            "A complete precision/recall comparison "
            "is unavailable because historical metrics "
            "are missing from MLflow."
        )

    return {
        "baseline": {
            "precision": baseline_precision,
            "recall": baseline_recall,
            "f1": baseline_f1,
        },
        "current": {
            "precision": current_precision,
            "recall": current_recall,
            "f1": current_f1,
        },
        "precision_change": precision_change,
        "recall_change": recall_change,
        "f1_change": f1_change,
        "interpretation": interpretation,
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": REGISTERED_MODEL_NAME,
        "alias": CHAMPION_ALIAS,
    }


# ============================================================
# FRAUD PREDICTION
# ============================================================

@app.post("/predict")
def predict(
    request: TransactionRequest,
):

    try:

        return predict_transaction(
            request.transaction
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


# ============================================================
# CURRENT CHAMPION
# ============================================================

@app.get("/model/current")
def current_model():

    try:

        client = get_mlflow_client()

        champion = (
            client.get_model_version_by_alias(
                REGISTERED_MODEL_NAME,
                CHAMPION_ALIAS,
            )
        )

        metrics = get_version_metrics(
            champion
        )

        config = load_production_config()

        # Verify that the production model
        # can actually be loaded.
        model = load_production_model()

        return {
            "model": REGISTERED_MODEL_NAME,

            "version": int(
                champion.version
            ),

            "run_id": champion.run_id,

            "alias": CHAMPION_ALIAS,

            "status": champion.status,

            "model_type": config.get(
                "model_type"
            ),

            "xgb_weight": config.get(
                "xgb_weight"
            ),

            "nn_weight": config.get(
                "nn_weight"
            ),

            "production_threshold": config.get(
                "production_threshold"
            ),

            "selection_metric": config.get(
                "selection_metric"
            ),

            "threshold_selection_metric": config.get(
                "threshold_selection_metric"
            ),

            "metrics": metrics,

            "model_loaded": (
                model is not None
            ),
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


# ============================================================
# MODEL BENCHMARK
# ============================================================

@app.get("/model/benchmark")
def model_benchmark():

    try:

        client = get_mlflow_client()

        versions = get_model_versions()

        if not versions:

            raise HTTPException(
                status_code=404,
                detail=(
                    "No registered model "
                    "versions found."
                ),
            )

        # Current production champion
        champion = (
            client.get_model_version_by_alias(
                REGISTERED_MODEL_NAME,
                CHAMPION_ALIAS,
            )
        )

        # Oldest registered model becomes
        # the historical baseline.
        baseline_version = versions[0]

        baseline = get_version_metrics(
            baseline_version
        )

        current = get_version_metrics(
            champion
        )

        improvement = (
            build_metric_improvement(
                baseline,
                current,
            )
        )

        recall_tradeoff = (
            build_recall_tradeoff(
                baseline,
                current,
            )
        )

        config = load_production_config()

        return {

            "model": REGISTERED_MODEL_NAME,

            "baseline": baseline,

            "current": current,

            "improvement": improvement,

            "recall_tradeoff": recall_tradeoff,

            "production_configuration": {

                "production_threshold": config.get(
                    "production_threshold"
                ),

                "selection_metric": config.get(
                    "selection_metric"
                ),

                "threshold_selection_metric": config.get(
                    "threshold_selection_metric"
                ),

                "xgb_weight": config.get(
                    "xgb_weight"
                ),

                "nn_weight": config.get(
                    "nn_weight"
                ),
            },

            "interpretation": {
                "recall_importance": (
                    "Recall represents the proportion "
                    "of fraudulent transactions detected. "
                    "A lower recall means more fraud may "
                    "be missed."
                ),

                "precision_importance": (
                    "Precision represents the proportion "
                    "of flagged transactions that are "
                    "actually fraudulent. Higher precision "
                    "reduces unnecessary legitimate "
                    "transaction flags."
                ),

                "tradeoff": (
                    "The production model should be evaluated "
                    "using both recall and precision rather "
                    "than optimizing either metric in isolation."
                ),
            },
        }

    except HTTPException:
        raise

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


# ============================================================
# MODEL HISTORY
# ============================================================

@app.get("/model/history")
def model_history():

    try:

        client = get_mlflow_client()

        versions = get_model_versions()

        champion = (
            client.get_model_version_by_alias(
                REGISTERED_MODEL_NAME,
                CHAMPION_ALIAS,
            )
        )

        history = []

        for version in versions:

            metrics = get_version_metrics(
                version
            )

            history.append(
                metrics
            )

        return {

            "model": REGISTERED_MODEL_NAME,

            "current_version": int(
                champion.version
            ),

            "current_alias": CHAMPION_ALIAS,

            "versions": history,

        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )