from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.inference.predict import (
    FEATURE_COLUMNS,
    predict_transaction,
)


app = FastAPI(
    title="AI Risk Manager",
    description="Production fraud-risk prediction API",
    version="1.0.0",
)


class TransactionRequest(BaseModel):
    transaction: dict


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "AI-Risk-Manager-XGBoost",
        "alias": "champion",
    }


@app.post("/predict")
def predict(request: TransactionRequest):

    try:

        result = predict_transaction(
            request.transaction
        )

        return result

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