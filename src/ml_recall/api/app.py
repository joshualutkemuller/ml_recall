"""FastAPI app exposing the MVP recall prediction contract."""

from __future__ import annotations

import os

from fastapi import FastAPI

from ml_recall.api.schemas import LoanPrediction, PredictionRequest, PredictionResponse
from ml_recall.common.config import RecallConfig
from ml_recall.scoring.pipeline import RecallScorer
from ml_recall.scoring.risk import assign_risk_band, generate_reason_codes

app = FastAPI(title="Loan Recall Prediction API", version="0.1.0")
CONFIG = RecallConfig()


def _load_configured_scorer() -> RecallScorer | None:
    artifact_path = os.getenv("ML_RECALL_MODEL_ARTIFACT")
    if not artifact_path:
        return None
    try:
        return RecallScorer.from_artifact(artifact_path, thresholds=CONFIG.thresholds)
    except (OSError, ValueError):
        return None


SCORER = _load_configured_scorer()


def _deterministic_probability(loan_id: str, horizon: int) -> float:
    """Placeholder deterministic scorer until a trained model artifact is registered."""

    seed = sum(ord(character) for character in loan_id) + horizon * 17
    return round(min(0.95, 0.02 + (seed % 37) / 100), 4)


@app.post("/v1/loan-recall/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Score requested loans with the MVP API response shape."""

    if SCORER is not None and request.feature_rows is not None:
        scoring_result = SCORER.score(request.feature_rows, horizons=request.horizons)
        if scoring_result.quality.passed:
            return PredictionResponse(
                model_version=scoring_result.model_version,
                predictions=[
                    LoanPrediction(
                        loan_id=prediction.loan_id,
                        as_of_timestamp=request.as_of_timestamp,
                        probabilities=prediction.probabilities,
                        risk_band=prediction.risk_band,
                        reason_codes=prediction.reason_codes,
                        data_quality_status=prediction.data_quality_status,
                    )
                    for prediction in scoring_result.predictions
                ],
            )

    predictions: list[LoanPrediction] = []
    for loan_id in request.loan_ids:
        probabilities = {
            f"{horizon}d": _deterministic_probability(loan_id, horizon)
            for horizon in request.horizons
        }
        max_probability = max(probabilities.values())
        predictions.append(
            LoanPrediction(
                loan_id=loan_id,
                as_of_timestamp=request.as_of_timestamp,
                probabilities=probabilities,
                risk_band=assign_risk_band(max_probability, CONFIG.thresholds),
                reason_codes=generate_reason_codes(
                    {
                        "lender_recent_recall_rate": max_probability,
                        "utilization_percentile": max_probability / 2,
                        "borrow_fee_percentile": max_probability / 3,
                    }
                ),
            )
        )
    return PredictionResponse(model_version=CONFIG.model_version, predictions=predictions)
