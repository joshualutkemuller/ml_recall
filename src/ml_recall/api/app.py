"""FastAPI app exposing the MVP recall prediction contract."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException

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


def _normalized_timestamp(value: Any) -> str:
    """Normalize request and row timestamps before alignment checks."""

    if isinstance(value, datetime):
        timestamp = value
    else:
        timestamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc).isoformat()


def _validate_request_feature_alignment(request: PredictionRequest) -> None:
    """Make the relationship between requested loans and supplied feature rows explicit."""

    if request.feature_rows is None:
        return

    requested_ids = list(request.loan_ids)
    requested_id_set = set(requested_ids)
    row_ids = [str(row.get("loan_id")) for row in request.feature_rows if row.get("loan_id") not in (None, "")]
    row_id_set = set(row_ids)
    missing = [loan_id for loan_id in requested_ids if loan_id not in row_id_set]
    extra = sorted(row_id_set.difference(requested_id_set))
    errors: list[str] = []
    if missing:
        errors.append(f"feature_rows missing requested loan_ids: {missing}")
    if extra:
        errors.append(f"feature_rows include loan_ids not requested: {extra}")

    request_as_of = _normalized_timestamp(request.as_of_timestamp)
    mismatched_rows = []
    for index, row in enumerate(request.feature_rows):
        if row.get("as_of_timestamp") in (None, ""):
            continue
        try:
            row_as_of = _normalized_timestamp(row["as_of_timestamp"])
        except ValueError:
            mismatched_rows.append(index)
            continue
        if row_as_of != request_as_of:
            mismatched_rows.append(index)
    if mismatched_rows:
        errors.append(
            "feature_rows as_of_timestamp must match request as_of_timestamp; "
            f"mismatched rows: {mismatched_rows}"
        )
    if errors:
        raise HTTPException(status_code=422, detail={"data_quality_status": "FAIL", "errors": errors})


def _ordered_scored_predictions(request: PredictionRequest, scorer: RecallScorer) -> PredictionResponse:
    """Score explicitly supplied feature rows and return predictions in request loan order."""

    assert request.feature_rows is not None
    scoring_result = scorer.score(request.feature_rows, horizons=request.horizons)
    if not scoring_result.quality.passed:
        raise HTTPException(
            status_code=422,
            detail={
                "data_quality_status": "FAIL",
                "errors": list(scoring_result.quality.errors),
                "warnings": list(scoring_result.quality.warnings),
            },
        )

    predictions_by_loan = {prediction.loan_id: prediction for prediction in scoring_result.predictions}
    return PredictionResponse(
        model_version=scoring_result.model_version,
        predictions=[
            LoanPrediction(
                loan_id=loan_id,
                as_of_timestamp=request.as_of_timestamp,
                probabilities=predictions_by_loan[loan_id].probabilities,
                risk_band=predictions_by_loan[loan_id].risk_band,
                reason_codes=predictions_by_loan[loan_id].reason_codes,
                data_quality_status=predictions_by_loan[loan_id].data_quality_status,
            )
            for loan_id in request.loan_ids
        ],
    )


@app.post("/v1/loan-recall/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Score requested loans with the MVP API response shape."""

    _validate_request_feature_alignment(request)
    if SCORER is not None and request.feature_rows is not None:
        try:
            return _ordered_scored_predictions(request, SCORER)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

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
