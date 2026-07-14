"""Pydantic schemas for the recall prediction API."""

from __future__ import annotations

from datetime import datetime

from typing import Any

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request for loan recall prediction scoring."""

    loan_ids: list[str] = Field(min_length=1)
    as_of_timestamp: datetime
    horizons: list[int] = Field(default_factory=lambda: [1, 3, 5, 10])
    feature_rows: list[dict[str, Any]] | None = None


class LoanPrediction(BaseModel):
    """Single loan prediction response."""

    loan_id: str
    as_of_timestamp: datetime
    probabilities: dict[str, float]
    risk_band: str
    reason_codes: list[str]
    data_quality_status: str = "PASS"


class PredictionResponse(BaseModel):
    """Versioned API response containing one prediction per requested loan."""

    model_version: str
    predictions: list[LoanPrediction]
