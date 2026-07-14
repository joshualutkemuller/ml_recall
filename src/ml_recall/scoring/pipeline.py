"""Scoring pipeline for persisted per-horizon recall model bundles."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ml_recall.common.config import RiskBandThresholds
from ml_recall.modeling.artifacts import load_model_bundle
from ml_recall.modeling.training import RecallModelBundle, predict_recall_probabilities
from ml_recall.quality.checks import QualityCheckResult, validate_scoring_rows
from ml_recall.scoring.risk import assign_risk_band, generate_reason_codes


@dataclass(frozen=True)
class ScoredLoan:
    """Single scored loan emitted by the model scoring layer."""

    loan_id: str
    probabilities: dict[str, float]
    risk_band: str
    reason_codes: list[str]
    data_quality_status: str


@dataclass(frozen=True)
class ScoringResult:
    """Versioned scoring response plus validation diagnostics."""

    model_version: str
    feature_set_version: str
    predictions: tuple[ScoredLoan, ...]
    quality: QualityCheckResult


class RecallScorer:
    """Load, validate, and score rows with a fitted recall model bundle."""

    def __init__(self, bundle: RecallModelBundle, thresholds: RiskBandThresholds | None = None) -> None:
        self.bundle = bundle
        self.thresholds = thresholds or RiskBandThresholds()
        self.thresholds.validate()

    @classmethod
    def from_artifact(cls, path: str, thresholds: RiskBandThresholds | None = None) -> "RecallScorer":
        """Create a scorer from a validated on-disk model artifact."""

        return cls(load_model_bundle(path), thresholds=thresholds)

    @property
    def feature_columns(self) -> tuple[str, ...]:
        """Feature schema required by the loaded model bundle."""

        return self.bundle.models[0].feature_columns

    def validate_rows(self, rows: Iterable[Mapping[str, Any]]) -> QualityCheckResult:
        """Validate scoring keys and model feature coverage before scoring."""

        materialized = list(rows)
        result = validate_scoring_rows(materialized)
        errors = list(result.errors)
        for index, row in enumerate(materialized):
            missing = [column for column in self.feature_columns if column not in row]
            if missing:
                errors.append(f"row {index} missing model feature columns: {missing}")
        return QualityCheckResult("PASS" if not errors else "FAIL", tuple(errors), result.warnings)

    def score(
        self,
        rows: Iterable[Mapping[str, Any]],
        *,
        horizons: Sequence[int] | None = None,
    ) -> ScoringResult:
        """Validate and score rows, optionally filtering probabilities to requested horizons."""

        materialized = list(rows)
        quality = self.validate_rows(materialized)
        if not quality.passed:
            return ScoringResult(
                self.bundle.model_version,
                self.bundle.feature_set_version,
                tuple(),
                quality,
            )
        requested = set(horizons or self.bundle.horizons)
        unsupported = requested.difference(self.bundle.horizons)
        if unsupported:
            raise ValueError(f"requested horizons are not in the model bundle: {sorted(unsupported)}")

        predictions: list[ScoredLoan] = []
        for scored in predict_recall_probabilities(self.bundle, materialized):
            probabilities = {
                f"{horizon}d": scored[f"probability_recall_{horizon}d"]
                for horizon in self.bundle.horizons
                if horizon in requested
            }
            max_probability = max(probabilities.values())
            contributions = {column: float(scored.get(column) or 0.0) for column in self.feature_columns}
            predictions.append(
                ScoredLoan(
                    loan_id=str(scored["loan_id"]),
                    probabilities=probabilities,
                    risk_band=assign_risk_band(max_probability, self.thresholds),
                    reason_codes=generate_reason_codes(contributions),
                    data_quality_status=quality.status,
                )
            )
        return ScoringResult(
            self.bundle.model_version,
            self.bundle.feature_set_version,
            tuple(predictions),
            quality,
        )
