"""Train transparent per-horizon recall classifiers.

The first production phase in the handoff calls for interpretable binary classifiers for each
prediction horizon. This module provides a lightweight implementation that can be used in notebooks,
batch jobs, and CI before feature-store, model-registry, or heavier training infrastructure is added.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import exp, log
from statistics import fmean
from typing import Any


@dataclass(frozen=True)
class HorizonModel:
    """A fitted transparent scorecard for one recall horizon."""

    horizon: int
    label_column: str
    feature_columns: tuple[str, ...]
    intercept: float
    coefficients: dict[str, float]
    medians: dict[str, float]


@dataclass(frozen=True)
class RecallModelBundle:
    """Collection of fitted per-horizon scorecards sharing one feature schema."""

    models: tuple[HorizonModel, ...]
    model_version: str
    feature_set_version: str

    @property
    def horizons(self) -> tuple[int, ...]:
        """Return fitted horizons in ascending order."""

        return tuple(model.horizon for model in self.models)


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2


def _logit(probability: float) -> float:
    clipped = min(max(probability, 0.001), 0.999)
    return log(clipped / (1 - clipped))


def _fit_scorecard(
    rows: Sequence[Mapping[str, Any]],
    feature_columns: Sequence[str],
    label_column: str,
    horizon: int,
) -> HorizonModel:
    labels = [int(row[label_column]) for row in rows]
    positive_rate = (sum(labels) + 1) / (len(labels) + 2)
    medians = {
        column: _median([value for row in rows if (value := _safe_float(row.get(column))) is not None])
        for column in feature_columns
    }
    coefficients: dict[str, float] = {}
    for column in feature_columns:
        positives = []
        negatives = []
        for row in rows:
            value = _safe_float(row.get(column))
            imputed_value = medians[column] if value is None else value
            if int(row[label_column]) == 1:
                positives.append(imputed_value)
            else:
                negatives.append(imputed_value)
        if not positives or not negatives:
            coefficients[column] = 0.0
            continue
        all_values = positives + negatives
        value_range = max(all_values) - min(all_values) or 1.0
        coefficients[column] = (fmean(positives) - fmean(negatives)) / value_range
    return HorizonModel(
        horizon=horizon,
        label_column=label_column,
        feature_columns=tuple(feature_columns),
        intercept=_logit(positive_rate),
        coefficients=coefficients,
        medians=medians,
    )


def train_horizon_models(
    training_rows: Iterable[Mapping[str, Any]],
    *,
    feature_columns: Sequence[str],
    horizons: Sequence[int] = (1, 3, 5, 10),
    model_version: str = "recall_model_0.1.0",
    feature_set_version: str = "feature_set_0.1.0",
) -> RecallModelBundle:
    """Fit one binary scorecard per requested horizon.

    Rows must include all requested numeric feature columns and label columns named
    ``label_recall_{horizon}d``. The scorecard is intentionally transparent: each coefficient is
    the normalized mean feature difference between positive and negative examples.
    """

    rows = list(training_rows)
    if not rows:
        raise ValueError("training_rows must contain at least one row")
    if not feature_columns:
        raise ValueError("feature_columns must contain at least one feature")
    missing_features = [column for column in feature_columns if any(column not in row for row in rows)]
    if missing_features:
        raise ValueError(f"training rows missing feature columns: {sorted(set(missing_features))}")

    fitted: list[HorizonModel] = []
    for horizon in horizons:
        if horizon <= 0:
            raise ValueError("horizons must be positive business-day counts")
        label_column = f"label_recall_{horizon}d"
        if any(label_column not in row for row in rows):
            raise ValueError(f"training rows missing label column: {label_column}")
        fitted.append(_fit_scorecard(rows, feature_columns, label_column, horizon))
    return RecallModelBundle(tuple(fitted), model_version, feature_set_version)


def _predict_one(model: HorizonModel, row: Mapping[str, Any]) -> float:
    score = model.intercept
    for column in model.feature_columns:
        value = _safe_float(row.get(column))
        centered = (model.medians[column] if value is None else value) - model.medians[column]
        score += model.coefficients[column] * centered
    return 1 / (1 + exp(-score))


def predict_recall_probabilities(
    bundle: RecallModelBundle,
    scoring_rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Score rows with a fitted recall model bundle."""

    predictions: list[dict[str, Any]] = []
    for row in scoring_rows:
        scored = dict(row)
        for model in bundle.models:
            scored[f"probability_recall_{model.horizon}d"] = round(_predict_one(model, row), 6)
        predictions.append(scored)
    return predictions
