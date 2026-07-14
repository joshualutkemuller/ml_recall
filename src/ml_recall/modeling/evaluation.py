"""Evaluation metrics for recall model validation and monitoring."""

from __future__ import annotations

from collections.abc import Sequence


def _validate(y_true: Sequence[int], y_score: Sequence[float]) -> None:
    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have the same length")
    if not y_true:
        raise ValueError("at least one observation is required")


def precision_recall_at_k(y_true: Sequence[int], y_score: Sequence[float], k: int) -> dict[str, float]:
    """Compute top-k precision and recall for operational alert queues."""

    if k <= 0:
        raise ValueError("k must be positive")
    _validate(y_true, y_score)
    cutoff = min(k, len(y_score))
    ranked = sorted(enumerate(y_score), key=lambda item: item[1], reverse=True)
    top_indices = {index for index, _ in ranked[:cutoff]}
    true_positives = sum(1 for index in top_indices if y_true[index] == 1)
    total_positives = sum(1 for label in y_true if label == 1)
    return {
        "precision_at_k": true_positives / cutoff,
        "recall_at_k": true_positives / total_positives if total_positives else 0.0,
    }


def _average_precision(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    total_positives = sum(1 for label in y_true if label == 1)
    if total_positives == 0:
        return 0.0
    running_true_positives = 0
    precision_sum = 0.0
    for rank, (index, _) in enumerate(
        sorted(enumerate(y_score), key=lambda item: item[1], reverse=True), start=1
    ):
        if y_true[index] == 1:
            running_true_positives += 1
            precision_sum += running_true_positives / rank
    return precision_sum / total_positives


def classification_validation_metrics(y_true: Sequence[int], y_score: Sequence[float]) -> dict[str, float]:
    """Compute core metrics named in the model-risk handoff."""

    _validate(y_true, y_score)
    brier = sum((float(score) - int(label)) ** 2 for label, score in zip(y_true, y_score)) / len(y_true)
    return {
        "pr_auc": _average_precision(y_true, y_score),
        "brier_score": brier,
    }
