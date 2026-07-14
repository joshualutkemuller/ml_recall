"""Point-in-time loan feature engineering for recall prediction.

The helpers in this module are intentionally dataframe-free so they can be reused in batch jobs,
unit tests, and lightweight API paths before the production feature store is integrated.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Any


def _parse_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    return float(value)


def _percentile_rank(value: float, history: list[float]) -> float:
    if not history:
        return 0.0
    below_or_equal = sum(1 for item in history if item <= value)
    return round(below_or_equal / len(history), 6)


def build_loan_features(
    observations: Iterable[Mapping[str, Any]],
    *,
    recall_history: Iterable[Mapping[str, Any]] = (),
    market_history: Iterable[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Build point-in-time features for loan observation rows.

    Required observation fields are ``loan_id``, ``security_id``, ``lender_id``,
    ``as_of_timestamp``, ``loan_open_timestamp``, ``quantity``, ``available_inventory``, and
    ``borrow_fee_bps``. Recall and market history rows after ``as_of_timestamp`` are ignored to
    prevent look-ahead leakage.
    """

    required = {
        "loan_id",
        "security_id",
        "lender_id",
        "as_of_timestamp",
        "loan_open_timestamp",
        "quantity",
        "available_inventory",
        "borrow_fee_bps",
    }
    recall_rows = list(recall_history)
    market_rows = list(market_history)
    feature_rows: list[dict[str, Any]] = []

    for observation in observations:
        missing = required.difference(observation)
        if missing:
            raise ValueError(f"observation row missing required fields: {sorted(missing)}")

        as_of = _parse_timestamp(observation["as_of_timestamp"])
        loan_open = _parse_timestamp(observation["loan_open_timestamp"])
        loan_age_days = max(0, (as_of.date() - loan_open.date()).days)
        quantity = _safe_float(observation["quantity"])
        available_inventory = _safe_float(observation["available_inventory"])
        borrow_fee_bps = _safe_float(observation["borrow_fee_bps"])
        utilization = quantity / (quantity + available_inventory) if quantity + available_inventory else 0.0

        lender_id = str(observation["lender_id"])
        lender_recalls = [
            row
            for row in recall_rows
            if str(row.get("lender_id")) == lender_id
            and row.get("event_type") == "LENDER_RECALL"
            and _parse_timestamp(row["recall_timestamp"]) <= as_of
        ]
        lender_recent_recalls = [
            row
            for row in lender_recalls
            if (_parse_timestamp(row["recall_timestamp"]) - as_of).days >= -30
        ]

        security_id = str(observation["security_id"])
        historical_fees = [
            _safe_float(row.get("borrow_fee_bps"))
            for row in market_rows
            if str(row.get("security_id")) == security_id and _parse_timestamp(row["as_of_timestamp"]) <= as_of
        ]

        feature_rows.append(
            {
                **dict(observation),
                "loan_age_days": loan_age_days,
                "utilization_ratio": round(utilization, 6),
                "lender_recent_recall_count_30d": len(lender_recent_recalls),
                "lender_recent_recall_rate": round(len(lender_recent_recalls) / max(len(lender_recalls), 1), 6),
                "borrow_fee_percentile": _percentile_rank(borrow_fee_bps, historical_fees),
            }
        )
    return feature_rows
