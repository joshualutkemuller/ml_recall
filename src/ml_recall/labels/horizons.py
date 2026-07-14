"""Point-in-time label construction for recall horizons."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta, timezone
from typing import Any


def _parse_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _add_business_days(start: datetime, business_days: int) -> datetime:
    current = start
    remaining = business_days
    while remaining > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current


def build_recall_labels(
    observations: Iterable[Mapping[str, Any]],
    recall_events: Iterable[Mapping[str, Any]],
    horizons: tuple[int, ...] = (1, 3, 5, 10),
) -> list[dict[str, Any]]:
    """Attach binary lender-initiated recall labels to loan observation rows.

    Required observation fields are ``loan_id`` and ``as_of_timestamp``. Required event fields are
    ``loan_id``, ``recall_timestamp``, and ``event_type``. Only ``LENDER_RECALL`` events are
    positive labels, preserving the event taxonomy from the handoff.
    """

    event_times_by_loan: dict[str, list[datetime]] = {}
    for event in recall_events:
        if not {"loan_id", "recall_timestamp", "event_type"}.issubset(event):
            raise ValueError("recall event rows require loan_id, recall_timestamp, and event_type")
        if event["event_type"] != "LENDER_RECALL":
            continue
        event_times_by_loan.setdefault(str(event["loan_id"]), []).append(
            _parse_timestamp(event["recall_timestamp"])
        )

    labeled_rows: list[dict[str, Any]] = []
    for observation in observations:
        if not {"loan_id", "as_of_timestamp"}.issubset(observation):
            raise ValueError("observation rows require loan_id and as_of_timestamp")
        labeled = dict(observation)
        as_of = _parse_timestamp(observation["as_of_timestamp"])
        for horizon in horizons:
            if horizon <= 0:
                raise ValueError("horizons must be positive business-day counts")
            horizon_end = _add_business_days(as_of, horizon)
            labeled[f"label_recall_{horizon}d"] = int(
                any(
                    as_of < event_time <= horizon_end
                    for event_time in event_times_by_loan.get(str(observation["loan_id"]), [])
                )
            )
        labeled_rows.append(labeled)
    return labeled_rows
