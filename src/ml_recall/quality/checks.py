"""Reusable data quality checks for scoring and training inputs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class QualityCheckResult:
    """Result from validating a collection of loan input rows."""

    status: str
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return self.status == "PASS"


_REQUIRED_FIELDS = ("loan_id", "security_id", "lender_id", "as_of_timestamp")


def validate_scoring_rows(rows: Iterable[Mapping[str, Any]]) -> QualityCheckResult:
    """Validate minimum fields and duplicate keys before scoring loans."""

    errors: list[str] = []
    warnings: list[str] = []
    seen: set[tuple[str, str]] = set()
    count = 0
    for index, row in enumerate(rows):
        count += 1
        missing = [field for field in _REQUIRED_FIELDS if row.get(field) in (None, "")]
        if missing:
            errors.append(f"row {index} missing required fields: {missing}")
        key = (str(row.get("loan_id")), str(row.get("as_of_timestamp")))
        if key in seen:
            errors.append(f"row {index} duplicates loan/as_of key: {key}")
        seen.add(key)
        if row.get("available_inventory") is not None and float(row["available_inventory"]) < 0:
            warnings.append(f"row {index} has negative available_inventory")
    if count == 0:
        errors.append("no rows supplied")
    return QualityCheckResult("PASS" if not errors else "FAIL", tuple(errors), tuple(warnings))
