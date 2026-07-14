"""Configuration loading for the loan recall platform."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RiskBandThresholds:
    """Probability cutoffs for operational risk bands."""

    moderate: float = 0.10
    high: float = 0.25
    critical: float = 0.50

    def validate(self) -> None:
        if not 0 <= self.moderate < self.high < self.critical <= 1:
            raise ValueError("risk band thresholds must satisfy 0 <= moderate < high < critical <= 1")


@dataclass(frozen=True)
class RecallConfig:
    """Runtime configuration shared by scoring, API, and tests."""

    horizons: tuple[int, ...] = (1, 3, 5, 10)
    model_version: str = "recall_model_0.1.0"
    feature_set_version: str = "feature_set_0.1.0"
    thresholds: RiskBandThresholds = RiskBandThresholds()

    def validate(self) -> None:
        if not self.horizons:
            raise ValueError("at least one prediction horizon is required")
        if any(horizon <= 0 for horizon in self.horizons):
            raise ValueError("prediction horizons must be positive business-day counts")
        self.thresholds.validate()


def _parse_scalar(value: str) -> Any:
    stripped = value.strip().strip('"').strip("'")
    if stripped.startswith("[") and stripped.endswith("]"):
        return [int(part.strip()) for part in stripped[1:-1].split(",") if part.strip()]
    try:
        return int(stripped)
    except ValueError:
        try:
            return float(stripped)
        except ValueError:
            return stripped


def _parse_simple_yaml(path: Path) -> dict[str, Any]:
    """Parse the small YAML subset used by repository config files without runtime deps."""

    parsed: dict[str, Any] = {}
    current_section: str | None = None
    for raw_line in path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line:
            continue
        if not line.startswith(" ") and line.endswith(":"):
            current_section = line[:-1]
            parsed[current_section] = {}
            continue
        key, value = line.strip().split(":", 1)
        target = parsed[current_section] if current_section else parsed
        target[key] = _parse_scalar(value)
    return parsed


def load_config(path: str | Path) -> RecallConfig:
    """Load a repository YAML configuration file into a typed config object."""

    raw = _parse_simple_yaml(Path(path))
    thresholds = RiskBandThresholds(**raw.get("thresholds", {}))
    config = RecallConfig(
        horizons=tuple(raw.get("horizons", (1, 3, 5, 10))),
        model_version=raw.get("model_version", "recall_model_0.1.0"),
        feature_set_version=raw.get("feature_set_version", "feature_set_0.1.0"),
        thresholds=thresholds,
    )
    config.validate()
    return config
