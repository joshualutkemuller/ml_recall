"""Command-line batch scoring for persisted recall model artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from ml_recall.scoring.pipeline import RecallScorer


def _read_rows(path: Path) -> list[dict[str, Any]]:
    """Read scoring rows from JSON, CSV, or Parquet based on file suffix."""

    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text())
        if isinstance(payload, dict):
            rows = payload.get("rows")
            if rows is None:
                rows = payload.get("feature_rows")
        else:
            rows = payload
        if not isinstance(rows, list):
            raise ValueError("JSON input must be a list of rows or an object with rows/feature_rows")
        return [dict(row) for row in rows]
    if suffix == ".csv":
        return pd.read_csv(path).to_dict(orient="records")
    if suffix == ".parquet":
        return pd.read_parquet(path).to_dict(orient="records")
    raise ValueError(f"unsupported input format: {suffix}")


def _write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    """Write long-form prediction rows to JSON, CSV, or Parquet based on file suffix."""

    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".json":
        path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
        return
    frame = pd.DataFrame(rows)
    if suffix == ".csv":
        frame.to_csv(path, index=False)
        return
    if suffix == ".parquet":
        frame.to_parquet(path, index=False)
        return
    raise ValueError(f"unsupported output format: {suffix}")


def _long_form_rows(scoring_result, input_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten scorer output to one row per loan, timestamp, and prediction horizon."""

    as_of_by_loan = {str(row["loan_id"]): row.get("as_of_timestamp") for row in input_rows}
    output: list[dict[str, Any]] = []
    for prediction in scoring_result.predictions:
        for horizon_key, probability in prediction.probabilities.items():
            output.append(
                {
                    "loan_id": prediction.loan_id,
                    "as_of_timestamp": as_of_by_loan[prediction.loan_id],
                    "horizon_days": int(horizon_key.removesuffix("d")),
                    "probability": probability,
                    "risk_band": prediction.risk_band,
                    "reason_codes": ";".join(prediction.reason_codes),
                    "data_quality_status": prediction.data_quality_status,
                    "model_version": scoring_result.model_version,
                    "feature_set_version": scoring_result.feature_set_version,
                }
            )
    return output


def _parse_horizons(value: str | None) -> tuple[int, ...] | None:
    if not value:
        return None
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def build_parser() -> argparse.ArgumentParser:
    """Build the batch scoring argument parser."""

    parser = argparse.ArgumentParser(description="Score loan recall risk from a model artifact.")
    parser.add_argument("--artifact", required=True, help="Path to JSON model artifact")
    parser.add_argument("--input", required=True, help="Path to JSON, CSV, or Parquet scoring rows")
    parser.add_argument("--output", required=True, help="Path for JSON, CSV, or Parquet predictions")
    parser.add_argument("--horizons", help="Comma-separated horizons to score, e.g. 1,3,5")
    parser.add_argument("--diagnostics", help="Optional JSON path for validation diagnostics")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run batch scoring and return a process exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)
    diagnostics: dict[str, Any] = {"data_quality_status": "UNKNOWN", "errors": [], "warnings": []}
    try:
        rows = _read_rows(Path(args.input))
        scorer = RecallScorer.from_artifact(args.artifact)
        result = scorer.score(rows, horizons=_parse_horizons(args.horizons))
        diagnostics = {
            "data_quality_status": result.quality.status,
            "errors": list(result.quality.errors),
            "warnings": list(result.quality.warnings),
            "model_version": result.model_version,
            "feature_set_version": result.feature_set_version,
            "prediction_count": sum(len(prediction.probabilities) for prediction in result.predictions),
        }
        if args.diagnostics:
            Path(args.diagnostics).parent.mkdir(parents=True, exist_ok=True)
            Path(args.diagnostics).write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n")
        if not result.quality.passed:
            print(json.dumps(diagnostics, sort_keys=True), file=sys.stderr)
            return 2
        _write_rows(_long_form_rows(result, rows), Path(args.output))
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI boundary converts failures to exit codes.
        diagnostics["errors"] = [str(exc)]
        if args.diagnostics:
            Path(args.diagnostics).parent.mkdir(parents=True, exist_ok=True)
            Path(args.diagnostics).write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n")
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
