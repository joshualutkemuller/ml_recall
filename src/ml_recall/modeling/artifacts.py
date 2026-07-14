"""Persist and validate reproducible recall model artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ml_recall.modeling.training import HorizonModel, RecallModelBundle, train_horizon_models

ARTIFACT_SCHEMA_VERSION = "recall-model-bundle/v1"
ARTIFACT_DIGEST_FIELD = "artifact_sha256"


@dataclass(frozen=True)
class ReproducibilityMetadata:
    """Versioned inputs needed to reproduce a trained model artifact."""

    dataset_version: str
    label_version: str
    training_window_start: str
    training_window_end: str
    code_commit: str
    random_seed: int = 0


def _canonical_json(payload: dict[str, Any]) -> str:
    """Return the canonical JSON representation used for artifact hashing."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def artifact_digest(artifact: dict[str, Any]) -> str:
    """Hash artifact semantics while excluding the self-referential digest field."""

    digestible = {key: value for key, value in artifact.items() if key != ARTIFACT_DIGEST_FIELD}
    return hashlib.sha256(_canonical_json(digestible).encode("utf-8")).hexdigest()


def bundle_to_artifact(
    bundle: RecallModelBundle,
    *,
    reproducibility: ReproducibilityMetadata | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert a fitted model bundle into a deterministic JSON-serializable artifact.

    The artifact intentionally excludes wall-clock creation time. Reproducibility is expressed by
    explicit versioned inputs plus a SHA-256 digest over the canonical artifact payload.
    """

    if reproducibility is None:
        raise ValueError("reproducibility metadata is required for model artifacts")
    metadata = (
        asdict(reproducibility)
        if isinstance(reproducibility, ReproducibilityMetadata)
        else reproducibility
    )
    artifact: dict[str, Any] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "model_version": bundle.model_version,
        "feature_set_version": bundle.feature_set_version,
        "horizons": list(bundle.horizons),
        "feature_columns": list(bundle.models[0].feature_columns) if bundle.models else [],
        "models": [asdict(model) for model in bundle.models],
        "reproducibility": metadata,
    }
    artifact[ARTIFACT_DIGEST_FIELD] = artifact_digest(artifact)
    return artifact


def validate_artifact(artifact: dict[str, Any]) -> None:
    """Validate artifact structure before it is promoted or loaded for scoring."""

    required = {
        "schema_version",
        "model_version",
        "feature_set_version",
        "horizons",
        "feature_columns",
        "models",
        "reproducibility",
        ARTIFACT_DIGEST_FIELD,
    }
    missing = required.difference(artifact)
    if missing:
        raise ValueError(f"model artifact missing required fields: {sorted(missing)}")
    if artifact["schema_version"] != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(f"unsupported artifact schema_version: {artifact['schema_version']}")
    expected_digest = artifact_digest(artifact)
    if artifact[ARTIFACT_DIGEST_FIELD] != expected_digest:
        raise ValueError("model artifact digest does not match artifact contents")
    if not artifact["models"]:
        raise ValueError("model artifact must contain at least one horizon model")
    horizons = tuple(int(horizon) for horizon in artifact["horizons"])
    model_horizons = tuple(int(model["horizon"]) for model in artifact["models"])
    if horizons != model_horizons:
        raise ValueError("artifact horizons must match model horizons in order")
    feature_columns = tuple(artifact["feature_columns"])
    if not feature_columns:
        raise ValueError("model artifact must declare feature_columns")
    reproducibility = artifact["reproducibility"]
    if not isinstance(reproducibility, dict):
        raise ValueError("artifact reproducibility metadata must be an object")
    required_reproducibility_fields = (
        "dataset_version",
        "label_version",
        "training_window_start",
        "training_window_end",
        "code_commit",
        "random_seed",
    )
    missing_reproducibility = [
        field for field in required_reproducibility_fields if field not in reproducibility
    ]
    if missing_reproducibility:
        raise ValueError(
            f"artifact reproducibility missing required fields: {missing_reproducibility}"
        )
    for field in required_reproducibility_fields:
        if reproducibility[field] in (None, ""):
            raise ValueError(f"artifact reproducibility field must be populated: {field}")
    for model in artifact["models"]:
        model_features = tuple(model.get("feature_columns", ()))
        if model_features != feature_columns:
            raise ValueError("all horizon models must share artifact feature_columns")
        for field in ("label_column", "intercept", "coefficients", "medians"):
            if field not in model:
                raise ValueError(f"horizon model missing required field: {field}")


def save_model_bundle(
    bundle: RecallModelBundle,
    path: str | Path,
    *,
    reproducibility: ReproducibilityMetadata | dict[str, Any] | None = None,
) -> Path:
    """Write a fitted bundle as sorted, indented JSON for reproducible diffs."""

    artifact = bundle_to_artifact(bundle, reproducibility=reproducibility)
    validate_artifact(artifact)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return destination


def load_model_bundle(path: str | Path) -> RecallModelBundle:
    """Load and validate a persisted recall model bundle artifact."""

    artifact = json.loads(Path(path).read_text())
    validate_artifact(artifact)
    models = tuple(
        HorizonModel(
            horizon=int(model["horizon"]),
            label_column=str(model["label_column"]),
            feature_columns=tuple(model["feature_columns"]),
            intercept=float(model["intercept"]),
            coefficients={key: float(value) for key, value in model["coefficients"].items()},
            medians={key: float(value) for key, value in model["medians"].items()},
        )
        for model in artifact["models"]
    )
    return RecallModelBundle(
        models=models,
        model_version=str(artifact["model_version"]),
        feature_set_version=str(artifact["feature_set_version"]),
    )


def train_and_save_model_bundle(
    training_rows: list[dict[str, Any]],
    *,
    feature_columns: list[str] | tuple[str, ...],
    output_path: str | Path,
    horizons: tuple[int, ...] = (1, 3, 5, 10),
    model_version: str = "recall_model_0.1.0",
    feature_set_version: str = "feature_set_0.1.0",
    reproducibility: ReproducibilityMetadata | dict[str, Any] | None = None,
) -> RecallModelBundle:
    """Train the per-horizon bundle and persist it as one reproducible operation."""

    bundle = train_horizon_models(
        training_rows,
        feature_columns=feature_columns,
        horizons=horizons,
        model_version=model_version,
        feature_set_version=feature_set_version,
    )
    save_model_bundle(bundle, output_path, reproducibility=reproducibility)
    return bundle
