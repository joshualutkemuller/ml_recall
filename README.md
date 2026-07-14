# Loan Recall Prediction Platform

This repository contains the initial MVP scaffold for a securities lending loan recall prediction platform. The build follows the handoff specification in `loan_recall_prediction_sdlc_handoff.md` and starts with the core contracts needed for Stage 3 implementation:

- typed configuration for horizons, model versions, feature-set versions, and risk-band thresholds;
- point-in-time label construction for lender-initiated recall events;
- deterministic placeholder scoring behind the API contract;
- risk band and reason-code utilities;
- point-in-time feature engineering and scoring data quality checks; and
- unit tests for label logic, feature construction, data quality, and scoring controls.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
```

## Modeling MVP

The repository now includes Stage 3 model-development utilities:

- `ml_recall.modeling.training.train_horizon_models` trains one binary classifier per recall horizon using numeric point-in-time features and `label_recall_{horizon}d` labels.
- `ml_recall.modeling.training.predict_recall_probabilities` scores rows with the fitted per-horizon bundle.
- `ml_recall.modeling.evaluation` exposes PR-AUC, Brier score, and precision/recall-at-K metrics aligned with the handoff success measures.
- `ml_recall.modeling.artifacts` persists fitted bundles as validated JSON artifacts with schema, version, horizon, and feature metadata for reproducible promotion.
- `ml_recall.scoring.pipeline.RecallScorer` loads a persisted bundle, validates scoring rows, filters requested horizons, assigns risk bands, and emits reason codes.

Set `ML_RECALL_MODEL_ARTIFACT=/path/to/bundle.json` to let the API use a persisted scorer when callers provide `feature_rows`; if the artifact is absent, invalid, or feature rows are omitted or fail validation, the API safely falls back to deterministic placeholder scoring.

## API contract

The MVP API exposes the handoff endpoint:

```text
POST /v1/loan-recall/predict
```

Run locally with:

```bash
uvicorn ml_recall.api.app:app --reload
```

The current scorer is deterministic and intentionally simple. It preserves the response contract while training, calibration, model registry, and feature-store integrations are built out.
