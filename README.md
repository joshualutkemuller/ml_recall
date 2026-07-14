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
