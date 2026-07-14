# Model Artifact and Scoring Pipeline Next Steps

This document captures the recommended next steps after the initial artifact-backed model scoring layer was introduced. The recommendations focus on making the implementation more reproducible, observable, testable, and production-ready while preserving the safe fallback posture of the current API.

## 1. Tighten artifact reproducibility semantics

The current artifact writer includes run-time metadata such as `created_at_utc`. That is useful for provenance, but it means two identical training runs will not produce byte-identical JSON artifacts.

Recommended improvements:

- Make creation timestamps injectable from the training run metadata.
- Separate immutable model content from mutable run metadata.
- Add a deterministic content hash over the model payload.
- Store the hash alongside the artifact so promotion and loading can verify integrity.
- Define whether “reproducible artifact” means byte-identical output, behavior-identical output, or both.

## 2. Promote artifact metadata from minimum viable to model-risk useful

The artifact currently captures the minimum metadata needed to reload and score a per-horizon model bundle. Production model governance will need richer provenance and validation context.

Recommended metadata additions:

- Training data window start and end.
- Training row count.
- Label positive rates by horizon.
- Validation metrics by horizon, including PR-AUC, Brier score, and precision/recall-at-K.
- Code version or git SHA.
- Training config path and config hash.
- Feature schema hash.
- Intended environment or promotion stage.
- Artifact creator, training job identifier, or run identifier.

## 3. Split validation failures from API fallback behavior

The API fallback should remain safe, but invalid caller-provided feature rows should not be silently hidden in production.

Recommended behavior:

- Continue fallback when no artifact is configured.
- Continue fallback when the configured artifact is unavailable during MVP operation, while logging that condition.
- If a caller explicitly provides `feature_rows` and validation fails, return either:
  - a structured prediction response with `data_quality_status = "FAIL"`, or
  - a clear 4xx validation response.
- Include validation errors in a controlled diagnostics field or server-side logs.
- Track validation failures as an operational metric.

## 4. Make request alignment explicit

The current request can contain both `loan_ids` and optional `feature_rows`. The relationship between those two inputs should be explicit before production use.

Recommended checks:

- Every requested `loan_id` must appear in `feature_rows` when feature rows are supplied.
- Extra feature rows should either be rejected or explicitly documented as allowed.
- Feature-row `as_of_timestamp` should match the request `as_of_timestamp`, or the API should document which timestamp is authoritative.
- Response ordering should follow request `loan_ids`.
- Duplicate loan/as-of rows should remain a hard validation failure.

## 5. Improve reason codes to reflect model contributions

The current scorer can generate reason codes from raw feature values. For a transparent scorecard, reason codes should be based on model contribution magnitudes.

Recommended approach:

1. For each model feature, compute:

   ```text
   contribution = coefficient * (feature_value - feature_median)
   ```

2. Rank positive contribution magnitudes per horizon.
3. Aggregate contributions across requested horizons when the API returns one risk band for multiple horizons.
4. Map the top contributors to business-friendly reason codes.
5. Include tests that reason codes change when feature contributions change.

## 6. Add a real batch scoring entry point

The reusable Python scoring layer is a good foundation, but scheduled scoring needs a runnable interface.

Recommended batch scoring capabilities:

- Load a JSON model artifact.
- Read scoring rows from JSON, CSV, or Parquet.
- Validate input rows before scoring.
- Score requested horizons.
- Write long-form predictions by `loan_id`, `as_of_timestamp`, and `horizon_days`.
- Emit validation diagnostics.
- Exit nonzero on hard data-quality failures.
- Include model version and feature-set version in every output row.

## 7. Add API tests for scorer and fallback paths

Current tests cover the artifact and scorer layer. API-level tests should lock in the intended runtime behavior.

Recommended API test cases:

- No artifact configured uses deterministic fallback.
- Artifact configured with valid `feature_rows` uses artifact-backed scoring.
- Artifact configured with invalid `feature_rows` follows the chosen validation behavior.
- Artifact load failure falls back safely and logs the issue.
- Requested unsupported horizon returns a controlled error.
- Request `loan_ids` and `feature_rows` mismatch is handled explicitly.
- Response ordering follows `loan_ids`.

## 8. Make scorer loading refreshable or dependency-injected

The current API loads the scorer at startup. That is simple and safe, but model rotation requires a process restart.

Recommended options:

- Keep restart-based promotion for the first production release and document it.
- Add dependency injection so tests and deployment code can provide a scorer explicitly.
- Add a small artifact registry abstraction.
- Add a controlled refresh path for model rotation if runtime reload is required.
- Record the active artifact path and model version in health or metadata endpoints.

## 9. Add structured logging and observability

Silent fallback is operationally risky unless it is visible.

Recommended signals:

- Artifact load success and failure.
- Active model version and feature-set version.
- Scorer usage count versus fallback usage count.
- Validation failure count and failure reason categories.
- Unsupported horizon requests.
- Per-request scoring latency.
- Prediction count by model version.
- Distribution of risk bands by horizon.

## 10. Add fixture artifacts for compatibility tests

The artifact schema now has an explicit version. Compatibility tests should ensure future changes do not break existing artifacts unintentionally.

Recommended tests:

- Check in a small golden artifact under `tests/fixtures/`.
- Verify the current loader can read the fixture.
- Verify predictions from the fixture stay stable.
- Add an explicit test for unsupported schema versions.
- Add migration tests if a future schema version is introduced.

## Suggested priority order

1. Clarify API validation versus fallback behavior.
2. Add artifact determinism, checksum, and richer metadata.
3. Add API integration tests for scorer and fallback paths.
4. Add a batch scoring command-line entry point.
5. Switch reason codes from raw feature values to contribution-based explanations.
6. Add structured logging and metrics.
7. Add golden artifact compatibility tests.
8. Decide and document the model promotion and scorer reload policy.

## Definition of done for the next implementation pass

A strong next implementation pass would be complete when:

- The artifact includes deterministic model content plus integrity metadata.
- API behavior is explicit for invalid feature rows.
- API tests cover both scorer-backed and fallback-backed responses.
- Batch scoring can run from a persisted artifact without importing FastAPI.
- Reason codes reflect transparent model contributions.
- Operators can tell whether the API is serving artifact scores or fallback scores.
