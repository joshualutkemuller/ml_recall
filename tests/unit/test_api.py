from fastapi import HTTPException

from ml_recall.api import app as api_app
from ml_recall.api.schemas import PredictionRequest
from ml_recall.modeling.training import train_horizon_models
from ml_recall.scoring.pipeline import RecallScorer


AS_OF = "2026-01-02T00:00:00Z"


def _training_rows():
    return [
        {
            "loan_id": "L1",
            "security_id": "S1",
            "lender_id": "A",
            "as_of_timestamp": AS_OF,
            "utilization_ratio": 0.9,
            "borrow_fee_percentile": 0.8,
            "label_recall_1d": 1,
            "label_recall_3d": 1,
        },
        {
            "loan_id": "L2",
            "security_id": "S2",
            "lender_id": "B",
            "as_of_timestamp": AS_OF,
            "utilization_ratio": 0.1,
            "borrow_fee_percentile": 0.2,
            "label_recall_1d": 0,
            "label_recall_3d": 0,
        },
        {
            "loan_id": "L3",
            "security_id": "S3",
            "lender_id": "C",
            "as_of_timestamp": AS_OF,
            "utilization_ratio": 0.7,
            "borrow_fee_percentile": 0.9,
            "label_recall_1d": 1,
            "label_recall_3d": 0,
        },
    ]


def _request(**overrides):
    payload = {
        "loan_ids": ["L2", "L1"],
        "as_of_timestamp": AS_OF,
        "horizons": [1],
    }
    payload.update(overrides)
    return payload


def test_api_uses_deterministic_fallback_without_configured_scorer(monkeypatch):
    monkeypatch.setattr(api_app, "SCORER", None)
    response = api_app.predict(PredictionRequest(**_request()))

    body = response.model_dump(mode="json")
    assert body["model_version"] == api_app.CONFIG.model_version
    assert [prediction["loan_id"] for prediction in body["predictions"]] == ["L2", "L1"]
    assert set(body["predictions"][0]["probabilities"]) == {"1d"}


def test_api_uses_artifact_scorer_and_preserves_request_order(monkeypatch):
    bundle = train_horizon_models(
        _training_rows(), feature_columns=("utilization_ratio", "borrow_fee_percentile"), horizons=(1, 3)
    )
    monkeypatch.setattr(api_app, "SCORER", RecallScorer(bundle))
    response = api_app.predict(
        PredictionRequest(**_request(feature_rows=[_training_rows()[1], _training_rows()[0]]))
    )

    body = response.model_dump(mode="json")
    assert body["model_version"] == bundle.model_version
    assert [prediction["loan_id"] for prediction in body["predictions"]] == ["L2", "L1"]
    assert all(prediction["data_quality_status"] == "PASS" for prediction in body["predictions"])


def test_api_rejects_invalid_feature_rows_without_fallback(monkeypatch):
    bundle = train_horizon_models(
        _training_rows(), feature_columns=("utilization_ratio", "borrow_fee_percentile"), horizons=(1, 3)
    )
    invalid_row = dict(_training_rows()[0])
    invalid_row.pop("borrow_fee_percentile")
    monkeypatch.setattr(api_app, "SCORER", RecallScorer(bundle))
    try:
        api_app.predict(PredictionRequest(**_request(loan_ids=["L1"], feature_rows=[invalid_row])))
    except HTTPException as exc:
        assert exc.status_code == 422
        assert exc.detail["data_quality_status"] == "FAIL"
        assert "missing model feature columns" in exc.detail["errors"][0]
    else:
        raise AssertionError("invalid feature rows should be rejected")


def test_api_rejects_request_and_feature_row_mismatch(monkeypatch):
    monkeypatch.setattr(api_app, "SCORER", None)
    try:
        api_app.predict(PredictionRequest(**_request(loan_ids=["L2"], feature_rows=[_training_rows()[0]])))
    except HTTPException as exc:
        assert exc.status_code == 422
        errors = exc.detail["errors"]
        assert "feature_rows missing requested loan_ids" in errors[0]
        assert "feature_rows include loan_ids not requested" in errors[1]
    else:
        raise AssertionError("mismatched feature rows should be rejected")


def test_api_rejects_unsupported_horizon_when_using_scorer(monkeypatch):
    bundle = train_horizon_models(
        _training_rows(), feature_columns=("utilization_ratio", "borrow_fee_percentile"), horizons=(1, 3)
    )
    monkeypatch.setattr(api_app, "SCORER", RecallScorer(bundle))
    try:
        api_app.predict(
            PredictionRequest(**_request(loan_ids=["L1"], horizons=[10], feature_rows=[_training_rows()[0]]))
        )
    except HTTPException as exc:
        assert exc.status_code == 422
        assert "requested horizons are not in the model bundle" in exc.detail
    else:
        raise AssertionError("unsupported horizons should be rejected")
