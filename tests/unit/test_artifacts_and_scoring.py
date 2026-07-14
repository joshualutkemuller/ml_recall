from ml_recall.modeling.artifacts import load_model_bundle, save_model_bundle, train_and_save_model_bundle
from ml_recall.modeling.training import predict_recall_probabilities, train_horizon_models
from ml_recall.scoring.pipeline import RecallScorer


def _rows():
    return [
        {"loan_id": "L1", "security_id": "S1", "lender_id": "A", "as_of_timestamp": "2026-01-02T00:00:00Z", "utilization_ratio": 0.9, "borrow_fee_percentile": 0.8, "label_recall_1d": 1, "label_recall_3d": 1},
        {"loan_id": "L2", "security_id": "S2", "lender_id": "B", "as_of_timestamp": "2026-01-02T00:00:00Z", "utilization_ratio": 0.1, "borrow_fee_percentile": 0.2, "label_recall_1d": 0, "label_recall_3d": 0},
        {"loan_id": "L3", "security_id": "S3", "lender_id": "C", "as_of_timestamp": "2026-01-02T00:00:00Z", "utilization_ratio": 0.7, "borrow_fee_percentile": 0.9, "label_recall_1d": 1, "label_recall_3d": 0},
    ]


def test_save_load_round_trips_predictions(tmp_path):
    features = ("utilization_ratio", "borrow_fee_percentile")
    bundle = train_horizon_models(_rows(), feature_columns=features, horizons=(1, 3))
    artifact_path = save_model_bundle(bundle, tmp_path / "bundle.json")

    loaded = load_model_bundle(artifact_path)

    assert loaded.horizons == (1, 3)
    assert predict_recall_probabilities(bundle, _rows()) == predict_recall_probabilities(loaded, _rows())


def test_train_and_save_helper_writes_artifact(tmp_path):
    artifact_path = tmp_path / "bundle.json"

    bundle = train_and_save_model_bundle(
        _rows(),
        feature_columns=["utilization_ratio", "borrow_fee_percentile"],
        output_path=artifact_path,
        horizons=(1, 3),
        model_version="recall_model_test",
    )

    assert artifact_path.exists()
    assert load_model_bundle(artifact_path).model_version == bundle.model_version


def test_recall_scorer_validates_and_scores_requested_horizons():
    bundle = train_horizon_models(
        _rows(), feature_columns=("utilization_ratio", "borrow_fee_percentile"), horizons=(1, 3)
    )
    scorer = RecallScorer(bundle)

    result = scorer.score(_rows()[:1], horizons=(1,))

    assert result.quality.passed
    assert result.model_version == bundle.model_version
    assert result.predictions[0].loan_id == "L1"
    assert set(result.predictions[0].probabilities) == {"1d"}
    assert result.predictions[0].data_quality_status == "PASS"


def test_recall_scorer_returns_quality_failure_without_scoring():
    bundle = train_horizon_models(
        _rows(), feature_columns=("utilization_ratio", "borrow_fee_percentile"), horizons=(1, 3)
    )
    scorer = RecallScorer(bundle)

    result = scorer.score([{"loan_id": "L1", "security_id": "S1", "as_of_timestamp": "2026-01-02"}])

    assert not result.quality.passed
    assert result.predictions == ()
    assert "missing required fields" in result.quality.errors[0]
