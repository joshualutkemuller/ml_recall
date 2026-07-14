from ml_recall.modeling.evaluation import classification_validation_metrics, precision_recall_at_k
from ml_recall.modeling.training import predict_recall_probabilities, train_horizon_models


def _training_rows() -> list[dict[str, float | int]]:
    return [
        {
            "loan_age_days": 1,
            "utilization_ratio": 0.10,
            "lender_recent_recall_rate": 0.00,
            "borrow_fee_percentile": 0.10,
            "label_recall_1d": 0,
            "label_recall_3d": 0,
        },
        {
            "loan_age_days": 10,
            "utilization_ratio": 0.90,
            "lender_recent_recall_rate": 1.00,
            "borrow_fee_percentile": 0.95,
            "label_recall_1d": 1,
            "label_recall_3d": 1,
        },
        {
            "loan_age_days": 3,
            "utilization_ratio": 0.30,
            "lender_recent_recall_rate": 0.20,
            "borrow_fee_percentile": 0.40,
            "label_recall_1d": 0,
            "label_recall_3d": 1,
        },
    ]


def test_train_and_score_horizon_models() -> None:
    feature_columns = (
        "loan_age_days",
        "utilization_ratio",
        "lender_recent_recall_rate",
        "borrow_fee_percentile",
    )
    bundle = train_horizon_models(_training_rows(), feature_columns=feature_columns, horizons=(1, 3))
    scored = predict_recall_probabilities(bundle, [_training_rows()[0]])

    assert bundle.horizons == (1, 3)
    assert 0 <= scored[0]["probability_recall_1d"] <= 1
    assert 0 <= scored[0]["probability_recall_3d"] <= 1


def test_classification_metrics_include_handoff_measures() -> None:
    metrics = classification_validation_metrics([0, 1, 1], [0.1, 0.8, 0.7])
    top_k = precision_recall_at_k([0, 1, 1], [0.1, 0.8, 0.7], k=1)

    assert metrics["pr_auc"] == 1.0
    assert 0 <= metrics["brier_score"] <= 1
    assert top_k == {"precision_at_k": 1.0, "recall_at_k": 0.5}
