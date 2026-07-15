import pytest

from ml_recall.common.config import RiskBandThresholds
from ml_recall.scoring.risk import assign_risk_band, generate_reason_codes


def test_assign_risk_band_default_thresholds() -> None:
    assert assign_risk_band(0.09) == "LOW"
    assert assign_risk_band(0.10) == "MODERATE"
    assert assign_risk_band(0.25) == "HIGH"
    assert assign_risk_band(0.50) == "CRITICAL"


def test_assign_risk_band_validates_probability() -> None:
    with pytest.raises(ValueError):
        assign_risk_band(1.01)


def test_assign_risk_band_allows_configured_thresholds() -> None:
    thresholds = RiskBandThresholds(moderate=0.2, high=0.4, critical=0.7)
    assert assign_risk_band(0.3, thresholds) == "MODERATE"


def test_generate_reason_codes_returns_top_positive_contributors() -> None:
    reasons = generate_reason_codes(
        {
            "borrow_fee_percentile": 0.4,
            "loan_age_days": -0.5,
            "utilization_percentile": 0.2,
            "lender_recent_recall_rate": 0.8,
        }
    )
    assert reasons == [
        "LENDER_RECENT_RECALL_ACTIVITY",
        "ELEVATED_BORROW_FEE",
        "HIGH_SECURITY_UTILIZATION",
    ]


def test_reason_codes_use_model_contributions_not_raw_feature_values() -> None:
    from ml_recall.modeling.training import HorizonModel, RecallModelBundle
    from ml_recall.scoring.pipeline import RecallScorer

    bundle = RecallModelBundle(
        models=(
            HorizonModel(
                horizon=1,
                label_column="label_recall_1d",
                feature_columns=("borrow_fee_percentile", "utilization_percentile"),
                intercept=0.0,
                coefficients={"borrow_fee_percentile": 10.0, "utilization_percentile": 0.1},
                medians={"borrow_fee_percentile": 0.5, "utilization_percentile": 0.0},
            ),
        ),
        model_version="unit_test_model",
        feature_set_version="unit_test_features",
    )
    row = {
        "loan_id": "L1",
        "security_id": "S1",
        "lender_id": "A",
        "as_of_timestamp": "2026-01-02T00:00:00Z",
        "borrow_fee_percentile": 0.6,
        "utilization_percentile": 9.0,
    }

    result = RecallScorer(bundle).score([row], horizons=(1,))

    assert result.predictions[0].reason_codes[:2] == [
        "ELEVATED_BORROW_FEE",
        "HIGH_SECURITY_UTILIZATION",
    ]


def test_reason_codes_aggregate_contributions_across_requested_horizons() -> None:
    from ml_recall.modeling.training import HorizonModel, RecallModelBundle
    from ml_recall.scoring.pipeline import RecallScorer

    bundle = RecallModelBundle(
        models=(
            HorizonModel(
                horizon=1,
                label_column="label_recall_1d",
                feature_columns=("borrow_fee_percentile", "utilization_percentile"),
                intercept=0.0,
                coefficients={"borrow_fee_percentile": 1.0, "utilization_percentile": 0.0},
                medians={"borrow_fee_percentile": 0.0, "utilization_percentile": 0.0},
            ),
            HorizonModel(
                horizon=3,
                label_column="label_recall_3d",
                feature_columns=("borrow_fee_percentile", "utilization_percentile"),
                intercept=0.0,
                coefficients={"borrow_fee_percentile": 0.0, "utilization_percentile": 2.0},
                medians={"borrow_fee_percentile": 0.0, "utilization_percentile": 0.0},
            ),
        ),
        model_version="unit_test_model",
        feature_set_version="unit_test_features",
    )
    row = {
        "loan_id": "L1",
        "security_id": "S1",
        "lender_id": "A",
        "as_of_timestamp": "2026-01-02T00:00:00Z",
        "borrow_fee_percentile": 1.0,
        "utilization_percentile": 1.0,
    }

    one_day = RecallScorer(bundle).score([row], horizons=(1,))
    both = RecallScorer(bundle).score([row], horizons=(1, 3))

    assert one_day.predictions[0].reason_codes[0] == "ELEVATED_BORROW_FEE"
    assert both.predictions[0].reason_codes[0] == "HIGH_SECURITY_UTILIZATION"
