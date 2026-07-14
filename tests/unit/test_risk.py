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
