"""Risk banding and reason-code helpers for recall predictions."""

from __future__ import annotations

from collections.abc import Mapping

from ml_recall.common.config import RiskBandThresholds

_REASON_CODE_MAP = {
    "lender_recent_recall_rate": "LENDER_RECENT_RECALL_ACTIVITY",
    "days_to_proxy_event": "UPCOMING_PROXY_EVENT",
    "fund_turnover_percentile": "HIGH_FUND_TURNOVER",
    "borrow_fee_percentile": "ELEVATED_BORROW_FEE",
    "utilization_percentile": "HIGH_SECURITY_UTILIZATION",
    "loan_age_days": "SEASONED_LOAN",
}


def assign_risk_band(probability: float, thresholds: RiskBandThresholds | None = None) -> str:
    """Convert a calibrated probability into an operational risk band."""

    if not 0 <= probability <= 1:
        raise ValueError("probability must be between 0 and 1")
    cutoffs = thresholds or RiskBandThresholds()
    cutoffs.validate()
    if probability >= cutoffs.critical:
        return "CRITICAL"
    if probability >= cutoffs.high:
        return "HIGH"
    if probability >= cutoffs.moderate:
        return "MODERATE"
    return "LOW"


def generate_reason_codes(feature_contributions: Mapping[str, float], limit: int = 3) -> list[str]:
    """Return stable business reason codes from feature contribution magnitudes."""

    if limit <= 0:
        raise ValueError("limit must be positive")
    ranked = sorted(feature_contributions.items(), key=lambda item: abs(item[1]), reverse=True)
    reason_codes: list[str] = []
    for feature_name, contribution in ranked:
        if contribution <= 0:
            continue
        reason_code = _REASON_CODE_MAP.get(feature_name, feature_name.upper())
        if reason_code not in reason_codes:
            reason_codes.append(reason_code)
        if len(reason_codes) == limit:
            break
    while len(reason_codes) < limit:
        reason_codes.append("MODEL_BASELINE_RISK")
    return reason_codes
