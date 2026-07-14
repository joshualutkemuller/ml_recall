from ml_recall.features.loan_features import build_loan_features


def test_build_loan_features_is_point_in_time() -> None:
    rows = [
        {
            "loan_id": "LN1",
            "security_id": "SEC1",
            "lender_id": "LEND1",
            "as_of_timestamp": "2026-07-14T12:00:00Z",
            "loan_open_timestamp": "2026-07-10T12:00:00Z",
            "quantity": 100,
            "available_inventory": 300,
            "borrow_fee_bps": 25,
        }
    ]
    recall_history = [
        {
            "lender_id": "LEND1",
            "recall_timestamp": "2026-07-13T12:00:00Z",
            "event_type": "LENDER_RECALL",
        },
        {
            "lender_id": "LEND1",
            "recall_timestamp": "2026-07-15T12:00:00Z",
            "event_type": "LENDER_RECALL",
        },
    ]
    market_history = [
        {"security_id": "SEC1", "as_of_timestamp": "2026-07-12T12:00:00Z", "borrow_fee_bps": 10},
        {"security_id": "SEC1", "as_of_timestamp": "2026-07-13T12:00:00Z", "borrow_fee_bps": 25},
        {"security_id": "SEC1", "as_of_timestamp": "2026-07-15T12:00:00Z", "borrow_fee_bps": 100},
    ]

    features = build_loan_features(rows, recall_history=recall_history, market_history=market_history)

    assert features[0]["loan_age_days"] == 4
    assert features[0]["utilization_ratio"] == 0.25
    assert features[0]["lender_recent_recall_count_30d"] == 1
    assert features[0]["borrow_fee_percentile"] == 1.0
