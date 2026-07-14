from ml_recall.labels.horizons import build_recall_labels


def test_build_recall_labels_uses_lender_recall_events_only() -> None:
    observations = [
        {"loan_id": "LN1", "as_of_timestamp": "2026-07-10T13:00:00Z"},
        {"loan_id": "LN2", "as_of_timestamp": "2026-07-10T13:00:00Z"},
    ]
    events = [
        {
            "loan_id": "LN1",
            "recall_timestamp": "2026-07-13T13:00:00Z",
            "event_type": "LENDER_RECALL",
        },
        {
            "loan_id": "LN2",
            "recall_timestamp": "2026-07-13T13:00:00Z",
            "event_type": "BORROWER_RETURN",
        },
    ]

    labeled = build_recall_labels(observations, events, horizons=(1, 3))

    assert labeled[0]["label_recall_1d"] == 1
    assert labeled[0]["label_recall_3d"] == 1
    assert labeled[1]["label_recall_1d"] == 0
    assert labeled[1]["label_recall_3d"] == 0
