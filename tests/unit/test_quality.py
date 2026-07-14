from ml_recall.quality.checks import validate_scoring_rows


def test_validate_scoring_rows_fails_duplicate_and_missing_fields() -> None:
    result = validate_scoring_rows(
        [
            {"loan_id": "LN1", "security_id": "SEC1", "lender_id": "L1", "as_of_timestamp": "T"},
            {"loan_id": "LN1", "security_id": "SEC1", "lender_id": "L1", "as_of_timestamp": "T"},
            {"loan_id": "LN2", "security_id": "", "lender_id": "L2", "as_of_timestamp": "T"},
        ]
    )

    assert result.status == "FAIL"
    assert any("duplicates" in error for error in result.errors)
    assert any("missing required fields" in error for error in result.errors)


def test_validate_scoring_rows_passes_with_warning() -> None:
    result = validate_scoring_rows(
        [
            {
                "loan_id": "LN1",
                "security_id": "SEC1",
                "lender_id": "L1",
                "as_of_timestamp": "T",
                "available_inventory": -1,
            }
        ]
    )

    assert result.status == "PASS"
    assert result.warnings == ("row 0 has negative available_inventory",)
