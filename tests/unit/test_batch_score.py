import json

from ml_recall.cli.batch_score import main
from ml_recall.modeling.artifacts import ReproducibilityMetadata, save_model_bundle
from ml_recall.modeling.training import train_horizon_models


AS_OF = "2026-01-02T00:00:00Z"


def _rows():
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
    ]


def _artifact_path(tmp_path):
    bundle = train_horizon_models(
        _rows(), feature_columns=("utilization_ratio", "borrow_fee_percentile"), horizons=(1, 3)
    )
    return save_model_bundle(
        bundle,
        tmp_path / "bundle.json",
        reproducibility=ReproducibilityMetadata(
            dataset_version="unit_test_dataset",
            label_version="recall_labels_v1",
            training_window_start="2025-01-01",
            training_window_end="2025-12-31",
            code_commit="abc1234",
            random_seed=7,
        ),
    )


def test_batch_score_json_writes_long_form_predictions_and_diagnostics(tmp_path):
    artifact_path = _artifact_path(tmp_path)
    input_path = tmp_path / "rows.json"
    output_path = tmp_path / "predictions.json"
    diagnostics_path = tmp_path / "diagnostics.json"
    input_path.write_text(json.dumps(_rows()))

    exit_code = main(
        [
            "--artifact",
            str(artifact_path),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--horizons",
            "1",
            "--diagnostics",
            str(diagnostics_path),
        ]
    )

    predictions = json.loads(output_path.read_text())
    diagnostics = json.loads(diagnostics_path.read_text())
    assert exit_code == 0
    assert len(predictions) == 2
    assert set(predictions[0]) == {
        "loan_id",
        "as_of_timestamp",
        "horizon_days",
        "probability",
        "risk_band",
        "reason_codes",
        "data_quality_status",
        "model_version",
        "feature_set_version",
    }
    assert {row["horizon_days"] for row in predictions} == {1}
    assert diagnostics["data_quality_status"] == "PASS"
    assert diagnostics["prediction_count"] == 2


def test_batch_score_exits_nonzero_on_hard_data_quality_failure(tmp_path):
    artifact_path = _artifact_path(tmp_path)
    input_path = tmp_path / "rows.json"
    output_path = tmp_path / "predictions.json"
    diagnostics_path = tmp_path / "diagnostics.json"
    invalid_row = dict(_rows()[0])
    invalid_row.pop("borrow_fee_percentile")
    input_path.write_text(json.dumps([invalid_row]))

    exit_code = main(
        [
            "--artifact",
            str(artifact_path),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--diagnostics",
            str(diagnostics_path),
        ]
    )

    diagnostics = json.loads(diagnostics_path.read_text())
    assert exit_code == 2
    assert not output_path.exists()
    assert diagnostics["data_quality_status"] == "FAIL"
    assert "missing model feature columns" in diagnostics["errors"][0]
