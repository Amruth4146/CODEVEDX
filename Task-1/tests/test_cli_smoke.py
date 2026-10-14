"""Comprehensive CLI smoke tests for Task-1 Utility Usage Prediction Tool.

Uses temporary directories and synthetic fixtures to ensure tests NEVER overwrite real user data.
Verifies all command-line automation flags:
- --generate-data
- --summary
- --view-records
- --add-record
- --update-record
- --train
- --model-info
- --predict
"""

from pathlib import Path
import sys
import tempfile
import pytest

from main import main
import src.data_manager
import src.predictor
import src.utils


@pytest.fixture
def isolated_cli_environment(monkeypatch):
    """Fixture providing temporary isolated paths for data and models so tests never touch real user data."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        temp_data_csv = tmp_path / "data" / "usage_data.csv"
        temp_model_joblib = tmp_path / "models" / "electricity_model.joblib"

        # Monkeypatch default paths across all modules
        monkeypatch.setattr("main.DEFAULT_DATA_PATH", temp_data_csv)
        monkeypatch.setattr("main.DEFAULT_MODEL_PATH", temp_model_joblib)
        monkeypatch.setattr("src.data_manager.DEFAULT_DATA_PATH", temp_data_csv)
        monkeypatch.setattr("src.predictor.DEFAULT_MODEL_PATH", temp_model_joblib)
        monkeypatch.setattr("src.utils.DEFAULT_DATA_PATH", temp_data_csv)
        monkeypatch.setattr("src.utils.DEFAULT_MODEL_PATH", temp_model_joblib)

        yield {
            "tmp_dir": tmp_path,
            "data_path": temp_data_csv,
            "model_path": temp_model_joblib,
        }


def test_cli_smoke_generate_data(isolated_cli_environment, monkeypatch, capsys):
    """Verify --generate-data creates demonstration CSV at the isolated path."""
    data_file = isolated_cli_environment["data_path"]
    monkeypatch.setattr(sys, "argv", ["main.py", "--generate-data"])
    main()

    captured = capsys.readouterr()
    assert "[SUCCESS] Saved" in captured.out
    assert data_file.is_file()


def test_cli_smoke_summary(isolated_cli_environment, monkeypatch, capsys):
    """Verify --summary displays consumption metrics from the dataset."""
    # First generate data
    monkeypatch.setattr(sys, "argv", ["main.py", "--generate-data"])
    main()
    capsys.readouterr()

    # Run --summary
    monkeypatch.setattr(sys, "argv", ["main.py", "--summary"])
    main()
    captured = capsys.readouterr()
    assert "General Consumption Overview" in captured.out
    assert "Total Electricity Usage" in captured.out
    assert "Average Daily Usage" in captured.out


def test_cli_smoke_view_records(isolated_cli_environment, monkeypatch, capsys):
    """Verify --view-records displays formatted records table."""
    monkeypatch.setattr(sys, "argv", ["main.py", "--generate-data"])
    main()
    capsys.readouterr()

    monkeypatch.setattr(sys, "argv", ["main.py", "--view-records", "10"])
    main()
    captured = capsys.readouterr()
    assert "kWh" in captured.out
    lines = [ln for ln in captured.out.strip().splitlines() if "kWh" in ln]
    assert len(lines) == 10


def test_cli_smoke_add_and_update_record(isolated_cli_environment, monkeypatch, capsys):
    """Verify --add-record and --update-record modify the dataset properly."""
    monkeypatch.setattr(sys, "argv", ["main.py", "--generate-data"])
    main()
    capsys.readouterr()

    # Add record
    monkeypatch.setattr(sys, "argv", ["main.py", "--add-record", "2026-02-01", "29.75"])
    main()
    captured = capsys.readouterr()
    assert "[SUCCESS] Added record for 2026-02-01: 29.75 kWh" in captured.out

    # Update record
    monkeypatch.setattr(sys, "argv", ["main.py", "--update-record", "2026-02-01", "31.25"])
    main()
    captured = capsys.readouterr()
    assert "[SUCCESS] Updated record for 2026-02-01: 31.25 kWh" in captured.out


def test_cli_smoke_train_and_model_info(isolated_cli_environment, monkeypatch, capsys):
    """Verify --train fits baseline and --model-info inspects trained artifacts."""
    monkeypatch.setattr(sys, "argv", ["main.py", "--generate-data"])
    main()
    capsys.readouterr()

    # Train model
    monkeypatch.setattr(sys, "argv", ["main.py", "--train"])
    main()
    captured = capsys.readouterr()
    assert "[SUCCESS] Model successfully trained" in captured.out
    assert isolated_cli_environment["model_path"].is_file()

    # View model info
    monkeypatch.setattr(sys, "argv", ["main.py", "--model-info"])
    main()
    captured = capsys.readouterr()
    assert "Model Architecture & Metadata" in captured.out
    assert "LinearRegression" in captured.out
    assert "Out-of-Sample Forward Evaluation" in captured.out


def test_cli_smoke_predict_future(isolated_cli_environment, monkeypatch, capsys):
    """Verify --predict outputs forecast for a future date."""
    monkeypatch.setattr(sys, "argv", ["main.py", "--generate-data"])
    main()
    monkeypatch.setattr(sys, "argv", ["main.py", "--train"])
    main()
    capsys.readouterr()

    monkeypatch.setattr(sys, "argv", ["main.py", "--predict", "--date", "2026-07-04"])
    main()
    captured = capsys.readouterr()
    assert "Predicted Electricity Consumption for 2026-07-04" in captured.out
    assert "kWh" in captured.out


def test_cli_smoke_predict_auto_trains_if_model_missing(isolated_cli_environment, monkeypatch, capsys):
    """Verify --predict triggers training if the model file is not present yet."""
    monkeypatch.setattr(sys, "argv", ["main.py", "--generate-data"])
    main()
    capsys.readouterr()

    # Predict without running --train first
    assert not isolated_cli_environment["model_path"].is_file()
    monkeypatch.setattr(sys, "argv", ["main.py", "--predict", "--date", "2026-08-15"])
    main()
    captured = capsys.readouterr()
    assert "Predicted Electricity Consumption for 2026-08-15" in captured.out
    assert isolated_cli_environment["model_path"].is_file()


def test_cli_smoke_pipeline_end_to_end(isolated_cli_environment, monkeypatch, capsys):
    """Verify complete end-to-end sequential CLI smoke workflow."""
    data_file = isolated_cli_environment["data_path"]
    model_file = isolated_cli_environment["model_path"]

    # 1: Generate data
    monkeypatch.setattr(sys, "argv", ["main.py", "--generate-data"])
    main()
    captured = capsys.readouterr()
    assert "[SUCCESS] Saved" in captured.out
    assert data_file.is_file()

    # 2: Summary
    monkeypatch.setattr(sys, "argv", ["main.py", "--summary"])
    main()
    captured = capsys.readouterr()
    assert "General Consumption Overview" in captured.out

    # 3: View records
    monkeypatch.setattr(sys, "argv", ["main.py", "--view-records", "5"])
    main()
    captured = capsys.readouterr()
    assert "kWh" in captured.out

    # 4: Add record
    monkeypatch.setattr(sys, "argv", ["main.py", "--add-record", "2026-01-01", "33.5"])
    main()
    captured = capsys.readouterr()
    assert "[SUCCESS] Added record for 2026-01-01" in captured.out

    # 5: Update record
    monkeypatch.setattr(sys, "argv", ["main.py", "--update-record", "2026-01-01", "35.0"])
    main()
    captured = capsys.readouterr()
    assert "[SUCCESS] Updated record for 2026-01-01" in captured.out

    # 6: Train
    monkeypatch.setattr(sys, "argv", ["main.py", "--train"])
    main()
    captured = capsys.readouterr()
    assert "[SUCCESS] Model successfully trained" in captured.out
    assert model_file.is_file()

    # 7: Model info
    monkeypatch.setattr(sys, "argv", ["main.py", "--model-info"])
    main()
    captured = capsys.readouterr()
    assert "Model Architecture & Metadata" in captured.out

    # 8: Predict
    monkeypatch.setattr(sys, "argv", ["main.py", "--predict", "--date", "2026-06-01"])
    main()
    captured = capsys.readouterr()
    assert "Predicted Electricity Consumption for 2026-06-01" in captured.out
