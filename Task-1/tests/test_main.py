"""Unit tests for error handling and user interaction in main.py."""

from pathlib import Path
import tempfile
import pandas as pd
import pytest

from main import (
    APPLICATION_TITLE,
    ensure_dataset_exists,
    interactive_application_loop,
    menu_view_records,
)
from src.utils import DATE_COLUMN, TARGET_COLUMN


def test_interactive_application_loop_exit(monkeypatch):
    """Verify choosing 8 exits the application cleanly with code 0."""
    inputs = iter(["8"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    with pytest.raises(SystemExit) as exc_info:
        interactive_application_loop()
    assert exc_info.value.code == 0


def test_interactive_application_loop_invalid_then_exit(monkeypatch, capsys):
    """Verify invalid menu choices produce clear error messages and do not crash."""
    # Sequence: invalid string "99", invalid string "xyz", valid exit "8"
    inputs = iter(["99", "", "xyz", "", "8"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    with pytest.raises(SystemExit) as exc_info:
        interactive_application_loop()
    assert exc_info.value.code == 0

    out = capsys.readouterr().out
    assert "Invalid choice '99'" in out
    assert "Invalid choice 'xyz'" in out
    assert "Please select a valid number between 1 and 8" in out


def test_interactive_application_loop_blank_then_exit(monkeypatch, capsys):
    """Verify blank input produces a clear warning message and does not crash."""
    # Sequence: blank input "", pause Enter "", exit "8"
    inputs = iter(["", "", "8"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    with pytest.raises(SystemExit) as exc_info:
        interactive_application_loop()
    assert exc_info.value.code == 0

    out = capsys.readouterr().out
    assert "Blank input received" in out
    assert "Please enter a valid menu number (1-8)" in out


def test_ensure_dataset_exists_missing_user_declines(monkeypatch, capsys):
    """Verify ensure_dataset_exists returns None when dataset is missing and user declines generation."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        missing_csv = Path(tmp_dir) / "missing_usage.csv"
        # User answers 'n' to generation prompt
        monkeypatch.setattr("builtins.input", lambda _: "n")

        result = ensure_dataset_exists(missing_csv)
        assert result is None

        out = capsys.readouterr().out
        assert "Dataset not found" in out
        assert "Operation aborted" in out


def test_ensure_dataset_exists_corrupted_user_declines(monkeypatch, capsys):
    """Verify ensure_dataset_exists handles corrupted CSV gracefully when user declines overwrite."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        corrupted_csv = Path(tmp_dir) / "corrupted.csv"
        corrupted_csv.write_text("date,usage_kwh\n2024-01-01,-999.0\n", encoding="utf-8")

        # User answers 'n' to overwrite prompt
        monkeypatch.setattr("builtins.input", lambda _: "n")

        result = ensure_dataset_exists(corrupted_csv)
        assert result is None

        out = capsys.readouterr().out
        assert "Corrupted or invalid dataset" in out


def test_menu_view_records_unrecognized_subchoice(monkeypatch, capsys):
    """Verify menu_view_records handles unrecognized sub-choice by defaulting safely."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        data_path = Path(tmp_dir) / "test_data.csv"
        df = pd.DataFrame({
            DATE_COLUMN: [f"2024-01-{i:02d}" for i in range(1, 10)],
            TARGET_COLUMN: [20.0 + i for i in range(1, 10)],
        })
        df.to_csv(data_path, index=False)

        # User enters unrecognized option "invalid_option"
        monkeypatch.setattr("builtins.input", lambda _: "invalid_option")

        menu_view_records(data_path)

        out = capsys.readouterr().out
        assert "Unrecognized option 'invalid_option'" in out
        assert "Defaulting to latest 20 records" in out
