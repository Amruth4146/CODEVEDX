"""Unit tests for src/data_manager.py."""

from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.data_manager import (
    SYNTHETIC_DATA_DISCLAIMER,
    add_record,
    create_or_init_csv,
    extract_features,
    generate_synthetic_sample_data,
    get_dataset_summary,
    load_data,
    preprocess_data,
    save_data,
    split_data,
    update_record,
    validate_data,
    validate_iso_date,
    validate_usage_value,
    view_all_records,
)
from src.utils import ALL_FEATURES, DATE_COLUMN, TARGET_COLUMN


# =====================================================================
# Validation Tests
# =====================================================================

def test_validate_iso_date_valid():
    """Verify valid ISO dates pass validation."""
    assert validate_iso_date("2024-01-15") == "2024-01-15"
    assert validate_iso_date("2024-02-29") == "2024-02-29"  # Leap year


def test_validate_iso_date_invalid_format():
    """Verify improper date formats raise ValueError."""
    with pytest.raises(ValueError, match="Invalid ISO date format"):
        validate_iso_date("15-01-2024")
    with pytest.raises(ValueError, match="Invalid ISO date format"):
        validate_iso_date("2024/01/15")
    with pytest.raises(ValueError, match="Invalid ISO date format"):
        validate_iso_date("not-a-date")


def test_validate_iso_date_invalid_calendar_date():
    """Verify non-existent calendar dates raise ValueError."""
    with pytest.raises(ValueError, match="Invalid calendar date"):
        validate_iso_date("2023-02-29")  # 2023 is not leap year
    with pytest.raises(ValueError, match="Invalid calendar date"):
        validate_iso_date("2024-04-31")  # April has only 30 days


def test_validate_iso_date_non_string():
    """Verify non-string inputs raise TypeError."""
    with pytest.raises(TypeError, match="Date must be a string"):
        validate_iso_date(20240101)


def test_validate_usage_value_valid():
    """Verify positive and zero values pass."""
    assert validate_usage_value(25.5) == 25.5
    assert validate_usage_value(0.0) == 0.0
    assert validate_usage_value("18.4") == 18.4


def test_validate_usage_value_negative():
    """Verify negative values raise ValueError."""
    with pytest.raises(ValueError, match="cannot be negative"):
        validate_usage_value(-5.0)


def test_validate_usage_value_nan_inf():
    """Verify NaN and Inf values raise ValueError."""
    with pytest.raises(ValueError, match="cannot be NaN"):
        validate_usage_value(float("nan"))
    with pytest.raises(ValueError, match="cannot be infinity"):
        validate_usage_value(float("inf"))


def test_validate_usage_value_invalid_type():
    """Verify non-convertible types raise TypeError."""
    with pytest.raises(TypeError, match="must be a valid number"):
        validate_usage_value("invalid_num")


# =====================================================================
# File Operations Tests
# =====================================================================

def test_create_or_init_csv():
    """Verify CSV initialization creates file with correct schema."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / "test_init.csv"
        df = create_or_init_csv(csv_path)

        assert csv_path.is_file()
        assert list(df.columns) == [DATE_COLUMN, TARGET_COLUMN]
        assert len(df) == 0

        # Accidental overwrite without flag must raise FileExistsError
        with pytest.raises(FileExistsError, match="already exists"):
            create_or_init_csv(csv_path, allow_overwrite=False)


def test_load_data_file_not_found():
    """Verify load_data raises FileNotFoundError for missing path."""
    with pytest.raises(FileNotFoundError, match="not found"):
        load_data("non_existent_path_to_data.csv")


def test_load_data_empty_file():
    """Verify load_data raises ValueError on empty file."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        empty_path = Path(tmp_dir) / "empty.csv"
        empty_path.touch()

        with pytest.raises(ValueError, match="is empty"):
            load_data(empty_path)


def test_save_data_prevent_accidental_overwrite():
    """Verify save_data enforces overwrite safety flag."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / "usage.csv"
        df = pd.DataFrame({
            DATE_COLUMN: ["2024-01-01"],
            TARGET_COLUMN: [22.5],
        })
        save_data(df, csv_path, allow_overwrite=True)

        with pytest.raises(FileExistsError, match="already exists"):
            save_data(df, csv_path, allow_overwrite=False)


# =====================================================================
# Record Operations Tests (Add, Update, View)
# =====================================================================

def test_add_record_success():
    """Verify add_record inserts new rows sorted by date."""
    initial_df = pd.DataFrame({
        DATE_COLUMN: ["2024-01-01", "2024-01-03"],
        TARGET_COLUMN: [20.0, 25.0],
    })
    # Insert in-between date
    updated_df = add_record(initial_df, date_str="2024-01-02", usage_kwh=22.0)

    assert len(updated_df) == 3
    # Check chronological order
    assert list(updated_df[DATE_COLUMN]) == ["2024-01-01", "2024-01-02", "2024-01-03"]
    assert list(updated_df[TARGET_COLUMN]) == [20.0, 22.0, 25.0]


def test_add_record_duplicate_prevention():
    """Verify add_record prevents duplicate dates."""
    initial_df = pd.DataFrame({
        DATE_COLUMN: ["2024-01-01"],
        TARGET_COLUMN: [20.0],
    })
    with pytest.raises(ValueError, match="already exists"):
        add_record(initial_df, date_str="2024-01-01", usage_kwh=25.0)


def test_update_record_success():
    """Verify update_record modifies existing usage."""
    initial_df = pd.DataFrame({
        DATE_COLUMN: ["2024-01-01", "2024-01-02"],
        TARGET_COLUMN: [20.0, 22.0],
    })
    updated_df = update_record(initial_df, date_str="2024-01-02", new_usage_kwh=29.5)

    assert len(updated_df) == 2
    row = updated_df[updated_df[DATE_COLUMN] == "2024-01-02"].iloc[0]
    assert row[TARGET_COLUMN] == 29.5


def test_update_record_not_found():
    """Verify update_record raises KeyError for missing date."""
    initial_df = pd.DataFrame({
        DATE_COLUMN: ["2024-01-01"],
        TARGET_COLUMN: [20.0],
    })
    with pytest.raises(KeyError, match="not found"):
        update_record(initial_df, date_str="2024-01-05", new_usage_kwh=30.0)


def test_view_all_records():
    """Verify view_all_records returns expected sorted slices."""
    df = pd.DataFrame({
        DATE_COLUMN: ["2024-01-03", "2024-01-01", "2024-01-02"],
        TARGET_COLUMN: [30.0, 10.0, 20.0],
    })
    view = view_all_records(df, limit=2, tail=True)

    assert len(view) == 2
    assert list(view[DATE_COLUMN]) == ["2024-01-02", "2024-01-03"]


# =====================================================================
# Synthetic Dataset Tests (Demonstration Disclaimer)
# =====================================================================

def test_generate_synthetic_sample_data():
    """Verify synthetic generation includes disclaimer and valid values."""
    days = 100
    df = generate_synthetic_sample_data(
        start_date="2024-01-01", days=days, output_path=None, random_seed=42
    )

    assert len(df) == days
    assert DATE_COLUMN in df.columns
    assert TARGET_COLUMN in df.columns
    assert (df[TARGET_COLUMN] >= 5.0).all()

    # Metadata check: synthetic disclaimer present
    assert "disclaimer" in df.attrs
    assert "DEMONSTRATION ONLY" in df.attrs["disclaimer"]


# =====================================================================
# Feature Extraction and Splitting Tests
# =====================================================================

def test_extract_features_and_split():
    """Verify feature engineering produces lag variables and valid splits."""
    df = generate_synthetic_sample_data(days=60, output_path=None, random_seed=42)
    feat_df = extract_features(df)

    for feat in ALL_FEATURES:
        assert feat in feat_df.columns

    # Lag 7 drops first 7 records
    assert len(feat_df) == 60 - 7

    X_train, X_test, y_train, y_test = split_data(feat_df, test_size=0.2, shuffle=False)
    assert len(X_train) + len(X_test) == len(feat_df)
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)


def test_get_dataset_summary():
    """Verify dataset summary calculates statistical parameters."""
    df = generate_synthetic_sample_data(days=30, output_path=None, random_seed=42)
    summary = get_dataset_summary(df)

    assert summary["total_records"] == 30
    assert summary["min_kwh"] <= summary["mean_kwh"] <= summary["max_kwh"]
