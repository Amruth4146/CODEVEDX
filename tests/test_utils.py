"""Unit tests for src/utils.py."""

from datetime import datetime
from io import StringIO
import sys
import numpy as np
import pandas as pd
import pytest

from src.utils import (
    ALL_FEATURES,
    CALENDAR_FEATURES,
    DATE_COLUMN,
    TARGET_COLUMN,
    USAGE_FEATURES,
    calculate_usage_statistics,
    confirm_action,
    display_usage_statistics,
    format_kwh,
    print_header,
    print_metric,
    print_section,
    prompt_date,
    prompt_float,
)


def test_feature_lists_integrity():
    """Verify feature definition consistency."""
    assert len(CALENDAR_FEATURES) > 0
    assert len(USAGE_FEATURES) > 0
    assert set(ALL_FEATURES) == set(CALENDAR_FEATURES + USAGE_FEATURES)
    assert len(ALL_FEATURES) == len(CALENDAR_FEATURES) + len(USAGE_FEATURES)


def test_print_formatting(capsys):
    """Verify printing utilities do not throw and produce output."""
    print_header("Test Header")
    print_section("Test Section")
    print_metric("Test Metric", 42.1234, unit="kWh")

    captured = capsys.readouterr()
    assert "TEST HEADER" in captured.out
    assert "Test Section" in captured.out
    assert "Test Metric" in captured.out
    assert "42.1234 kWh" in captured.out


def test_format_kwh():
    """Verify format_kwh formats numbers with 2 decimal places and unit suffix."""
    assert format_kwh(24.5) == "24.50 kWh"
    assert format_kwh(10) == "10.00 kWh"
    assert format_kwh(18.1234) == "18.12 kWh"


def test_prompt_float_with_default(monkeypatch):
    """Verify prompt_float returns default when user inputs empty string."""
    monkeypatch.setattr("builtins.input", lambda _: "")
    val = prompt_float("Enter value", default=18.5)
    assert val == 18.5


def test_prompt_date_with_default(monkeypatch):
    """Verify prompt_date returns default date when user inputs empty string."""
    monkeypatch.setattr("builtins.input", lambda _: "")
    dt = prompt_date("Enter date", default="2024-05-10")
    assert dt == datetime(2024, 5, 10)


def test_confirm_action_yes_and_no(monkeypatch):
    """Verify confirm_action responds properly to y, n, and default."""
    monkeypatch.setattr("builtins.input", lambda _: "y")
    assert confirm_action("Continue?") is True

    monkeypatch.setattr("builtins.input", lambda _: "n")
    assert confirm_action("Continue?") is False

    monkeypatch.setattr("builtins.input", lambda _: "")
    assert confirm_action("Continue?", default=True) is True
    assert confirm_action("Continue?", default=False) is False


# =====================================================================
# Usage Statistics Unit Tests
# =====================================================================

def test_calculate_usage_statistics_standard_data():
    """Verify statistical calculation on a standard 10-day dataset."""
    dates = pd.date_range("2024-01-01", periods=10, freq="D").strftime("%Y-%m-%d")
    # Usage values: 10, 20, 30, 40, 50, 60, 70, 80, 90, 100
    values = [10.0 * i for i in range(1, 11)]
    df = pd.DataFrame({DATE_COLUMN: dates, TARGET_COLUMN: values})

    stats = calculate_usage_statistics(df)
    assert stats["is_valid"] is True
    assert stats["record_count"] == 10
    assert stats["start_date"] == "2024-01-01"
    assert stats["end_date"] == "2024-01-10"
    assert stats["total_kwh"] == sum(values)  # 550.0 kWh
    assert stats["average_daily_kwh"] == sum(values) / 10.0  # 55.0 kWh
    assert stats["min_kwh"] == 10.0
    assert stats["max_kwh"] == 100.0
    assert stats["has_sufficient_7_days"] is True

    # Last 7 days: 40, 50, 60, 70, 80, 90, 100 -> sum = 490.0, avg = 70.0
    expected_last_7_sum = sum(values[-7:])
    assert stats["last_7_total_kwh"] == expected_last_7_sum
    assert stats["last_7_avg_kwh"] == expected_last_7_sum / 7.0


def test_calculate_usage_statistics_insufficient_records():
    """Verify statistical calculation handles fewer than 7 days gracefully."""
    dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
    values = [20.0, 30.0, 40.0]
    df = pd.DataFrame({DATE_COLUMN: dates, TARGET_COLUMN: values})

    stats = calculate_usage_statistics(df)
    assert stats["is_valid"] is True
    assert stats["record_count"] == 3
    assert stats["has_sufficient_7_days"] is False
    assert stats["total_kwh"] == 90.0
    assert stats["average_daily_kwh"] == 30.0
    assert stats["min_kwh"] == 20.0
    assert stats["max_kwh"] == 40.0
    assert stats["last_7_total_kwh"] == 90.0


def test_calculate_usage_statistics_empty_and_none():
    """Verify empty DataFrame or None returns invalid status dictionary."""
    stats_none = calculate_usage_statistics(None)
    assert stats_none["is_valid"] is False
    assert "empty" in stats_none["error"].lower()

    stats_empty = calculate_usage_statistics(pd.DataFrame())
    assert stats_empty["is_valid"] is False
    assert "empty" in stats_empty["error"].lower()


def test_display_usage_statistics_output(capsys):
    """Verify display_usage_statistics prints formatted table, units, and explanation."""
    dates = pd.date_range("2024-01-01", periods=8, freq="D").strftime("%Y-%m-%d")
    values = [25.0] * 8
    df = pd.DataFrame({DATE_COLUMN: dates, TARGET_COLUMN: values})

    display_usage_statistics(df)
    out = capsys.readouterr().out

    # Check key metrics and units
    assert "Total Electricity Usage" in out
    assert "200.00 kWh" in out
    assert "Average Daily Usage" in out
    assert "25.00 kWh" in out
    assert "Trailing 7-Day Usage" in out
    assert "Last 7 Days Total Usage" in out
    assert "175.00 kWh" in out

    # Check conceptual explanation of Total kWh vs Average Daily kWh
    assert "Understanding Your Metrics: Total kWh vs. Average Daily kWh" in out
    assert "TOTAL kWh" in out
    assert "AVERAGE DAILY kWh" in out


def test_display_usage_statistics_empty_warning(capsys):
    """Verify display_usage_statistics gracefully handles empty data with warning."""
    display_usage_statistics(pd.DataFrame())
    out = capsys.readouterr().out
    assert "[WARNING]" in out
    assert "empty" in out.lower()


# =====================================================================
# Error Handling Audit Tests for Utilities
# =====================================================================

def test_prompt_float_rejects_blank_nan_inf_and_text(monkeypatch, capsys):
    """Verify prompt_float rejects blank, text, NaN, and Infinity before accepting valid input."""
    # Sequence of bad inputs followed by valid 28.5:
    # 1: "" (blank, no default)
    # 2: "abc" (text)
    # 3: "nan" (NaN)
    # 4: "inf" (+Infinity)
    # 5: "-inf" (-Infinity)
    # 6: "-5.0" (violates min_val=0.0)
    # 7: "28.5" (valid)
    inputs = iter(["", "abc", "nan", "inf", "-inf", "-5.0", "28.5"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    result = prompt_float("Enter usage", min_val=0.0)
    assert result == 28.5

    out = capsys.readouterr().out
    assert "Input cannot be blank" in out
    assert "Text is not accepted" in out
    assert "cannot be NaN or infinity" in out
    assert "greater than or equal to 0.0" in out


def test_prompt_date_rejects_blank_format_and_calendar(monkeypatch, capsys):
    """Verify prompt_date rejects blank, wrong pattern, and invalid leap day before valid date."""
    # Sequence of bad inputs followed by valid "2024-02-29":
    # 1: "" (blank, no default)
    # 2: "2024/02/29" (wrong delimiter)
    # 3: "29-02-2024" (DD-MM-YYYY)
    # 4: "2023-02-29" (2023 is not a leap year)
    # 5: "2024-02-29" (valid leap date)
    inputs = iter(["", "2024/02/29", "29-02-2024", "2023-02-29", "2024-02-29"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    dt = prompt_date("Enter date")
    assert dt == datetime(2024, 2, 29)

    out = capsys.readouterr().out
    assert "Date cannot be blank" in out
    assert "Invalid date format" in out
    assert "Invalid calendar date" in out

