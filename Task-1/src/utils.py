"""Utility helpers, project constants, and statistics for electricity consumption prediction.

This module provides:
- File paths and project column constants.
- Safe console UI prompts, headers, and confirmation dialogs.
- Mathematical and statistical calculations for electricity usage using pandas and NumPy.
- Educational console explanations for utility metrics (Total kWh vs. Average Daily kWh).
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

# Regular expression pattern for strict ISO date format: YYYY-MM-DD
ISO_DATE_PATTERN = re.compile(r"^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$")

# Base directory paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DOCS_DIR = BASE_DIR / "docs"

# Default file paths
DEFAULT_DATA_PATH = DATA_DIR / "usage_data.csv"
DEFAULT_MODEL_PATH = MODELS_DIR / "electricity_model.joblib"

# Column and feature definitions
DATE_COLUMN = "date"
TARGET_COLUMN = "usage_kwh"

CALENDAR_FEATURES: List[str] = [
    "day_of_week",
    "day_of_month",
    "month",
    "quarter",
    "is_weekend",
    "day_of_year",
]

USAGE_FEATURES: List[str] = [
    "lag_1_usage",
    "lag_7_usage",
    "rolling_7_mean",
]

ALL_FEATURES: List[str] = CALENDAR_FEATURES + USAGE_FEATURES


# =====================================================================
# Console UI & Formatting Helpers
# =====================================================================

def print_header(title: str) -> None:
    """Print an attractive, standardized console header.

    Args:
        title: Title string to display.
    """
    line = "=" * 65
    print(f"\n{line}")
    print(f"  {title.upper()}")
    print(f"{line}\n")


def print_section(title: str) -> None:
    """Print a subsection divider for clear console readability.

    Args:
        title: Subsection title.
    """
    print(f"\n--- {title} ---")


def print_metric(name: str, value: float, unit: str = "", decimal_places: int = 4) -> None:
    """Print a formatted metric key-value pair.

    Args:
        name: Metric name label.
        value: Numeric value of the metric.
        unit: Optional unit suffix (e.g. kWh, %).
        decimal_places: Number of decimal digits to show.
    """
    unit_str = f" {unit}" if unit else ""
    formatted_val = f"{value:.{decimal_places}f}{unit_str}"
    print(f"  * {name:<28}: {formatted_val}")


def format_kwh(val: float) -> str:
    """Format a kilowatt-hour value with two decimal places.

    Args:
        val: Consumption value in kWh.

    Returns:
        Formatted string (e.g. '24.50 kWh').
    """
    return f"{val:.2f} kWh"


def prompt_float(
    prompt_text: str,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    default: Optional[float] = None,
) -> float:
    """Safely prompt the user for a floating-point number with validation.

    Rejects blank inputs (when no default is provided), text, NaN, infinity,
    and out-of-range values.

    Args:
        prompt_text: Question or prompt to display.
        min_val: Optional lower bound (inclusive).
        max_val: Optional upper bound (inclusive).
        default: Optional fallback if the user presses Enter without typing.

    Returns:
        Validated float entered by the user or default value.
    """
    while True:
        default_str = f" [default: {default}]" if default is not None else ""
        raw_input = input(f"{prompt_text}{default_str}: ").strip()

        if not raw_input:
            if default is not None:
                return default
            print("  [!] Input cannot be blank. Please enter a valid decimal number.")
            continue

        try:
            val = float(raw_input)
        except ValueError:
            print("  [!] Invalid input. Text is not accepted; please enter a valid decimal number.")
            continue

        if np.isnan(val) or np.isinf(val):
            print("  [!] Value must be a finite decimal number (cannot be NaN or infinity).")
            continue

        if min_val is not None and val < min_val:
            print(f"  [!] Value must be greater than or equal to {min_val}.")
            continue

        if max_val is not None and val > max_val:
            print(f"  [!] Value must be less than or equal to {max_val}.")
            continue

        return val


def prompt_date(prompt_text: str, default: Optional[str] = None) -> datetime:
    """Safely prompt the user for a date in YYYY-MM-DD format.

    Rejects blank inputs (when no default is provided), invalid formats,
    and impossible calendar dates (such as leap year discrepancies).

    Args:
        prompt_text: Question or prompt to display.
        default: Optional default date string (YYYY-MM-DD).

    Returns:
        Parsed datetime object.
    """
    while True:
        default_str = f" [default: {default}]" if default is not None else ""
        raw_input = input(f"{prompt_text} (YYYY-MM-DD){default_str}: ").strip()

        if not raw_input:
            if default is not None:
                raw_input = default
            else:
                print("  [!] Date cannot be blank. Please enter a date in YYYY-MM-DD format (e.g. 2024-05-15).")
                continue

        if not ISO_DATE_PATTERN.match(raw_input):
            print(f"  [!] Invalid date format: '{raw_input}'. Please use YYYY-MM-DD (e.g. 2024-05-15).")
            continue

        try:
            parsed = datetime.strptime(raw_input, "%Y-%m-%d")
            return parsed
        except ValueError as exc:
            print(f"  [!] Invalid calendar date '{raw_input}': {exc}")


def confirm_action(prompt_text: str, default: bool = True) -> bool:
    """Prompt the user for safe confirmation (yes/no).

    Args:
        prompt_text: Action confirmation question.
        default: Default answer if input is blank (True = Yes, False = No).

    Returns:
        True if confirmed, False otherwise.
    """
    hint = "[Y/n]" if default else "[y/N]"
    while True:
        try:
            ans = input(f"{prompt_text} {hint}: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print()
            return False

        if not ans:
            return default
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print("  [!] Please enter 'y' for yes or 'n' for no.")


# =====================================================================
# Usage Statistics Calculations (Pandas & NumPy)
# =====================================================================

def calculate_usage_statistics(df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """Calculate key electricity consumption metrics using pandas and NumPy.

    Calculates:
    - Total usage in kWh
    - Average daily usage in kWh
    - Minimum daily usage in kWh
    - Maximum daily usage in kWh
    - Total record count (number of recorded days)
    - Total and average usage over the last 7 recorded days

    Handles empty datasets and insufficient records gracefully.

    Args:
        df: Input DataFrame containing 'date' and 'usage_kwh' columns.

    Returns:
        Dictionary containing calculated metrics and status flags.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return {
            "is_valid": False,
            "error": "Dataset is empty. No usage records available.",
            "record_count": 0,
        }

    if TARGET_COLUMN not in df.columns:
        return {
            "is_valid": False,
            "error": f"Required target column '{TARGET_COLUMN}' not found in dataset.",
            "record_count": 0,
        }

    clean_df = df.copy()

    # Parse and sort chronologically by date if date column exists
    if DATE_COLUMN in clean_df.columns:
        clean_df[DATE_COLUMN] = pd.to_datetime(clean_df[DATE_COLUMN], errors="coerce")
        clean_df = clean_df.dropna(subset=[DATE_COLUMN]).sort_values(by=DATE_COLUMN)

    # Convert usage to numeric and extract NumPy array
    usage_series = pd.to_numeric(clean_df[TARGET_COLUMN], errors="coerce").dropna()
    usage_values = usage_series.to_numpy()

    if len(usage_values) == 0:
        return {
            "is_valid": False,
            "error": "Dataset contains no valid numeric usage records.",
            "record_count": 0,
        }

    count = int(len(usage_values))
    total_kwh = float(np.sum(usage_values))
    avg_kwh = float(np.mean(usage_values))
    min_kwh = float(np.min(usage_values))
    max_kwh = float(np.max(usage_values))

    # Date range strings
    if DATE_COLUMN in clean_df.columns and not clean_df.empty:
        start_date = clean_df[DATE_COLUMN].min().strftime("%Y-%m-%d")
        end_date = clean_df[DATE_COLUMN].max().strftime("%Y-%m-%d")
    else:
        start_date = "N/A"
        end_date = "N/A"

    # Trailing 7-day usage calculation
    has_sufficient_7_days = count >= 7
    if has_sufficient_7_days:
        last_7_values = usage_values[-7:]
        last_7_total_kwh = float(np.sum(last_7_values))
        last_7_avg_kwh = float(np.mean(last_7_values))
        if DATE_COLUMN in clean_df.columns:
            last_7_date_range = (
                f"{clean_df[DATE_COLUMN].iloc[-7].strftime('%Y-%m-%d')} to {end_date}"
            )
        else:
            last_7_date_range = "Last 7 recorded records"
    else:
        # Insufficient records (< 7 days)
        last_7_total_kwh = total_kwh
        last_7_avg_kwh = avg_kwh
        last_7_date_range = f"{start_date} to {end_date}"

    return {
        "is_valid": True,
        "record_count": count,
        "start_date": start_date,
        "end_date": end_date,
        "total_kwh": total_kwh,
        "average_daily_kwh": avg_kwh,
        "min_kwh": min_kwh,
        "max_kwh": max_kwh,
        "has_sufficient_7_days": has_sufficient_7_days,
        "last_7_total_kwh": last_7_total_kwh,
        "last_7_avg_kwh": last_7_avg_kwh,
        "last_7_date_range": last_7_date_range,
    }


def display_usage_statistics(df: Optional[pd.DataFrame]) -> None:
    """Display electricity consumption statistics with units, aligned formatting, and explanations.

    Outputs:
    - Record count and observed date range.
    - Total electricity usage (kWh).
    - Average daily usage (kWh).
    - Minimum and maximum daily usage (kWh).
    - Trailing 7-day usage metrics (total and average kWh) or insufficient records notice.
    - Clear conceptual breakdown explaining the difference between Total kWh and Average Daily kWh.

    Args:
        df: Input DataFrame containing usage records.
    """
    stats = calculate_usage_statistics(df)

    if not stats["is_valid"]:
        print_section("Usage Statistics")
        print(f"  [WARNING] {stats['error']}")
        print("  Please add or generate usage records first.")
        return

    print_section("General Consumption Overview")
    print_metric("Record Count", float(stats["record_count"]), unit="recorded days", decimal_places=0)
    print(f"  * Date Range                  : {stats['start_date']} to {stats['end_date']}")
    print_metric("Total Electricity Usage", stats["total_kwh"], unit="kWh", decimal_places=2)
    print_metric("Average Daily Usage", stats["average_daily_kwh"], unit="kWh", decimal_places=2)
    print_metric("Minimum Daily Usage", stats["min_kwh"], unit="kWh", decimal_places=2)
    print_metric("Maximum Daily Usage", stats["max_kwh"], unit="kWh", decimal_places=2)

    print_section("Trailing 7-Day Usage")
    if stats["has_sufficient_7_days"]:
        print(f"  * 7-Day Window                : {stats['last_7_date_range']}")
        print_metric("Last 7 Days Total Usage", stats["last_7_total_kwh"], unit="kWh", decimal_places=2)
        print_metric("Last 7 Days Daily Average", stats["last_7_avg_kwh"], unit="kWh", decimal_places=2)
    else:
        count = stats["record_count"]
        print(f"  * Notice                      : Insufficient records for a full 7-day trailing window.")
        print(f"  * Available Days              : Only {count} recorded day(s) found (minimum 7 required).")
        print_metric(f"Cumulative Usage ({count} days)", stats["last_7_total_kwh"], unit="kWh", decimal_places=2)
        print_metric(f"Daily Average ({count} days)", stats["last_7_avg_kwh"], unit="kWh", decimal_places=2)

    print_section("Understanding Your Metrics: Total kWh vs. Average Daily kWh")
    print("  * TOTAL kWh (Cumulative Energy Consumed):")
    print("    - What it is : The absolute sum of all electricity consumed over the")
    print(f"                   entire measured period ({stats['record_count']} days = {stats['total_kwh']:.2f} kWh).")
    print("    - Key Utility: Direct utility billing, gross resource consumption accounting,")
    print("                   and long-term energy budget monitoring.")
    print()
    print("  * AVERAGE DAILY kWh (Baseline Consumption Rate):")
    print("    - What it is : Total consumption divided by the number of recorded days")
    print(f"                   ({stats['total_kwh']:.2f} kWh / {stats['record_count']} days = {stats['average_daily_kwh']:.2f} kWh/day).")
    print("    - Key Utility: Normalizes energy consumption across different time frames,")
    print("                   enabling fair comparisons between months, detecting unusual spikes,")
    print("                   and establishing the baseline demand pattern for ML forecasting.")
