"""Data Manager module for daily electricity consumption records.

This module provides robust file I/O and data management utilities:
- Create, load, and safely persist CSV datasets in data/usage_data.csv.
- Strict input validation for ISO dates (YYYY-MM-DD) and finite non-negative kWh values.
- Duplicate detection, record addition, in-place updates, and chronological sorting.
- Graceful exception handling (FileNotFoundError, PermissionError, empty files, malformed rows).
- Generation of clearly labeled synthetic demonstration datasets (never presented as actual data).
- Feature engineering (calendar variables and historical usage lags) for ML pipelines.
"""

from datetime import datetime
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.utils import (
    ALL_FEATURES,
    CALENDAR_FEATURES,
    DATE_COLUMN,
    DEFAULT_DATA_PATH,
    TARGET_COLUMN,
    USAGE_FEATURES,
)

logger = logging.getLogger(__name__)

# Regular expression pattern for strict ISO date format: YYYY-MM-DD
ISO_DATE_PATTERN = re.compile(r"^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$")

# Demonstration disclaimer banner
SYNTHETIC_DATA_DISCLAIMER = (
    "DEMONSTRATION ONLY: Synthetic records generated for modeling demonstration. "
    "Never present or treat synthetic data as actual utility consumption."
)


# =====================================================================
# Validation Helpers
# =====================================================================

def validate_iso_date(date_str: Any) -> str:
    """Validate that a date string strictly follows the ISO 8601 format (YYYY-MM-DD).

    Args:
        date_str: Date value to validate.

    Returns:
        Validated date string in YYYY-MM-DD format.

    Raises:
        TypeError: If input is not a string.
        ValueError: If format is not YYYY-MM-DD or represents an impossible calendar date.
    """
    if not isinstance(date_str, str):
        raise TypeError(f"Date must be a string, received {type(date_str).__name__}.")

    trimmed = date_str.strip()
    if not ISO_DATE_PATTERN.match(trimmed):
        raise ValueError(
            f"Invalid ISO date format: '{date_str}'. Expected format is YYYY-MM-DD (e.g. 2024-05-15)."
        )

    # Validate actual calendar date (e.g., leap years, days per month)
    try:
        datetime.strptime(trimmed, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"Invalid calendar date '{date_str}': {exc}") from exc

    return trimmed


def validate_usage_value(value: Any) -> float:
    """Validate that electricity consumption is a finite, non-negative numeric value.

    Args:
        value: Electricity consumption value in kWh.

    Returns:
        Validated non-negative float value.

    Raises:
        TypeError: If input cannot be converted to a float.
        ValueError: If input is negative, NaN, or infinite.
    """
    try:
        val = float(value)
    except (ValueError, TypeError) as exc:
        raise TypeError(f"Usage value must be a valid number, received: {value}") from exc

    if np.isnan(val):
        raise ValueError("Electricity usage cannot be NaN.")

    if np.isinf(val):
        raise ValueError("Electricity usage must be a finite number, cannot be infinity.")

    if val < 0.0:
        raise ValueError(
            f"Electricity usage cannot be negative. Received: {val:.2f} kWh (expected >= 0.0)."
        )

    return val


def validate_data(df: pd.DataFrame) -> bool:
    """Validate DataFrame integrity, required columns, and value domains.

    Strictly checks:
    - Expected DataFrame type and non-empty status.
    - Presence of required 'date' and 'usage_kwh' columns.
    - Valid ISO 8601 YYYY-MM-DD date formats and real calendar dates (no silent drops).
    - Uniqueness of dates (rejecting duplicate date entries).
    - Numeric validity of electricity consumption (rejects text, NaN, infinity, and negatives).

    Args:
        df: Input DataFrame to check.

    Returns:
        True if validation passes.

    Raises:
        TypeError: If input is not a pandas DataFrame.
        ValueError: If validation criteria are violated.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, received {type(df).__name__}.")

    if df.empty:
        raise ValueError("Dataset is empty. At least one record is required.")

    if DATE_COLUMN not in df.columns:
        raise ValueError(f"Missing required column: '{DATE_COLUMN}'")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing required column: '{TARGET_COLUMN}'")

    if df[DATE_COLUMN].isnull().any():
        raise ValueError(f"Dataset contains missing or null dates in '{DATE_COLUMN}'.")

    # Validate ISO date format YYYY-MM-DD
    date_strs = df[DATE_COLUMN].astype(str).str.strip()
    pattern_mask = date_strs.str.match(r"^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$")
    if not pattern_mask.all():
        invalid_examples = date_strs[~pattern_mask].tolist()[:3]
        raise ValueError(
            f"Dataset contains invalid date format(s) in '{DATE_COLUMN}': {invalid_examples}. "
            f"Dates must strictly follow ISO 8601 format YYYY-MM-DD (e.g. 2024-05-15)."
        )

    # Check for impossible calendar dates (e.g. 2023-02-29 or 2024-04-31)
    parsed_dates = pd.to_datetime(date_strs, format="%Y-%m-%d", errors="coerce")
    if parsed_dates.isnull().any():
        bad_calendar_dates = date_strs[parsed_dates.isnull()].tolist()[:3]
        raise ValueError(
            f"Dataset contains non-existent calendar date(s) in '{DATE_COLUMN}': {bad_calendar_dates}."
        )

    # Check for duplicate dates
    duplicate_mask = df.duplicated(subset=[DATE_COLUMN], keep=False)
    if duplicate_mask.any():
        dups = df.loc[duplicate_mask, DATE_COLUMN].unique().tolist()[:3]
        raise ValueError(
            f"Dataset contains duplicate date records: {dups}. "
            f"Every date entry in the dataset must be unique."
        )

    # Validate usage values
    raw_target = df[TARGET_COLUMN]
    numeric_target = pd.to_numeric(raw_target, errors="coerce")
    if numeric_target.isnull().any():
        bad_vals = raw_target[numeric_target.isnull()].tolist()[:3]
        raise ValueError(
            f"Dataset contains non-numeric, NaN, or missing usage values in '{TARGET_COLUMN}': {bad_vals}."
        )

    if np.isinf(numeric_target).any():
        raise ValueError(
            f"Dataset contains infinite values in '{TARGET_COLUMN}'. Electricity usage must be finite."
        )

    if (numeric_target < 0.0).any():
        neg_vals = numeric_target[numeric_target < 0.0].tolist()[:3]
        raise ValueError(
            f"Electricity consumption cannot contain negative values. Found: {neg_vals} kWh (expected >= 0.0)."
        )

    return True


# =====================================================================
# File Operations (Create, Load, Save)
# =====================================================================

def create_or_init_csv(
    file_path: Union[str, Path] = DEFAULT_DATA_PATH,
    allow_overwrite: bool = False,
) -> pd.DataFrame:
    """Create an empty usage dataset CSV with standard columns.

    Args:
        file_path: Path to the target CSV file.
        allow_overwrite: If False, prevents accidental overwrites of existing files.

    Returns:
        Empty DataFrame with [DATE_COLUMN, TARGET_COLUMN] schema.

    Raises:
        FileExistsError: If file already exists and allow_overwrite is False.
        PermissionError: If write permission is denied.
    """
    path = Path(file_path)
    if path.exists() and not allow_overwrite:
        raise FileExistsError(
            f"File already exists at '{path}'. Set allow_overwrite=True to overwrite."
        )

    empty_df = pd.DataFrame(columns=[DATE_COLUMN, TARGET_COLUMN])
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        empty_df.to_csv(path, index=False)
    except PermissionError as exc:
        raise PermissionError(
            f"Permission denied while writing to '{path}': {exc}. Check directory write permissions."
        ) from exc

    return empty_df


def load_data(file_path: Union[str, Path] = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load daily electricity consumption dataset from a CSV file.

    Handles FileNotFoundError, PermissionError, empty files, and malformed rows gracefully.
    Never silently discards malformed or corrupted records.

    Args:
        file_path: Path to the CSV file.

    Returns:
        Validated pd.DataFrame sorted chronologically.

    Raises:
        FileNotFoundError: If file does not exist.
        PermissionError: If file cannot be accessed due to permissions.
        ValueError: If file is empty, malformed, or missing required columns.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Electricity usage data file not found at: '{path.resolve()}'."
        )

    try:
        df = pd.read_csv(
            path,
            dtype={DATE_COLUMN: str},
            on_bad_lines="error",  # Never silently discard malformed rows
        )
    except PermissionError as exc:
        raise PermissionError(
            f"Permission denied reading '{path}': {exc}. Check file read permissions."
        ) from exc
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"The CSV file at '{path}' is empty. At least one header and one record are required.") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(
            f"Corrupted or malformed CSV file at '{path}': {exc}.\n"
            f"One or more rows have an unexpected number of delimiters or corrupted data. "
            f"Please repair the CSV file or regenerate a clean demonstration dataset using Menu Option 5."
        ) from exc
    except Exception as exc:
        raise ValueError(f"Failed to parse CSV file at '{path}': {exc}") from exc

    if df.empty:
        raise ValueError(f"The CSV file at '{path}' contains no records (empty dataset).")

    validate_data(df)
    return preprocess_data(df)


def save_data(
    df: pd.DataFrame,
    file_path: Union[str, Path] = DEFAULT_DATA_PATH,
    allow_overwrite: bool = True,
) -> Path:
    """Save records to CSV, strictly sorted chronologically and preventing accidental overwrites.

    Args:
        df: DataFrame containing date and usage_kwh records.
        file_path: Target path for the CSV.
        allow_overwrite: If False and target exists, prevents overwrite.

    Returns:
        Path of the saved file.

    Raises:
        FileExistsError: If file exists and allow_overwrite is False.
        PermissionError: If write permission is denied.
        ValueError: If validation fails.
    """
    path = Path(file_path)
    if path.exists() and not allow_overwrite:
        raise FileExistsError(
            f"File already exists at '{path}'. Set allow_overwrite=True to overwrite."
        )

    validate_data(df)

    # Clean and sort chronologically prior to saving
    sorted_df = preprocess_data(df)

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        sorted_df.to_csv(path, index=False)
    except PermissionError as exc:
        raise PermissionError(
            f"Permission denied writing to '{path}': {exc}. Check file write permissions."
        ) from exc
    except OSError as exc:
        raise IOError(f"OS error writing to '{path}': {exc}") from exc

    return path


# =====================================================================
# Record Operations (Add, Update, View)
# =====================================================================

def add_record(
    df: pd.DataFrame,
    date_str: str,
    usage_kwh: float,
) -> pd.DataFrame:
    """Add a new daily electricity consumption record.

    Validates date format, finite non-negative usage, and duplicate dates.

    Args:
        df: Existing DataFrame.
        date_str: ISO date string in YYYY-MM-DD format.
        usage_kwh: Consumption value in kWh.

    Returns:
        New updated pd.DataFrame sorted by date.

    Raises:
        ValueError: If date is duplicate, date format is invalid, or usage is negative/non-finite.
    """
    valid_date = validate_iso_date(date_str)
    valid_usage = validate_usage_value(usage_kwh)

    # Check for duplicate date in existing records
    if not df.empty and DATE_COLUMN in df.columns:
        existing_dates = df[DATE_COLUMN].astype(str).str[:10].values
        if valid_date in existing_dates:
            raise ValueError(
                f"Record for date '{valid_date}' already exists. Use update_record() to modify it."
            )

    new_row = pd.DataFrame([{DATE_COLUMN: valid_date, TARGET_COLUMN: valid_usage}])
    updated_df = pd.concat([df, new_row], ignore_index=True)
    return preprocess_data(updated_df)


def update_record(
    df: pd.DataFrame,
    date_str: str,
    new_usage_kwh: float,
) -> pd.DataFrame:
    """Update electricity consumption for an existing date record.

    Args:
        df: Existing DataFrame.
        date_str: ISO date string in YYYY-MM-DD format.
        new_usage_kwh: New consumption value in kWh.

    Returns:
        Updated pd.DataFrame.

    Raises:
        KeyError: If the specified date does not exist in the dataset.
        ValueError: If date format or usage value is invalid.
    """
    valid_date = validate_iso_date(date_str)
    valid_usage = validate_usage_value(new_usage_kwh)

    updated_df = df.copy()
    if updated_df.empty or DATE_COLUMN not in updated_df.columns:
        raise KeyError(f"Date '{valid_date}' not found. Dataset is empty.")

    dates_str = updated_df[DATE_COLUMN].astype(str).str[:10]
    match_mask = dates_str == valid_date

    if not match_mask.any():
        raise KeyError(f"Record for date '{valid_date}' not found in dataset.")

    updated_df.loc[match_mask, TARGET_COLUMN] = valid_usage
    return preprocess_data(updated_df)


def view_all_records(
    df: pd.DataFrame,
    limit: Optional[int] = None,
    tail: bool = False,
) -> pd.DataFrame:
    """View sorted electricity consumption records.

    Args:
        df: DataFrame to view.
        limit: Optional maximum number of records to return.
        tail: If True, returns the most recent records; otherwise oldest.

    Returns:
        pd.DataFrame view of sorted records.
    """
    sorted_df = preprocess_data(df)
    if limit is None:
        return sorted_df

    if tail:
        return sorted_df.tail(limit).reset_index(drop=True)
    return sorted_df.head(limit).reset_index(drop=True)


# =====================================================================
# Synthetic Dataset Generation (Clearly Labeled for Demonstration)
# =====================================================================

def generate_synthetic_sample_data(
    start_date: str = "2023-01-01",
    days: int = 730,
    output_path: Optional[Union[str, Path]] = DEFAULT_DATA_PATH,
    random_seed: int = 42,
    allow_overwrite: bool = True,
) -> pd.DataFrame:
    """Generate a clearly labeled synthetic sample dataset for demonstration purposes.

    IMPORTANT: Synthetic records are generated strictly for statistical and machine
    learning demonstration. Synthetic data must NEVER be presented as actual utility records.

    Args:
        start_date: Starting date in ISO format YYYY-MM-DD.
        days: Number of continuous daily records (default 730 = ~2 years).
        output_path: Optional destination CSV path.
        random_seed: Random state seed for reproducible demonstrations.
        allow_overwrite: If False, prevents overwriting existing files.

    Returns:
        pd.DataFrame containing 'date' and 'usage_kwh' columns.

    Raises:
        FileExistsError: If destination exists and allow_overwrite is False.
    """
    valid_start = validate_iso_date(start_date)
    rng = np.random.RandomState(random_seed)
    date_range = pd.date_range(start=valid_start, periods=days, freq="D")

    base_kwh = 24.0
    day_of_year = date_range.dayofyear
    day_of_week = date_range.dayofweek

    # Realistic seasonal waveform: cooling peak in summer and heating peak in winter
    seasonal_effect = 6.0 * np.cos(2 * np.pi * (day_of_year - 200) / 365) + 3.0 * np.cos(
        4 * np.pi * day_of_year / 365
    )

    # Weekend effect: higher domestic energy demand on Saturdays and Sundays
    weekend_effect = np.where(day_of_week >= 5, 4.5, 0.0)

    # Mild long-term trend
    trend = np.linspace(0, 1.5, days)

    # Natural stochastic noise
    noise = rng.normal(loc=0.0, scale=1.8, size=days)

    usage = base_kwh + seasonal_effect + weekend_effect + trend + noise
    # Ensure physical realism: finite non-negative electricity usage (minimum 5.0 kWh baseline)
    usage = np.clip(usage, a_min=5.0, a_max=None)
    usage = np.round(usage, 2)

    df = pd.DataFrame({
        DATE_COLUMN: date_range.strftime("%Y-%m-%d"),
        TARGET_COLUMN: usage,
    })

    # Embed disclaimer in DataFrame metadata attributes
    df.attrs["data_type"] = "SYNTHETIC_DEMONSTRATION"
    df.attrs["disclaimer"] = SYNTHETIC_DATA_DISCLAIMER

    if output_path is not None:
        save_data(df, file_path=output_path, allow_overwrite=allow_overwrite)

    return df


# Alias for backwards compatibility
generate_synthetic_data = generate_synthetic_sample_data


# =====================================================================
# Preprocessing and Feature Engineering
# =====================================================================

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean dataset, ensure chronological ordering, and handle missing values.

    Args:
        df: Raw DataFrame.

    Returns:
        Cleaned pd.DataFrame strictly formatted with ISO date strings and float usage.
    """
    clean_df = df.copy()

    # Convert date to standard string YYYY-MM-DD
    clean_df[DATE_COLUMN] = pd.to_datetime(clean_df[DATE_COLUMN], errors="coerce").dt.strftime(
        "%Y-%m-%d"
    )
    clean_df = clean_df.dropna(subset=[DATE_COLUMN])

    # Convert target to float
    clean_df[TARGET_COLUMN] = pd.to_numeric(clean_df[TARGET_COLUMN], errors="coerce")

    # Drop duplicate dates, keeping first
    clean_df = clean_df.drop_duplicates(subset=[DATE_COLUMN], keep="first")

    # Sort strictly chronologically by date
    clean_df = clean_df.sort_values(by=DATE_COLUMN).reset_index(drop=True)

    # Impute missing values with forward fill, then backward fill if needed
    clean_df[TARGET_COLUMN] = clean_df[TARGET_COLUMN].ffill().bfill()
    clean_df[TARGET_COLUMN] = clean_df[TARGET_COLUMN].round(2)

    return clean_df


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract calendar-based and historical usage lag features.

    Features generated:
    - Calendar: day_of_week, day_of_month, month, quarter, is_weekend, day_of_year
    - Historical usage: lag_1_usage (yesterday), lag_7_usage (last week same day),
                        rolling_7_mean (7-day trailing average)

    Args:
        df: Cleaned DataFrame with 'date' and 'usage_kwh' columns.

    Returns:
        Feature-enriched pd.DataFrame with invalid initial lag rows dropped.
    """
    feat_df = preprocess_data(df)
    dt_series = pd.to_datetime(feat_df[DATE_COLUMN])

    # 1. Calendar Features
    feat_df["day_of_week"] = dt_series.dt.dayofweek
    feat_df["day_of_month"] = dt_series.dt.day
    feat_df["month"] = dt_series.dt.month
    feat_df["quarter"] = dt_series.dt.quarter
    feat_df["is_weekend"] = (feat_df["day_of_week"] >= 5).astype(int)
    feat_df["day_of_year"] = dt_series.dt.dayofyear

    # 2. Historical Usage Features
    feat_df["lag_1_usage"] = feat_df[TARGET_COLUMN].shift(1)
    feat_df["lag_7_usage"] = feat_df[TARGET_COLUMN].shift(7)
    feat_df["rolling_7_mean"] = (
        feat_df[TARGET_COLUMN].shift(1).rolling(window=7, min_periods=7).mean()
    )

    # Drop the first 7 days that do not have full lag history
    feat_df = feat_df.dropna(subset=USAGE_FEATURES).reset_index(drop=True)
    return feat_df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    shuffle: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split feature dataset into train and test sets.

    Args:
        df: Enriched DataFrame containing ALL_FEATURES and TARGET_COLUMN.
        test_size: Fraction of samples reserved for testing (default 0.2).
        shuffle: Whether to shuffle data (False preserves chronological order).

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    X = df[ALL_FEATURES]
    y = df[TARGET_COLUMN]

    if not shuffle:
        split_idx = int(len(df) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    else:
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

    return X_train, X_test, y_train, y_test


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Union[int, float, str]]:
    """Compute summary statistics for electricity consumption data.

    Args:
        df: Input DataFrame.

    Returns:
        Dictionary containing statistical metrics.
    """
    clean_df = preprocess_data(df)
    total_records = len(clean_df)
    min_date = str(clean_df[DATE_COLUMN].min())
    max_date = str(clean_df[DATE_COLUMN].max())
    mean_val = float(clean_df[TARGET_COLUMN].mean())
    std_val = float(clean_df[TARGET_COLUMN].std())
    min_val = float(clean_df[TARGET_COLUMN].min())
    max_val = float(clean_df[TARGET_COLUMN].max())
    median_val = float(clean_df[TARGET_COLUMN].median())

    return {
        "total_records": total_records,
        "start_date": min_date,
        "end_date": max_date,
        "mean_kwh": mean_val,
        "std_kwh": std_val,
        "min_kwh": min_val,
        "max_kwh": max_val,
        "median_kwh": median_val,
    }
