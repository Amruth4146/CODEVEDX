"""Dataset Loading and Cleaning Module for Task 3: AI-Based Fake News Detection.

Responsible for:
1. Loading raw CSV files (True.csv and Fake.csv) with robust error and missing-file handling.
2. Assigning consistent numeric target labels:
   - Real News = 1
   - Fake News = 0
3. Combining article headlines ('title') and article body ('text') into a single field: 'content'.
4. Preserving original 'title' and 'text' columns for error analysis and auditing.
5. Excluding 'subject' and 'date' to prevent structural target leakage.
6. Filtering empty, whitespace-only, or invalid records.
7. Deduplicating articles using normalized text representations to prevent train-test contamination.
8. Generating comprehensive dataset statistics.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Expected minimum required columns in raw ISOT files
REQUIRED_COLUMNS: tuple[str, ...] = ("title", "text")


def get_default_data_dir() -> Path:
    """Return the default Task-3/data directory path relative to this source file.

    Returns:
        Path object pointing to Task-3/data.
    """
    return Path(__file__).resolve().parent.parent / "data"


def load_single_csv(
    file_path: Path | str,
    expected_columns: tuple[str, ...] = REQUIRED_COLUMNS,
) -> pd.DataFrame:
    """Load and validate an individual dataset CSV file.

    Performs comprehensive validation:
    1. Checks if the file path exists.
    2. Checks if the file is non-empty.
    3. Handles parsing errors gracefully.
    4. Validates the presence of required columns ('title', 'text').

    Args:
        file_path: Path to the target CSV file.
        expected_columns: Tuple of column names that must exist in the file.

    Returns:
        Loaded pandas DataFrame.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file is empty, malformed, or missing required columns.
    """
    path_obj = Path(file_path).resolve()

    if not path_obj.exists():
        raise FileNotFoundError(
            f"Dataset file not found at: '{path_obj}'. "
            f"Please verify that the file exists and is located in the data directory."
        )

    if not path_obj.is_file():
        raise ValueError(f"Path is not a file: '{path_obj}'")

    if path_obj.stat().st_size == 0:
        raise ValueError(f"Dataset file is empty (0 bytes): '{path_obj}'")

    try:
        df = pd.read_csv(path_obj)
    except pd.errors.EmptyDataError as err:
        raise ValueError(f"Dataset file contains no data or headers: '{path_obj}'") from err
    except pd.errors.ParserError as err:
        raise ValueError(f"Malformed CSV file could not be parsed: '{path_obj}'. Error: {err}") from err
    except Exception as err:
        raise ValueError(f"Failed to read CSV file at '{path_obj}': {err}") from err

    # Validate required columns
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Dataset at '{path_obj}' is missing required column(s): {missing_cols}. "
            f"Found columns: {list(df.columns)}"
        )

    return df


def combine_title_and_text(
    df: pd.DataFrame,
    title_col: str = "title",
    text_col: str = "text",
    output_col: str = "content",
) -> pd.DataFrame:
    """Combine headline and body text into a single unified 'content' column.

    Preserves original title and body columns. Handles missing or null values
    gracefully by filling NaNs with empty strings prior to concatenation.
    Surrounding whitespace is trimmed.

    Args:
        df: Input pandas DataFrame.
        title_col: Name of the headline column.
        text_col: Name of the body text column.
        output_col: Name of the resulting combined column.

    Returns:
        DataFrame with the new output_col populated.
    """
    df_copy = df.copy()

    # Safely extract and fill NaNs
    titles = df_copy[title_col].fillna("").astype(str).str.strip() if title_col in df_copy.columns else ""
    texts = df_copy[text_col].fillna("").astype(str).str.strip() if text_col in df_copy.columns else ""

    # Combine with a space, then strip extra margins
    df_copy[output_col] = (titles + " " + texts).str.strip()

    return df_copy


def remove_empty_records(
    df: pd.DataFrame,
    content_col: str = "content",
    min_length: int = 1,
) -> pd.DataFrame:
    """Remove records where the combined content is empty or below minimum length.

    Args:
        df: Input DataFrame.
        content_col: Column to inspect for usable text.
        min_length: Minimum character length required for a record to be considered usable.

    Returns:
        Filtered DataFrame containing only records with usable text content.
    """
    if content_col not in df.columns:
        raise ValueError(f"Column '{content_col}' not found in DataFrame.")

    # Compute trimmed length
    trimmed = df[content_col].fillna("").astype(str).str.strip()
    valid_mask = trimmed.str.len() >= min_length

    return df[valid_mask].copy()


def remove_duplicates(
    df: pd.DataFrame,
    content_col: str = "content",
    keep: str = "first",
) -> pd.DataFrame:
    """Detect and remove duplicate articles based on normalized content text.

    Uses a normalized representation (lowercase + collapsed whitespace) to ensure
    articles that differ only by minor formatting or casing are identified as
    duplicates, protecting train-test splits from data leakage.

    Args:
        df: Input DataFrame.
        content_col: Text column used for deduplication.
        keep: Which duplicate to retain ('first', 'last', or False).

    Returns:
        Deduplicated DataFrame.
    """
    if content_col not in df.columns:
        raise ValueError(f"Column '{content_col}' not found in DataFrame.")

    # Create temporary normalized text series for duplicate detection
    norm_series = (
        df[content_col]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    duplicate_mask = norm_series.duplicated(keep=keep)
    return df[~duplicate_mask].copy()


def clean_dataset(
    df: pd.DataFrame,
    drop_duplicates: bool = True,
    min_content_length: int = 1,
    content_col: str = "content",
) -> pd.DataFrame:
    """Clean the dataset by filtering empty records and optionally dropping duplicates.

    Args:
        df: Input combined DataFrame.
        drop_duplicates: Whether to drop duplicate articles based on content.
        min_content_length: Minimum characters required in content.
        content_col: Name of the combined text column.

    Returns:
        Cleaned DataFrame.
    """
    # 1. Remove empty / whitespace-only content
    cleaned = remove_empty_records(df, content_col=content_col, min_length=min_content_length)

    # 2. Deduplicate articles
    if drop_duplicates:
        cleaned = remove_duplicates(cleaned, content_col=content_col, keep="first")

    return cleaned.reset_index(drop=True)


def load_and_prepare_dataset(
    raw_dir: Path | str | None = None,
    true_file: str = "True.csv",
    fake_file: str = "Fake.csv",
    clean: bool = True,
    drop_duplicates: bool = True,
    exclude_metadata: bool = True,
) -> pd.DataFrame:
    """Load, label, combine, and clean the Real and Fake news datasets.

    Pipeline steps:
    1. Resolves relative data paths (defaults to Task-3/data/raw).
    2. Loads True.csv and Fake.csv with input validation.
    3. Assigns binary target labels: Real = 1, Fake = 0.
    4. Combines 'title' and 'text' into 'content'.
    5. Concatenates both datasets into a unified DataFrame.
    6. Removes empty records and duplicates (if clean=True).
    7. Excludes 'subject' and 'date' to prevent data leakage (if exclude_metadata=True).

    Args:
        raw_dir: Directory containing True.csv and Fake.csv (defaults to Task-3/data/raw).
        true_file: Filename of the Real News CSV.
        fake_file: Filename of the Fake News CSV.
        clean: Whether to apply empty-record and duplicate filtering.
        drop_duplicates: Whether to drop duplicate articles during cleaning.
        exclude_metadata: Whether to drop 'subject' and 'date' columns from model features.

    Returns:
        Prepared pandas DataFrame ready for subsequent feature extraction.
    """
    if raw_dir is None:
        data_dir = get_default_data_dir() / "raw"
    else:
        data_dir = Path(raw_dir).resolve()

    true_path = data_dir / true_file
    fake_path = data_dir / fake_file

    # Load both CSVs
    true_df = load_single_csv(true_path)
    fake_df = load_single_csv(fake_path)

    # Assign labels
    true_df["label"] = 1
    fake_df["label"] = 0

    # Combine title and text in each subset
    true_df = combine_title_and_text(true_df)
    fake_df = combine_title_and_text(fake_df)

    # Concatenate into one corpus
    combined_df = pd.concat([true_df, fake_df], ignore_index=True)

    # Exclude metadata columns to prevent structural leakage
    if exclude_metadata:
        cols_to_drop = [c for c in ("subject", "date") if c in combined_df.columns]
        if cols_to_drop:
            combined_df = combined_df.drop(columns=cols_to_drop)

    # Apply data cleaning
    if clean:
        combined_df = clean_dataset(combined_df, drop_duplicates=drop_duplicates)

    return combined_df


def get_dataset_statistics(df: pd.DataFrame) -> dict[str, Any]:
    """Compute summary statistics for a loaded or cleaned dataset.

    Args:
        df: Input DataFrame containing at least 'label' and 'content' columns.

    Returns:
        Dictionary containing record counts, class distributions, and length statistics.
    """
    total = len(df)
    if total == 0:
        return {
            "total_records": 0,
            "real_records": 0,
            "fake_records": 0,
            "real_ratio": 0.0,
            "fake_ratio": 0.0,
            "empty_content_count": 0,
            "mean_content_length": 0.0,
            "median_content_length": 0.0,
        }

    real_count = int((df["label"] == 1).sum()) if "label" in df.columns else 0
    fake_count = int((df["label"] == 0).sum()) if "label" in df.columns else 0

    content_lengths = df["content"].fillna("").astype(str).str.len() if "content" in df.columns else pd.Series([], dtype=float)
    empty_count = int((content_lengths == 0).sum())

    return {
        "total_records": total,
        "real_records": real_count,
        "fake_records": fake_count,
        "real_ratio": round(real_count / total, 4) if total > 0 else 0.0,
        "fake_ratio": round(fake_count / total, 4) if total > 0 else 0.0,
        "empty_content_count": empty_count,
        "mean_content_length": round(float(content_lengths.mean()), 2) if len(content_lengths) > 0 else 0.0,
        "median_content_length": round(float(content_lengths.median()), 2) if len(content_lengths) > 0 else 0.0,
    }
