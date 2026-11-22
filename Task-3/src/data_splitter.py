"""Dataset Splitting Module for Task 3: AI-Based Fake News Detection.

Provides reproducible, stratified splitting of preprocessed datasets into:
- Training set (approx. 80%): for fitting feature extractors and training classifiers.
- Validation set (approx. 10%): for model selection, threshold tuning, and hyperparameters.
- Test set (approx. 10%): isolated evaluation set strictly held out until final benchmark.

Key safety features:
1. Strict Stratification: Preserves the exact Real/Fake target class distribution across splits.
2. Reproducibility: Uses a fixed random seed (default=42).
3. Leakage Verification: Validates that no duplicate article text crosses split boundaries.
4. Descriptive Statistics: Generates class counts and proportion summaries for each partition.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


@dataclass
class SplitResult:
    """Container holding the three isolated dataset partitions."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def __iter__(self):
        """Allow unpacking: train, val, test = split_dataset(...)."""
        return iter((self.train, self.val, self.test))


def split_dataset(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state: int = 42,
    label_col: str = "label",
) -> SplitResult:
    """Split a DataFrame into stratified train, validation, and test sets.

    Uses a robust two-stage stratified split:
    Stage 1: Separates the Training partition (train_ratio) from the combined holdout (val + test).
    Stage 2: Divides the holdout into equal Validation and Test sets.

    Args:
        df: Input pandas DataFrame containing features and target labels.
        train_ratio: Fraction of data allocated to training (default 0.8).
        val_ratio: Fraction of data allocated to validation (default 0.1).
        test_ratio: Fraction of data allocated to testing (default 0.1).
        random_state: Seed for reproducible pseudorandom splitting (default 42).
        label_col: Name of the binary target label column (default 'label').

    Returns:
        SplitResult dataclass with train, val, and test DataFrames.

    Raises:
        ValueError: If df is empty, ratios do not sum to 1.0, or label_col is missing/invalid.
    """
    if not isinstance(df, pd.DataFrame) or len(df) == 0:
        raise ValueError("Cannot split an empty or non-DataFrame input.")

    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-4:
        raise ValueError(
            f"Split ratios must sum to 1.0. Given: train={train_ratio}, "
            f"val={val_ratio}, test={test_ratio} (sum={total_ratio:.4f})"
        )

    if any(r <= 0 for r in (train_ratio, val_ratio, test_ratio)):
        raise ValueError("Each partition ratio (train, val, test) must be strictly greater than 0.")

    # Validate label column
    if label_col not in df.columns:
        raise ValueError(f"Target column '{label_col}' not found in DataFrame columns: {list(df.columns)}")

    unique_classes = df[label_col].unique()
    if len(unique_classes) < 2:
        raise ValueError(
            f"Stratified splitting requires at least 2 target classes. Found: {len(unique_classes)} class ({unique_classes})"
        )

    # Verify each class has enough members for a 2-stage split (minimum 2 samples per class in temp)
    min_class_count = df[label_col].value_counts().min()
    if min_class_count < 4:
        raise ValueError(
            f"Smallest class has only {min_class_count} sample(s). "
            f"Stratified 3-way split requires at least 4 samples per class."
        )

    # Stage 1: Split into Train and Temp (Validation + Test)
    temp_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        random_state=random_state,
        stratify=df[label_col],
    )

    # Stage 2: Split Temp into Validation and Test
    # Proportion of test within the temp set
    test_share_of_temp = test_ratio / temp_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_share_of_temp,
        random_state=random_state,
        stratify=temp_df[label_col],
    )

    return SplitResult(
        train=train_df.reset_index(drop=True),
        val=val_df.reset_index(drop=True),
        test=test_df.reset_index(drop=True),
    )


def verify_no_data_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    content_col: str = "content",
) -> dict[str, Any]:
    """Verify that no duplicate article text exists across split boundaries.

    Uses normalized text representations (lowercased + collapsed whitespace)
    to detect subtle cross-split duplications that could cause optimistic bias.

    Args:
        train_df: Training set DataFrame.
        val_df: Validation set DataFrame.
        test_df: Test set DataFrame.
        content_col: Column containing article text to verify.

    Returns:
        Dictionary reporting pairwise overlap counts and a boolean 'has_leakage' flag.
    """
    if any(content_col not in df.columns for df in (train_df, val_df, test_df)):
        raise ValueError(f"Content column '{content_col}' must be present in all split DataFrames.")

    def _normalize(series: pd.Series) -> set[str]:
        return set(
            series.fillna("")
            .astype(str)
            .str.lower()
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    train_set = _normalize(train_df[content_col])
    val_set = _normalize(val_df[content_col])
    test_set = _normalize(test_df[content_col])

    # Exclude empty string from overlap count if present
    train_set.discard("")
    val_set.discard("")
    test_set.discard("")

    train_val_overlap = len(train_set.intersection(val_set))
    train_test_overlap = len(train_set.intersection(test_set))
    val_test_overlap = len(val_set.intersection(test_set))

    has_leakage = (train_val_overlap > 0) or (train_test_overlap > 0) or (val_test_overlap > 0)

    return {
        "train_val_overlap": train_val_overlap,
        "train_test_overlap": train_test_overlap,
        "val_test_overlap": val_test_overlap,
        "has_leakage": has_leakage,
    }


def get_split_statistics(splits: SplitResult, label_col: str = "label") -> dict[str, Any]:
    """Calculate descriptive metrics for each partition in a SplitResult.

    Args:
        splits: SplitResult containing train, val, and test DataFrames.
        label_col: Name of the binary target label column.

    Returns:
        Dictionary of partition counts, class distributions, and proportions.
    """
    total = len(splits.train) + len(splits.val) + len(splits.test)

    def _stats(df: pd.DataFrame, name: str) -> dict[str, Any]:
        count = len(df)
        pct = round((count / total) * 100, 2) if total > 0 else 0.0
        real_count = int((df[label_col] == 1).sum()) if label_col in df.columns else 0
        fake_count = int((df[label_col] == 0).sum()) if label_col in df.columns else 0
        real_ratio = round(real_count / count, 4) if count > 0 else 0.0
        fake_ratio = round(fake_count / count, 4) if count > 0 else 0.0
        return {
            f"{name}_count": count,
            f"{name}_pct": pct,
            f"{name}_real_count": real_count,
            f"{name}_fake_count": fake_count,
            f"{name}_real_ratio": real_ratio,
            f"{name}_fake_ratio": fake_ratio,
        }

    stats = {"total_records": total}
    stats.update(_stats(splits.train, "train"))
    stats.update(_stats(splits.val, "val"))
    stats.update(_stats(splits.test, "test"))

    # Add leakage audit
    leakage = verify_no_data_leakage(splits.train, splits.val, splits.test)
    stats["has_leakage"] = leakage["has_leakage"]

    return stats


def save_splits_to_csv(
    splits: SplitResult,
    output_dir: Path | str,
    train_filename: str = "train.csv",
    val_filename: str = "val.csv",
    test_filename: str = "test.csv",
) -> dict[str, Path]:
    """Save the train, validation, and test splits into CSV files.

    Args:
        splits: SplitResult containing the three partitions.
        output_dir: Directory where CSV files should be written.
        train_filename: Filename for the training set.
        val_filename: Filename for the validation set.
        test_filename: Filename for the test set.

    Returns:
        Dictionary mapping partition names to saved Path objects.
    """
    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    paths = {
        "train": out_path / train_filename,
        "val": out_path / val_filename,
        "test": out_path / test_filename,
    }

    splits.train.to_csv(paths["train"], index=False)
    splits.val.to_csv(paths["val"], index=False)
    splits.test.to_csv(paths["test"], index=False)

    return paths
