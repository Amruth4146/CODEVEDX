"""Unit tests for data_splitter.py in Task-3."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data_splitter import (
    SplitResult,
    get_split_statistics,
    save_splits_to_csv,
    split_dataset,
    verify_no_data_leakage,
)


@pytest.fixture
def synthetic_classification_df() -> pd.DataFrame:
    """Fixture creating a synthetic DataFrame with balanced classes."""
    n_samples = 100
    # 60 Real (1) and 40 Fake (0)
    labels = [1] * 60 + [0] * 40
    contents = [f"Article content sample number {i} for classification" for i in range(n_samples)]
    return pd.DataFrame({"content": contents, "label": labels})


class TestDataSplitter:
    """Tests for dataset splitting and stratification."""

    def test_correct_proportions(self, synthetic_classification_df: pd.DataFrame):
        splits = split_dataset(
            synthetic_classification_df,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            random_state=42,
        )

        assert len(splits.train) == 80
        assert len(splits.val) == 10
        assert len(splits.test) == 10
        assert len(splits.train) + len(splits.val) + len(splits.test) == 100

    def test_stratification_preserves_class_ratios(self, synthetic_classification_df: pd.DataFrame):
        splits = split_dataset(synthetic_classification_df, random_state=42)

        # Overall ratio: 60% Real (1), 40% Fake (0)
        # Train: 80 samples -> 48 Real (60%), 32 Fake (40%)
        assert (splits.train["label"] == 1).sum() == 48
        assert (splits.train["label"] == 0).sum() == 32

        # Val: 10 samples -> 6 Real (60%), 4 Fake (40%)
        assert (splits.val["label"] == 1).sum() == 6
        assert (splits.val["label"] == 0).sum() == 4

        # Test: 10 samples -> 6 Real (60%), 4 Fake (40%)
        assert (splits.test["label"] == 1).sum() == 6
        assert (splits.test["label"] == 0).sum() == 4

    def test_reproducibility_with_same_seed(self, synthetic_classification_df: pd.DataFrame):
        splits_1 = split_dataset(synthetic_classification_df, random_state=42)
        splits_2 = split_dataset(synthetic_classification_df, random_state=42)

        pd.testing.assert_frame_equal(splits_1.train, splits_2.train)
        pd.testing.assert_frame_equal(splits_1.val, splits_2.val)
        pd.testing.assert_frame_equal(splits_1.test, splits_2.test)

    def test_different_seeds_produce_different_splits(self, synthetic_classification_df: pd.DataFrame):
        splits_1 = split_dataset(synthetic_classification_df, random_state=42)
        splits_2 = split_dataset(synthetic_classification_df, random_state=999)

        # Content ordering should differ
        assert not splits_1.train["content"].equals(splits_2.train["content"])

    def test_split_result_unpacking(self, synthetic_classification_df: pd.DataFrame):
        train, val, test = split_dataset(synthetic_classification_df, random_state=42)
        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10

    def test_empty_dataframe_raises_error(self):
        empty_df = pd.DataFrame(columns=["content", "label"])
        with pytest.raises(ValueError, match="Cannot split an empty"):
            split_dataset(empty_df)

    def test_invalid_ratios_raise_error(self, synthetic_classification_df: pd.DataFrame):
        # Do not sum to 1.0
        with pytest.raises(ValueError, match="ratios must sum to 1.0"):
            split_dataset(synthetic_classification_df, train_ratio=0.7, val_ratio=0.1, test_ratio=0.1)

        # Negative ratio
        with pytest.raises(ValueError, match="must be strictly greater than 0"):
            split_dataset(synthetic_classification_df, train_ratio=0.9, val_ratio=-0.1, test_ratio=0.2)

    def test_missing_label_column_raises_error(self):
        bad_df = pd.DataFrame({"text": ["a", "b"], "target": [1, 0]})
        with pytest.raises(ValueError, match="Target column 'label' not found"):
            split_dataset(bad_df, label_col="label")

    def test_single_class_raises_error(self):
        single_class_df = pd.DataFrame({"content": ["a", "b", "c", "d"], "label": [1, 1, 1, 1]})
        with pytest.raises(ValueError, match="requires at least 2 target classes"):
            split_dataset(single_class_df)

    def test_too_few_samples_raises_error(self):
        sparse_df = pd.DataFrame({"content": ["a", "b", "c"], "label": [1, 0, 0]})
        with pytest.raises(ValueError, match="requires at least 4 samples per class"):
            split_dataset(sparse_df)


class TestLeakageAndStatistics:
    """Tests for leak verification and statistics reporting."""

    def test_verify_no_leakage_clean_split(self, synthetic_classification_df: pd.DataFrame):
        splits = split_dataset(synthetic_classification_df, random_state=42)
        leakage = verify_no_data_leakage(splits.train, splits.val, splits.test)

        assert leakage["has_leakage"] is False
        assert leakage["train_val_overlap"] == 0
        assert leakage["train_test_overlap"] == 0
        assert leakage["val_test_overlap"] == 0

    def test_verify_no_leakage_detects_overlap(self):
        train = pd.DataFrame({"content": ["Story A", "Story B"]})
        val = pd.DataFrame({"content": ["Story B", "Story C"]})  # Story B leaks into val
        test = pd.DataFrame({"content": ["Story D"]})

        leakage = verify_no_data_leakage(train, val, test)
        assert leakage["has_leakage"] is True
        assert leakage["train_val_overlap"] == 1

    def test_get_split_statistics(self, synthetic_classification_df: pd.DataFrame):
        splits = split_dataset(synthetic_classification_df, random_state=42)
        stats = get_split_statistics(splits)

        assert stats["total_records"] == 100
        assert stats["train_count"] == 80
        assert stats["val_count"] == 10
        assert stats["test_count"] == 10
        assert stats["train_pct"] == 80.0
        assert stats["val_pct"] == 10.0
        assert stats["test_pct"] == 10.0
        assert stats["has_leakage"] is False

    def test_save_splits_to_csv(self, synthetic_classification_df: pd.DataFrame, tmp_path: Path):
        splits = split_dataset(synthetic_classification_df, random_state=42)
        saved_paths = save_splits_to_csv(splits, tmp_path / "processed")

        assert saved_paths["train"].exists()
        assert saved_paths["val"].exists()
        assert saved_paths["test"].exists()

        loaded_train = pd.read_csv(saved_paths["train"])
        assert len(loaded_train) == 80
