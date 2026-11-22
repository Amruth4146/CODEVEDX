"""Unit tests for data_loader.py in Task-3."""

from pathlib import Path
import pandas as pd
import pytest

from src.data_loader import (
    clean_dataset,
    combine_title_and_text,
    get_dataset_statistics,
    load_and_prepare_dataset,
    load_single_csv,
    remove_duplicates,
    remove_empty_records,
)


@pytest.fixture
def sample_csv_dir(tmp_path: Path) -> Path:
    """Fixture creating small synthetic True.csv and Fake.csv files."""
    true_data = pd.DataFrame(
        {
            "title": [
                "U.S. Senate passes bipartisan budget bill",
                "Diplomats meet in Geneva for climate summit",
                "U.S. Senate passes bipartisan budget bill",  # Exact duplicate title/text
                "Title only real article",
                "",  # Empty title
            ],
            "text": [
                "WASHINGTON (Reuters) - Lawmakers voted 65-35 to pass the funding bill.",
                "GENEVA (Reuters) - Global delegates discussed emission targets.",
                "WASHINGTON (Reuters) - Lawmakers voted 65-35 to pass the funding bill.",
                "",  # Empty text (title only)
                "Body only real article text without a title.",  # Body only
            ],
            "subject": ["politicsNews", "worldnews", "politicsNews", "politicsNews", "worldnews"],
            "date": ["December 31, 2017", "December 30, 2017", "December 31, 2017", "December 29, 2017", "December 28, 2017"],
        }
    )

    fake_data = pd.DataFrame(
        {
            "title": [
                "SHOCKING: Secret memo reveals election plot!",
                "WATCH: Mayor caught on tape in scandal [VIDEO]",
                "SHOCKING: Secret memo reveals election plot!",  # Duplicate
                "   ",  # Whitespace title
                "Another fake story with text body",
            ],
            "text": [
                "An anonymous blogger reported a fabricated story about officials.",
                "   ",  # Whitespace-only text (multimedia post simulation)
                "An anonymous blogger reported a fabricated story about officials.",
                "   ",  # Completely empty record
                "Details about a fabricated controversy.",
            ],
            "subject": ["News", "politics", "News", "Government News", "left-news"],
            "date": ["December 31, 2017", "December 30, 2017", "December 31, 2017", "December 29, 2017", "December 28, 2017"],
        }
    )

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    true_data.to_csv(raw_dir / "True.csv", index=False)
    fake_data.to_csv(raw_dir / "Fake.csv", index=False)

    return raw_dir


class TestSingleCSVLoading:
    """Tests for load_single_csv."""

    def test_load_valid_csv(self, sample_csv_dir: Path):
        df = load_single_csv(sample_csv_dir / "True.csv")
        assert len(df) == 5
        assert "title" in df.columns
        assert "text" in df.columns

    def test_missing_file_raises_error(self, tmp_path: Path):
        missing_file = tmp_path / "NonExistent.csv"
        with pytest.raises(FileNotFoundError, match="Dataset file not found"):
            load_single_csv(missing_file)

    def test_empty_file_raises_error(self, tmp_path: Path):
        empty_file = tmp_path / "empty.csv"
        empty_file.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="empty"):
            load_single_csv(empty_file)

    def test_missing_required_columns_raises_error(self, tmp_path: Path):
        bad_csv = tmp_path / "bad_columns.csv"
        pd.DataFrame({"headline": ["News"], "body": ["Content"]}).to_csv(bad_csv, index=False)
        with pytest.raises(ValueError, match="missing required column"):
            load_single_csv(bad_csv)


class TestContentCombination:
    """Tests for combining title and text."""

    def test_combine_both_title_and_text(self):
        df = pd.DataFrame({"title": ["Big Headline"], "text": ["Detailed story body."]})
        combined = combine_title_and_text(df)
        assert combined["content"].iloc[0] == "Big Headline Detailed story body."
        assert combined["title"].iloc[0] == "Big Headline"
        assert combined["text"].iloc[0] == "Detailed story body."

    def test_title_only_article(self):
        df = pd.DataFrame({"title": ["Breaking News Headline Only"], "text": [""]})
        combined = combine_title_and_text(df)
        assert combined["content"].iloc[0] == "Breaking News Headline Only"

    def test_body_only_article(self):
        df = pd.DataFrame({"title": [""], "text": ["Story body text with no headline."]})
        combined = combine_title_and_text(df)
        assert combined["content"].iloc[0] == "Story body text with no headline."

    def test_handles_nan_values_gracefully(self):
        df = pd.DataFrame({"title": [None], "text": ["Valid body text"]})
        combined = combine_title_and_text(df)
        assert combined["content"].iloc[0] == "Valid body text"


class TestDataCleaningAndDeduplication:
    """Tests for clean_dataset, empty removal, and duplicate removal."""

    def test_remove_empty_records(self):
        df = pd.DataFrame({"content": ["Valid content", "   ", "", "Short"]})
        filtered = remove_empty_records(df, min_length=2)
        assert len(filtered) == 2
        assert "Valid content" in filtered["content"].values
        assert "Short" in filtered["content"].values

    def test_remove_duplicates_exact(self):
        df = pd.DataFrame({"content": ["Story A", "Story B", "Story A", "Story C"]})
        deduped = remove_duplicates(df)
        assert len(deduped) == 3
        assert list(deduped["content"]) == ["Story A", "Story B", "Story C"]

    def test_remove_duplicates_whitespace_normalized(self):
        df = pd.DataFrame(
            {
                "content": [
                    "President signs   the new  law.",
                    "President signs the new law.",  # Duplicate with different whitespace
                    "Different story entirely.",
                ]
            }
        )
        deduped = remove_duplicates(df)
        assert len(deduped) == 2
        assert len(deduped[deduped["content"].str.contains("President")]) == 1


class TestLoadAndPrepareDataset:
    """Tests for end-to-end load_and_prepare_dataset."""

    def test_end_to_end_loading_and_labels(self, sample_csv_dir: Path):
        df = load_and_prepare_dataset(
            raw_dir=sample_csv_dir,
            clean=True,
            drop_duplicates=True,
            exclude_metadata=True,
        )

        # Verify labels
        assert set(df["label"].unique()) == {0, 1}
        # Real news has label 1, Fake has label 0
        assert (df[df["label"] == 1]["content"].str.contains("Senate|Diplomats|real")).any()
        assert (df[df["label"] == 0]["content"].str.contains("SHOCKING|WATCH|fake")).any()

        # Verify excluded metadata
        assert "subject" not in df.columns
        assert "date" not in df.columns

        # Verify required columns exist
        assert "content" in df.columns
        assert "title" in df.columns
        assert "text" in df.columns
        assert "label" in df.columns

        # Verify no empty content exists
        assert (df["content"].str.strip() == "").sum() == 0

        # Verify duplicates are dropped
        assert not df["content"].duplicated().any()

    def test_metadata_retained_when_exclude_metadata_false(self, sample_csv_dir: Path):
        df = load_and_prepare_dataset(
            raw_dir=sample_csv_dir,
            exclude_metadata=False,
        )
        assert "subject" in df.columns
        assert "date" in df.columns


class TestDatasetStatistics:
    """Tests for get_dataset_statistics."""

    def test_statistics_calculation(self):
        df = pd.DataFrame(
            {
                "content": ["Article one", "Article two text", "Article three longer text"],
                "label": [1, 1, 0],
            }
        )
        stats = get_dataset_statistics(df)
        assert stats["total_records"] == 3
        assert stats["real_records"] == 2
        assert stats["fake_records"] == 1
        assert stats["real_ratio"] == 0.6667
        assert stats["fake_ratio"] == 0.3333
        assert stats["empty_content_count"] == 0
        assert stats["mean_content_length"] > 0

    def test_empty_dataframe_statistics(self):
        df = pd.DataFrame(columns=["content", "label"])
        stats = get_dataset_statistics(df)
        assert stats["total_records"] == 0
        assert stats["real_records"] == 0
        assert stats["fake_records"] == 0
