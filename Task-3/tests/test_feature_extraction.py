"""Unit tests for feature_extraction.py in Task-3."""

from pathlib import Path
import pytest
from scipy.sparse import issparse

from src.feature_extraction import (
    TFIDFExtractor,
    build_tfidf_vectorizer,
    get_stop_words,
)


@pytest.fixture
def sample_training_texts() -> list[str]:
    """Sample corpus of training documents."""
    return [
        "President signs the federal funding bill into law after Senate vote.",
        "Breaking news: Congress passed economic reform legislation today.",
        "White House officials held a press briefing regarding international trade.",
        "Lawmakers debated tax reform and healthcare policies on Capitol Hill.",
        "The committee issued a formal report detailing budget expenditures.",
    ]


@pytest.fixture
def sample_holdout_texts() -> list[str]:
    """Sample validation or test documents containing unseen vocabulary."""
    return [
        "Unprecedented extraterrestrial spacecraft landing detected on Jupiter orbit.",
        "President addressed the public regarding new federal initiatives.",
    ]


class TestTFIDFConfigurationAndStopwords:
    """Tests for vectorizer configuration and negation protection."""

    def test_negations_preserved_in_stopwords(self):
        stops = get_stop_words(preserve_negations=True)
        # Negations must NOT be in the stop words list
        assert "not" not in stops
        assert "no" not in stops
        assert "never" not in stops
        assert "cannot" not in stops
        assert "neither" not in stops
        assert "nor" not in stops

        # Common stopwords should still be present
        assert "the" in stops
        assert "and" in stops
        assert "is" in stops

    def test_build_vectorizer_defaults(self):
        vec = build_tfidf_vectorizer()
        assert vec.max_features == 5000
        assert vec.ngram_range == (1, 2)
        assert vec.sublinear_tf is True


class TestTFIDFExtractorFittingAndTransform:
    """Tests for TFIDFExtractor lifecycle, transformation, and shapes."""

    def test_fit_and_transform_pipeline(self, sample_training_texts: list[str]):
        extractor = TFIDFExtractor(max_features=50, min_df=1)
        assert extractor.is_fitted is False
        assert extractor.vocabulary_size == 0

        # Fit strictly on train
        extractor.fit(sample_training_texts)
        assert extractor.is_fitted is True
        assert extractor.vocabulary_size > 0

        # Transform train
        X_train = extractor.transform(sample_training_texts)
        assert issparse(X_train)
        assert X_train.shape[0] == len(sample_training_texts)
        assert X_train.shape[1] == extractor.vocabulary_size

    def test_fit_transform_matches_fit_then_transform(self, sample_training_texts: list[str]):
        ext_1 = TFIDFExtractor(max_features=50, min_df=1)
        ext_2 = TFIDFExtractor(max_features=50, min_df=1)

        X_1 = ext_1.fit_transform(sample_training_texts)
        ext_2.fit(sample_training_texts)
        X_2 = ext_2.transform(sample_training_texts)

        assert (X_1 != X_2).nnz == 0  # Sparse matrix equality

    def test_transform_unseen_words_without_crashing(
        self, sample_training_texts: list[str], sample_holdout_texts: list[str]
    ):
        extractor = TFIDFExtractor(max_features=50, min_df=1)
        extractor.fit(sample_training_texts)

        # Transform holdout texts containing unseen words ('spacecraft', 'extraterrestrial')
        X_holdout = extractor.transform(sample_holdout_texts)
        assert issparse(X_holdout)
        assert X_holdout.shape == (len(sample_holdout_texts), extractor.vocabulary_size)

    def test_transform_single_text(self, sample_training_texts: list[str]):
        extractor = TFIDFExtractor(max_features=50, min_df=1)
        extractor.fit(sample_training_texts)

        vec_single = extractor.transform_single("White House press conference today.")
        assert issparse(vec_single)
        assert vec_single.shape == (1, extractor.vocabulary_size)

    def test_empty_string_handling(self, sample_training_texts: list[str]):
        extractor = TFIDFExtractor(max_features=50, min_df=1)
        extractor.fit(sample_training_texts)

        # Empty string should yield an all-zero sparse vector
        X_empty = extractor.transform(["", "   "])
        assert issparse(X_empty)
        assert X_empty.shape == (2, extractor.vocabulary_size)
        assert X_empty.nnz == 0  # No non-zero elements


class TestDataLeakageProtection:
    """Strict tests proving no vocabulary leakage occurs from holdout data."""

    def test_no_vocabulary_leakage_from_validation_or_test(self):
        train_texts = [
            "Federal budget resolution passed by Senate committee.",
            "Treasury department published annual tax figures.",
        ]
        val_texts = [
            "Extraterrestrial Martian astrobiology laboratory discovered organic molecules.",
        ]

        extractor = TFIDFExtractor(max_features=100, min_df=1)
        # Fitted strictly on train
        extractor.fit(train_texts)

        train_vocab = extractor.get_feature_names()

        # Transform validation data
        X_val = extractor.transform(val_texts)

        # Assert words exclusive to validation set are NOT in the vocabulary
        assert "extraterrestrial" not in train_vocab
        assert "martian" not in train_vocab
        assert "astrobiology" not in train_vocab

        # Vocabulary size must remain identical before and after holdout transform
        vocab_size_before = extractor.vocabulary_size
        extractor.transform(val_texts)
        assert extractor.vocabulary_size == vocab_size_before

    def test_transform_before_fit_raises_error(self):
        extractor = TFIDFExtractor()
        with pytest.raises(ValueError, match="not been fitted yet"):
            extractor.transform(["Sample sentence"])

    def test_get_feature_names_before_fit_raises_error(self):
        extractor = TFIDFExtractor()
        with pytest.raises(ValueError, match="not fitted yet"):
            extractor.get_feature_names()

    def test_fit_on_empty_collection_raises_error(self):
        extractor = TFIDFExtractor()
        with pytest.raises(ValueError, match="empty text collection"):
            extractor.fit([])


class TestSerialization:
    """Tests for joblib saving and loading of the fitted extractor."""

    def test_save_and_load_extractor(
        self, sample_training_texts: list[str], tmp_path: Path
    ):
        extractor = TFIDFExtractor(max_features=50, min_df=1)
        extractor.fit(sample_training_texts)

        model_path = tmp_path / "tfidf_extractor.joblib"
        extractor.save(model_path)
        assert model_path.exists()

        loaded_extractor = TFIDFExtractor.load(model_path)
        assert loaded_extractor.is_fitted is True
        assert loaded_extractor.vocabulary_size == extractor.vocabulary_size
        assert loaded_extractor.get_feature_names() == extractor.get_feature_names()

        # Check transformed vectors match
        orig_vec = extractor.transform(["President signed the bill"])
        loaded_vec = loaded_extractor.transform(["President signed the bill"])
        assert (orig_vec != loaded_vec).nnz == 0
