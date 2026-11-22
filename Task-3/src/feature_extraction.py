"""TF-IDF Feature Extraction Module for Task 3: AI-Based Fake News Detection.

Provides reproducible, modular TF-IDF text representation using scikit-learn's
TfidfVectorizer with strict data-leakage protection.

Core Architectural Principles:
1. Strict Training-Set Isolation:
   - Vectorizer is fitted strictly on Training data (`fit_transform` or `fit`).
   - Validation and Test sets are transformed using `.transform()` only.
   - Vocabulary is never updated or influenced by holdout data.
2. Negation Preservation:
   - English stopwords are pruned to remove common noise while explicitly protecting
     negation words ('not', 'no', 'never', 'neither', 'nor', 'cannot', 'without').
3. Sensible Baseline Configuration:
   - max_features: 5,000 (balances rich representation with memory efficiency and speed)
   - ngram_range: (1, 2) (captures unigrams and key bigrams like 'fake news', 'white house', 'did not')
   - min_df: 2 (discards unique single-occurrence typos)
   - max_df: 0.95 (ignores corpus-wide ubiquitous words appearing in >95% of documents)
   - sublinear_tf: True (uses 1 + log(tf) scaling to dampen repetitive term frequency spikes)
4. Robust Inference:
   - Gracefully handles unseen words, empty strings, and single-document inference without crashing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import pandas as pd
from scipy.sparse import csr_matrix, issparse
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

logger = logging.getLogger(__name__)

# Key negation words that carry critical semantic meaning in misinformation analysis
CRITICAL_NEGATION_WORDS: frozenset[str] = frozenset(
    {"not", "no", "never", "neither", "nor", "none", "cannot", "without", "nothing", "nowhere"}
)


def get_stop_words(preserve_negations: bool = True) -> list[str]:
    """Retrieve an English stop words list with optional negation protection.

    Standard scikit-learn 'english' stop words include words like 'not', 'no', and 'never'.
    In sentiment and fake news classification, removing negations inverts meaning
    (e.g., 'not true' becoming 'true'). This helper removes negations from the stop list.

    Args:
        preserve_negations: If True, removes negation words from the stopword set.

    Returns:
        Sorted list of stop words.
    """
    if preserve_negations:
        custom_set = set(ENGLISH_STOP_WORDS) - CRITICAL_NEGATION_WORDS
        return sorted(custom_set)
    return sorted(ENGLISH_STOP_WORDS)


def build_tfidf_vectorizer(
    max_features: int | None = 5000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int | float = 2,
    max_df: int | float = 0.95,
    sublinear_tf: bool = True,
    preserve_negations: bool = True,
    lowercase: bool = True,
) -> TfidfVectorizer:
    """Instantiate a configured TfidfVectorizer with sensible NLP defaults.

    Args:
        max_features: Maximum vocabulary size (None for unlimited).
        ngram_range: Lower and upper boundary of n-gram values (default (1, 2)).
        min_df: Minimum document frequency threshold (default 2).
        max_df: Maximum document frequency threshold ratio (default 0.95).
        sublinear_tf: Apply sublinear tf scaling: 1 + log(tf) (default True).
        preserve_negations: Whether to preserve negation words when filtering stop words.
        lowercase: Whether to convert text to lowercase during tokenization.

    Returns:
        Unfitted TfidfVectorizer instance.
    """
    stop_words_list = get_stop_words(preserve_negations=preserve_negations)

    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        stop_words=stop_words_list,
        lowercase=lowercase,
    )


class TFIDFExtractor:
    """Modular wrapper for managing TF-IDF feature extraction pipelines.

    Guarantees that feature fitting occurs strictly on training partitions,
    and holdout/inference partitions are transformed without vocabulary leakage.
    """

    def __init__(
        self,
        max_features: int | None = 5000,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int | float = 2,
        max_df: int | float = 0.95,
        sublinear_tf: bool = True,
        preserve_negations: bool = True,
        lowercase: bool = True,
    ) -> None:
        """Initialize the extractor with vectorization settings."""
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.sublinear_tf = sublinear_tf
        self.preserve_negations = preserve_negations
        self.lowercase = lowercase

        self.vectorizer: TfidfVectorizer = build_tfidf_vectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
            preserve_negations=preserve_negations,
            lowercase=lowercase,
        )
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        """Check whether the vectorizer has been fitted on training data."""
        return self._is_fitted

    @property
    def vocabulary_size(self) -> int:
        """Get the number of features in the fitted vocabulary."""
        if not self._is_fitted:
            return 0
        return len(self.vectorizer.vocabulary_)

    def fit(self, train_texts: Sequence[str] | pd.Series | Iterable[str]) -> TFIDFExtractor:
        """Fit the vectorizer vocabulary strictly on training text documents.

        Args:
            train_texts: Sequence of training text documents.

        Returns:
            self (fitted extractor instance).

        Raises:
            ValueError: If train_texts is empty or contains no valid documents.
        """
        texts = self._validate_and_sanitize(train_texts)
        if len(texts) == 0:
            raise ValueError("Cannot fit TFIDFExtractor on an empty text collection.")

        self.vectorizer.fit(texts)
        self._is_fitted = True
        return self

    def transform(self, texts: Sequence[str] | pd.Series | Iterable[str]) -> csr_matrix:
        """Transform text documents into a TF-IDF sparse matrix using the fitted vocabulary.

        Args:
            texts: Sequence of documents to transform.

        Returns:
            scipy.sparse.csr_matrix containing TF-IDF feature vectors.

        Raises:
            ValueError: If called before the vectorizer has been fitted.
        """
        if not self._is_fitted:
            raise ValueError(
                "TFIDFExtractor has not been fitted yet. Call fit() on training data "
                "before attempting to transform validation, test, or inference data."
            )

        sanitized_texts = self._validate_and_sanitize(texts)
        return self.vectorizer.transform(sanitized_texts)

    def fit_transform(self, train_texts: Sequence[str] | pd.Series | Iterable[str]) -> csr_matrix:
        """Fit the vocabulary and transform training documents in a single step.

        Args:
            train_texts: Sequence of training documents.

        Returns:
            scipy.sparse.csr_matrix of training feature vectors.
        """
        texts = self._validate_and_sanitize(train_texts)
        if len(texts) == 0:
            raise ValueError("Cannot fit_transform on an empty text collection.")

        X = self.vectorizer.fit_transform(texts)
        self._is_fitted = True
        return X

    def transform_single(self, text: str) -> csr_matrix:
        """Transform a single text string into a 2D sparse vector for inference.

        Args:
            text: Single document string.

        Returns:
            Sparse matrix with shape (1, n_features).
        """
        return self.transform([text])

    def get_feature_names(self) -> list[str]:
        """Return the list of vocabulary tokens ordered by feature index."""
        if not self._is_fitted:
            raise ValueError("TFIDFExtractor is not fitted yet.")
        return list(self.vectorizer.get_feature_names_out())

    def save(self, filepath: Path | str) -> Path:
        """Serialize the fitted extractor to disk using joblib.

        Args:
            filepath: Destination file path.

        Returns:
            Resolved Path object of the saved file.
        """
        path_obj = Path(filepath).resolve()
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path_obj)
        return path_obj

    @classmethod
    def load(cls, filepath: Path | str) -> TFIDFExtractor:
        """Load a serialized TFIDFExtractor instance from disk.

        Args:
            filepath: Path to the serialized extractor file.

        Returns:
            Loaded TFIDFExtractor instance.
        """
        path_obj = Path(filepath).resolve()
        if not path_obj.exists():
            raise FileNotFoundError(f"Extractor file not found at: '{path_obj}'")
        instance = joblib.load(path_obj)
        if not isinstance(instance, cls):
            raise TypeError(f"Loaded object is of type {type(instance)}, expected {cls.__name__}")
        return instance

    @staticmethod
    def _validate_and_sanitize(texts: Any) -> list[str]:
        """Convert input series or collections into sanitized strings, handling NaNs."""
        if texts is None:
            return []
        if isinstance(texts, pd.Series):
            return texts.fillna("").astype(str).tolist()
        return [str(t) if t is not None else "" for t in texts]
