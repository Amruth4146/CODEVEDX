"""Inference Engine and Prediction API for Task 3: AI-Based Fake News Detection.

Provides a clean, robust, and validated interface for predicting whether news articles
resemble Fake or Real news patterns according to the learned dataset models.

Core Capabilities:
1. Model Ingestion: Loads the trained champion pipeline (Logistic Regression) without retraining.
2. In-Memory Caching: Reuses the loaded pipeline across calls to prevent repeated disk I/O.
3. Strict Input Validation: Rejects None, non-string types, empty/whitespace strings, and overly short text.
4. Structured Predictions: Returns rich PredictionResult objects with class names, numeric labels,
   confidence scores, class probability distributions, and educational disclaimers.
5. Batch Inference: Efficiently evaluates sequences of articles in a single vectorized pass.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
from pathlib import Path
from typing import Any, Sequence

from src.model_utils import (
    DEFAULT_LABEL_MAPPING,
    FakeNewsPipeline,
    get_default_models_dir,
    load_model_pipeline,
)

logger = logging.getLogger(__name__)

# Default champion model filename inside Task-3/models/
CHAMPION_MODEL_FILENAME = "fake_news_model.joblib"

# Input validation constraints
MIN_WORDS_REQUIRED = 3
MIN_CHARS_REQUIRED = 10
MAX_CHARS_ALLOWED = 100_000

EDUCATIONAL_DISCLAIMER = (
    "Educational Notice: This prediction indicates whether the input text resembles patterns "
    "observed in the training dataset. It does not constitute independent factual verification."
)


@dataclass
class PredictionResult:
    """Structured output container for news article classification."""

    label: int
    label_name: str
    confidence: float
    fake_probability: float
    real_probability: float
    model_name: str
    cleaned_preview: str
    disclaimer: str = EDUCATIONAL_DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        """Convert result object into a standard dictionary."""
        return asdict(self)


def validate_news_input(
    text: Any,
    min_words: int = MIN_WORDS_REQUIRED,
    min_chars: int = MIN_CHARS_REQUIRED,
    max_chars: int = MAX_CHARS_ALLOWED,
) -> str:
    """Validate and sanitize single news text input.

    Args:
        text: Input object to validate.
        min_words: Minimum number of whitespace-delimited words required.
        min_chars: Minimum character length required (excluding edge whitespace).
        max_chars: Maximum character length allowed before truncation.

    Returns:
        Sanitized, non-empty string.

    Raises:
        TypeError: If input is not a string.
        ValueError: If input is empty, whitespace-only, or too short.
    """
    if text is None:
        raise TypeError("Input text cannot be None. Please provide a valid news string.")

    if not isinstance(text, str):
        raise TypeError(
            f"Invalid input type: expected str, got {type(text).__name__}. "
            f"Please provide news content as text."
        )

    stripped = text.strip()

    if not stripped:
        raise ValueError(
            "Input text cannot be empty or contain only whitespace. "
            "Please provide meaningful news headline or article text."
        )

    char_len = len(stripped)
    word_count = len(stripped.split())

    if char_len < min_chars or word_count < min_words:
        raise ValueError(
            f"Input text is too short to evaluate reliably as news. "
            f"Found {word_count} word(s) and {char_len} character(s); "
            f"minimum required is {min_words} words and {min_chars} characters."
        )

    if char_len > max_chars:
        logger.warning(
            "Input text length (%d chars) exceeds maximum threshold (%d chars). Truncating.",
            char_len,
            max_chars,
        )
        stripped = stripped[:max_chars]

    return stripped


class FakeNewsPredictor:
    """Inference engine for Fake News Detection.

    Loads and holds a pre-trained FakeNewsPipeline in memory to serve fast,
    repeatable predictions with calibrated confidence scores.
    """

    def __init__(
        self,
        model_path: Path | str | None = None,
        pipeline: FakeNewsPipeline | None = None,
    ) -> None:
        """Initialize the predictor.

        Args:
            model_path: Path to the serialized .joblib model file.
                        If None and pipeline is None, defaults to Task-3/models/fake_news_model.joblib.
            pipeline: Pre-loaded FakeNewsPipeline instance (useful for testing or direct injection).
        """
        if pipeline is not None:
            self._pipeline = pipeline
            self._model_path = None
        else:
            if model_path is None:
                self._model_path = get_default_models_dir() / CHAMPION_MODEL_FILENAME
            else:
                self._model_path = Path(model_path).resolve()
            self._pipeline = self._load_model()

    def _load_model(self) -> FakeNewsPipeline:
        """Load the model pipeline from disk with defensive error handling."""
        if self._model_path is None:
            raise ValueError("No model path specified.")

        logger.info("Loading champion model pipeline from '%s'...", self._model_path)
        try:
            return load_model_pipeline(self._model_path)
        except FileNotFoundError as err:
            raise FileNotFoundError(
                f"Champion model not found at '{self._model_path}'. "
                f"Please ensure Phase 5.3 model training has been executed."
            ) from err
        except Exception as err:
            raise ValueError(
                f"Failed to load champion model from '{self._model_path}': {err}"
            ) from err

    @property
    def model_name(self) -> str:
        """Return the name of the underlying classifier model."""
        return self._pipeline.model_name

    @property
    def feature_count(self) -> int:
        """Return the number of TF-IDF vocabulary features."""
        return self._pipeline.feature_extractor.vocabulary_size

    @property
    def pipeline(self) -> FakeNewsPipeline:
        """Access the underlying FakeNewsPipeline instance."""
        return self._pipeline

    def predict(
        self,
        text: Any,
        min_words: int = MIN_WORDS_REQUIRED,
        min_chars: int = MIN_CHARS_REQUIRED,
    ) -> PredictionResult:
        """Classify a single news article or headline.

        Args:
            text: Raw input news text or headline.
            min_words: Minimum words required.
            min_chars: Minimum characters required.

        Returns:
            PredictionResult object.

        Raises:
            TypeError: If text is None or not a string.
            ValueError: If text is empty or too short.
        """
        valid_text = validate_news_input(text, min_words=min_words, min_chars=min_chars)
        batch_results = self.predict_batch([valid_text], validate=False)
        return batch_results[0]

    def predict_batch(
        self,
        texts: Sequence[Any],
        validate: bool = True,
    ) -> list[PredictionResult]:
        """Classify a sequence of news articles in a single vectorized pass.

        Args:
            texts: Sequence of text strings to predict.
            validate: Whether to validate each text before inference.

        Returns:
            List of PredictionResult objects matching the input order.
        """
        if not hasattr(texts, "__iter__") or isinstance(texts, (str, bytes)):
            raise TypeError(
                f"predict_batch expects an iterable sequence of strings, got {type(texts).__name__}."
            )

        sanitized_texts: list[str] = []
        if validate:
            for idx, item in enumerate(texts):
                try:
                    clean_str = validate_news_input(item)
                    sanitized_texts.append(clean_str)
                except (TypeError, ValueError) as err:
                    raise ValueError(f"Validation failed for document at index {idx}: {err}") from err
        else:
            sanitized_texts = [str(t) for t in texts]

        if not sanitized_texts:
            return []

        # Generate probabilities and predictions via pipeline
        probas = self._pipeline.predict_proba(sanitized_texts)
        preds = self._pipeline.predict(sanitized_texts)

        results: list[PredictionResult] = []
        for text_str, pred, proba in zip(sanitized_texts, preds, probas):
            pred_int = int(pred)
            label_name = self._pipeline.label_mapping.get(pred_int, "Unknown")
            prob_fake = float(proba[0])
            prob_real = float(proba[1])
            confidence = float(max(prob_fake, prob_real))

            preview = text_str[:120].strip() + ("..." if len(text_str) > 120 else "")

            results.append(
                PredictionResult(
                    label=pred_int,
                    label_name=label_name,
                    confidence=round(confidence, 4),
                    fake_probability=round(prob_fake, 4),
                    real_probability=round(prob_real, 4),
                    model_name=self.model_name,
                    cleaned_preview=preview,
                    disclaimer=EDUCATIONAL_DISCLAIMER,
                )
            )

        return results


# Module-level cached predictor instance
_CACHED_PREDICTOR: FakeNewsPredictor | None = None


def get_default_predictor(model_path: Path | str | None = None) -> FakeNewsPredictor:
    """Retrieve or create the singleton cached FakeNewsPredictor instance.

    Avoids reloading model weights from disk on every prediction call.

    Args:
        model_path: Optional custom model path. If provided and different from the
                    current cached model, resets and loads from the new path.

    Returns:
        Cached FakeNewsPredictor instance.
    """
    global _CACHED_PREDICTOR

    if _CACHED_PREDICTOR is None:
        _CACHED_PREDICTOR = FakeNewsPredictor(model_path=model_path)
    elif model_path is not None and Path(model_path).resolve() != _CACHED_PREDICTOR._model_path:
        _CACHED_PREDICTOR = FakeNewsPredictor(model_path=model_path)

    return _CACHED_PREDICTOR


def reset_default_predictor() -> None:
    """Clear the cached singleton predictor (useful during test teardown)."""
    global _CACHED_PREDICTOR
    _CACHED_PREDICTOR = None


def predict_news(
    text: Any,
    model_path: Path | str | None = None,
) -> PredictionResult:
    """Convenience function to classify a single news text using the default predictor.

    Args:
        text: Raw news text or headline.
        model_path: Optional path to custom model artifact.

    Returns:
        PredictionResult object.
    """
    predictor = get_default_predictor(model_path=model_path)
    return predictor.predict(text)


def predict_news_batch(
    texts: Sequence[Any],
    model_path: Path | str | None = None,
) -> list[PredictionResult]:
    """Convenience function to classify a batch of news texts using the default predictor.

    Args:
        texts: Sequence of news strings.
        model_path: Optional path to custom model artifact.

    Returns:
        List of PredictionResult objects.
    """
    predictor = get_default_predictor(model_path=model_path)
    return predictor.predict_batch(texts)
