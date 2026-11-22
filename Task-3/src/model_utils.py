"""Model Persistence, Inference Pipeline, and Confidence Utilities for Task 3.

Provides an encapsulated FakeNewsPipeline that bundles:
1. The trained classifier (e.g. LogisticRegression or MultinomialNB).
2. The fitted TFIDFExtractor (holding the vocabulary and IDF weights).
3. Preprocessing configuration ('standard', 'minimal', or 'aggressive').
4. Target label mapping ({1: 'Real', 0: 'Fake'}).
5. Validation evaluation metrics and metadata.

Supports end-to-end inference on raw strings or collections with confidence scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd

from src.feature_extraction import TFIDFExtractor
from src.preprocessing import clean_text_by_mode

logger = logging.getLogger(__name__)

DEFAULT_LABEL_MAPPING: dict[int, str] = {1: "Real", 0: "Fake"}


def get_default_models_dir() -> Path:
    """Return default Task-3/models directory path."""
    return Path(__file__).resolve().parent.parent / "models"


@dataclass
class FakeNewsPipeline:
    """Self-contained, deployable model pipeline for Fake News classification.

    Bundles text preprocessing, TF-IDF vectorization, classifier inference,
    and calibrated probability/confidence estimation.
    """

    classifier: Any
    feature_extractor: TFIDFExtractor
    model_name: str
    preprocessing_mode: str = "standard"
    label_mapping: dict[int, str] = field(default_factory=lambda: {1: "Real", 0: "Fake"})
    metrics: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def _preprocess_inputs(self, texts: str | Sequence[str] | pd.Series) -> list[str]:
        """Apply the pipeline's configured preprocessing to raw text inputs."""
        if isinstance(texts, str):
            text_list = [texts]
        elif isinstance(texts, pd.Series):
            text_list = texts.fillna("").astype(str).tolist()
        elif hasattr(texts, "__iter__"):
            text_list = [str(t) if t is not None else "" for t in texts]
        else:
            text_list = [str(texts)]

        return [clean_text_by_mode(t, mode=self.preprocessing_mode) for t in text_list]

    def predict(self, texts: str | Sequence[str] | pd.Series) -> np.ndarray:
        """Predict binary class labels (1 for Real, 0 for Fake).

        Args:
            texts: Single text string, list of strings, or pandas Series.

        Returns:
            NumPy array of integer class predictions.
        """
        cleaned = self._preprocess_inputs(texts)
        X = self.feature_extractor.transform(cleaned)
        return self.classifier.predict(X)

    def predict_proba(self, texts: str | Sequence[str] | pd.Series) -> np.ndarray:
        """Predict class probabilities [P(Fake), P(Real)].

        Args:
            texts: Single text string, list of strings, or pandas Series.

        Returns:
            2D NumPy array of shape (n_samples, 2) where:
            col 0 = P(Fake / class 0)
            col 1 = P(Real / class 1)
        """
        cleaned = self._preprocess_inputs(texts)
        X = self.feature_extractor.transform(cleaned)

        if not hasattr(self.classifier, "predict_proba"):
            raise AttributeError(
                f"Classifier '{self.model_name}' does not support predict_proba."
            )

        return self.classifier.predict_proba(X)

    def predict_with_confidence(
        self, texts: str | Sequence[str] | pd.Series
    ) -> list[dict[str, Any]]:
        """Generate structured predictions with confidence scores and disclaimer.

        For each document:
        - label: integer class (1 or 0)
        - label_name: string class name ('Real' or 'Fake')
        - confidence: probability of the predicted class, bounded in [0.0, 1.0]
        - probabilities: dict mapping class name to predicted probability
        - disclaimer: educational reminder that confidence represents model similarity,
          not real-world factual verification.

        Args:
            texts: Single text string or sequence of texts.

        Returns:
            List of prediction dictionaries.
        """
        probas = self.predict_proba(texts)
        preds = self.predict(texts)

        results: list[dict[str, Any]] = []
        for pred, proba in zip(preds, probas):
            pred_int = int(pred)
            label_name = self.label_mapping.get(pred_int, "Unknown")
            # Confidence is the probability of the predicted class
            confidence = float(np.max(proba))
            prob_fake = float(proba[0])
            prob_real = float(proba[1])

            results.append(
                {
                    "label": pred_int,
                    "label_name": label_name,
                    "confidence": round(confidence, 4),
                    "probabilities": {
                        "Fake": round(prob_fake, 4),
                        "Real": round(prob_real, 4),
                    },
                    "disclaimer": (
                        "Educational Notice: This prediction indicates whether the input "
                        "text resembles patterns in the training dataset. It does not "
                        "constitute independent factual verification."
                    ),
                }
            )

        return results


def save_model_pipeline(
    pipeline: FakeNewsPipeline,
    filepath: Path | str,
) -> Path:
    """Serialize a FakeNewsPipeline instance to disk using joblib.

    Args:
        pipeline: FakeNewsPipeline instance to save.
        filepath: Destination file path (e.g. Task-3/models/logistic_regression.joblib).

    Returns:
        Resolved Path object where the model was written.
    """
    path_obj = Path(filepath).resolve()
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    try:
        joblib.dump(pipeline, path_obj)
        logger.info("Saved model pipeline '%s' to '%s'", pipeline.model_name, path_obj)
    except Exception as err:
        raise IOError(f"Failed to serialize model pipeline to '{path_obj}': {err}") from err

    return path_obj


def load_model_pipeline(filepath: Path | str) -> FakeNewsPipeline:
    """Load and validate a serialized FakeNewsPipeline from disk.

    Performs validation:
    1. Checks if the file path exists.
    2. Validates that the file is not 0 bytes.
    3. Verifies that the deserialized object is a valid FakeNewsPipeline.

    Args:
        filepath: Path to the target model file.

    Returns:
        Loaded FakeNewsPipeline instance ready for inference.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file is empty, corrupted, or not a valid pipeline object.
    """
    path_obj = Path(filepath).resolve()

    if not path_obj.exists():
        raise FileNotFoundError(
            f"Model artifact not found at: '{path_obj}'. "
            f"Please verify that the model has been trained and saved."
        )

    if not path_obj.is_file():
        raise ValueError(f"Specified model path is not a file: '{path_obj}'")

    if path_obj.stat().st_size == 0:
        raise ValueError(f"Model artifact file is empty (0 bytes): '{path_obj}'")

    try:
        pipeline = joblib.load(path_obj)
    except Exception as err:
        raise ValueError(
            f"Corrupted or invalid model file at '{path_obj}'. Deserialization failed: {err}"
        ) from err

    if not isinstance(pipeline, FakeNewsPipeline):
        # Support fallback if saved as a dictionary
        if isinstance(pipeline, dict) and "classifier" in pipeline and "feature_extractor" in pipeline:
            pipeline = FakeNewsPipeline(
                classifier=pipeline["classifier"],
                feature_extractor=pipeline["feature_extractor"],
                model_name=pipeline.get("model_name", "Custom Classifier"),
                preprocessing_mode=pipeline.get("preprocessing_mode", "standard"),
                label_mapping=pipeline.get("label_mapping", DEFAULT_LABEL_MAPPING),
                metrics=pipeline.get("metrics", {}),
            )
        else:
            raise ValueError(
                f"Loaded object from '{path_obj}' is of type {type(pipeline)}, "
                f"expected FakeNewsPipeline."
            )

    return pipeline
