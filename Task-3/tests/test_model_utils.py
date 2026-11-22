"""Unit tests for model_utils.py in Task-3."""

from pathlib import Path
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.feature_extraction import TFIDFExtractor
from src.model_utils import (
    FakeNewsPipeline,
    load_model_pipeline,
    save_model_pipeline,
)


@pytest.fixture
def trained_pipeline() -> FakeNewsPipeline:
    """Fixture providing a small fitted FakeNewsPipeline for testing."""
    train_texts = [
        "President signed the federal budget into law after Senate vote.",
        "Congress passed tax reform legislation today in Washington.",
        "White House officials announced new diplomatic initiatives.",
        "SHOCKING: Secret memo exposes corrupt election conspiracy!",
        "WATCH: Unverified viral video claims scandalous plot revealed!",
        "BREAKING: Anonymous blog alleges fabricated foreign interference!",
    ]
    y_train = np.array([1, 1, 1, 0, 0, 0])

    extractor = TFIDFExtractor(max_features=50, min_df=1)
    X_train = extractor.fit_transform(train_texts)

    clf = LogisticRegression(random_state=42)
    clf.fit(X_train, y_train)

    return FakeNewsPipeline(
        classifier=clf,
        feature_extractor=extractor,
        model_name="Logistic Regression",
        preprocessing_mode="standard",
        metrics={"accuracy": 1.0, "f1_weighted": 1.0},
    )


class TestFakeNewsPipelineInference:
    """Tests for pipeline predictions, probability calibration, and confidence."""

    def test_pipeline_predict_returns_valid_labels(self, trained_pipeline: FakeNewsPipeline):
        test_inputs = [
            "President signed federal legislation today.",
            "SHOCKING conspiracy video exposed!",
        ]
        preds = trained_pipeline.predict(test_inputs)

        assert isinstance(preds, np.ndarray)
        assert len(preds) == 2
        assert set(preds).issubset({0, 1})

    def test_predict_proba_bounds_and_sum(self, trained_pipeline: FakeNewsPipeline):
        test_inputs = [
            "White House announces budget agreement.",
            "Secret plot uncovered by viral blog.",
        ]
        probas = trained_pipeline.predict_proba(test_inputs)

        assert isinstance(probas, np.ndarray)
        assert probas.shape == (2, 2)
        # All probabilities must be bounded in [0.0, 1.0]
        assert np.all(probas >= 0.0)
        assert np.all(probas <= 1.0)
        # Row sums must equal 1.0
        np.testing.assert_allclose(probas.sum(axis=1), np.ones(2), atol=1e-5)

    def test_predict_with_confidence_structure(self, trained_pipeline: FakeNewsPipeline):
        result = trained_pipeline.predict_with_confidence("President signs bill")

        assert isinstance(result, list)
        assert len(result) == 1
        entry = result[0]

        assert "label" in entry
        assert entry["label"] in (0, 1)
        assert entry["label_name"] in ("Real", "Fake")
        assert "confidence" in entry
        assert 0.5 <= entry["confidence"] <= 1.0
        assert "probabilities" in entry
        assert "Fake" in entry["probabilities"]
        assert "Real" in entry["probabilities"]
        assert "disclaimer" in entry
        assert "Educational Notice" in entry["disclaimer"]

    def test_handles_single_string_and_empty_inputs(self, trained_pipeline: FakeNewsPipeline):
        # Single string
        res_str = trained_pipeline.predict("Single test sentence")
        assert len(res_str) == 1

        # Empty string
        res_empty = trained_pipeline.predict_with_confidence("")
        assert len(res_empty) == 1
        assert res_empty[0]["confidence"] >= 0.5


class TestModelPersistence:
    """Tests for saving and loading model pipeline artifacts."""

    def test_save_and_load_roundtrip(self, trained_pipeline: FakeNewsPipeline, tmp_path: Path):
        model_file = tmp_path / "test_fake_news_pipeline.joblib"
        saved_path = save_model_pipeline(trained_pipeline, model_file)
        assert saved_path.exists()

        # Load back
        loaded = load_model_pipeline(model_file)
        assert isinstance(loaded, FakeNewsPipeline)
        assert loaded.model_name == trained_pipeline.model_name
        assert loaded.metrics == trained_pipeline.metrics

        # Test identical predictions
        sample_query = "Senate passes budget resolution."
        orig_conf = trained_pipeline.predict_with_confidence(sample_query)
        loaded_conf = loaded.predict_with_confidence(sample_query)

        assert orig_conf[0]["label"] == loaded_conf[0]["label"]
        assert orig_conf[0]["confidence"] == loaded_conf[0]["confidence"]

    def test_load_non_existent_file_raises_error(self, tmp_path: Path):
        missing_file = tmp_path / "does_not_exist.joblib"
        with pytest.raises(FileNotFoundError, match="Model artifact not found"):
            load_model_pipeline(missing_file)

    def test_load_empty_file_raises_error(self, tmp_path: Path):
        empty_file = tmp_path / "empty_model.joblib"
        empty_file.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="empty"):
            load_model_pipeline(empty_file)

    def test_load_corrupted_file_raises_error(self, tmp_path: Path):
        corrupt_file = tmp_path / "corrupted.joblib"
        corrupt_file.write_text("random non-pickle text content", encoding="utf-8")
        with pytest.raises(ValueError, match="Corrupted or invalid model"):
            load_model_pipeline(corrupt_file)
