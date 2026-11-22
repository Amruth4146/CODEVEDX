"""Unit tests for predictor.py in Task-3."""

from pathlib import Path
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.feature_extraction import TFIDFExtractor
from src.model_utils import FakeNewsPipeline
from src.predictor import (
    FakeNewsPredictor,
    PredictionResult,
    get_default_predictor,
    predict_news,
    predict_news_batch,
    reset_default_predictor,
    validate_news_input,
)


@pytest.fixture
def synthetic_pipeline() -> FakeNewsPipeline:
    """Fixture providing a fast synthetic FakeNewsPipeline."""
    train_texts = [
        "Senate passed the bipartisan government funding legislation today.",
        "President signed executive order regarding federal agency operations.",
        "Treasury department published quarterly macroeconomic inflation figures.",
        "SHOCKING BREAKING: Secret alien conspiracy steals election [VIDEO]!",
        "WATCH: Unverified viral blogger exposes massive corrupt treason plot!",
        "FABRICATED HOAX: Outrageous fake scandal revealed by insider blog!",
    ]
    y_train = np.array([1, 1, 1, 0, 0, 0])

    extractor = TFIDFExtractor(max_features=50, min_df=1)
    X_train = extractor.fit_transform(train_texts)

    clf = LogisticRegression(random_state=42)
    clf.fit(X_train, y_train)

    return FakeNewsPipeline(
        classifier=clf,
        feature_extractor=extractor,
        model_name="Synthetic Logistic Regression",
        preprocessing_mode="standard",
        metrics={"accuracy": 1.0, "f1_weighted": 1.0},
    )


@pytest.fixture
def synthetic_predictor(synthetic_pipeline: FakeNewsPipeline) -> FakeNewsPredictor:
    """Fixture providing a predictor backed by the synthetic pipeline."""
    return FakeNewsPredictor(pipeline=synthetic_pipeline)


class TestInputValidation:
    """Tests for validate_news_input and defensive parameter handling."""

    def test_valid_text_passes_unchanged(self):
        text = "This is a legitimate news story containing enough words."
        validated = validate_news_input(text)
        assert validated == text

    def test_none_input_raises_type_error(self):
        with pytest.raises(TypeError, match="cannot be None"):
            validate_news_input(None)

    def test_non_string_types_raise_type_error(self):
        with pytest.raises(TypeError, match="Invalid input type"):
            validate_news_input(12345)

        with pytest.raises(TypeError, match="Invalid input type"):
            validate_news_input(["headline", "text"])

        with pytest.raises(TypeError, match="Invalid input type"):
            validate_news_input(True)

    def test_empty_string_raises_value_error(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_news_input("")

    def test_whitespace_only_raises_value_error(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_news_input("   \n\t  ")

    def test_extremely_short_text_raises_value_error(self):
        with pytest.raises(ValueError, match="too short to evaluate reliably"):
            validate_news_input("Hi")

        with pytest.raises(ValueError, match="too short to evaluate reliably"):
            validate_news_input("Two words")

    def test_extremely_long_text_truncated_safely(self):
        long_text = "word " * 30_000  # 150,000 characters
        validated = validate_news_input(long_text, max_chars=50_000)
        assert len(validated) == 50_000


class TestPredictorInference:
    """Tests for single and batch predictions using synthetic predictor."""

    def test_real_like_prediction(self, synthetic_predictor: FakeNewsPredictor):
        text = "Senate passed the bipartisan government funding legislation today."
        result = synthetic_predictor.predict(text)

        assert isinstance(result, PredictionResult)
        assert result.label == 1
        assert result.label_name == "Real"
        assert result.confidence >= 0.5
        assert result.real_probability >= 0.5
        assert result.fake_probability <= 0.5

    def test_fake_like_prediction(self, synthetic_predictor: FakeNewsPredictor):
        text = "SHOCKING BREAKING: Secret alien conspiracy steals election [VIDEO]!"
        result = synthetic_predictor.predict(text)

        assert isinstance(result, PredictionResult)
        assert result.label == 0
        assert result.label_name == "Fake"
        assert result.confidence >= 0.5
        assert result.fake_probability >= 0.5
        assert result.real_probability <= 0.5

    def test_probability_properties(self, synthetic_predictor: FakeNewsPredictor):
        text = "Federal court issues ruling regarding election procedures."
        result = synthetic_predictor.predict(text)

        # Probabilities bounded in [0, 1]
        assert 0.0 <= result.fake_probability <= 1.0
        assert 0.0 <= result.real_probability <= 1.0
        # Confidence is max probability
        assert result.confidence == max(result.fake_probability, result.real_probability)
        # Sum to 1.0
        assert pytest.approx(result.fake_probability + result.real_probability, abs=1e-4) == 1.0

    def test_batch_prediction(self, synthetic_predictor: FakeNewsPredictor):
        texts = [
            "Senate passed the bipartisan government funding legislation today.",
            "SHOCKING BREAKING: Secret alien conspiracy steals election [VIDEO]!",
            "Treasury department published quarterly macroeconomic inflation figures.",
        ]
        results = synthetic_predictor.predict_batch(texts)

        assert len(results) == 3
        assert results[0].label_name == "Real"
        assert results[1].label_name == "Fake"
        assert results[2].label_name == "Real"

    def test_prediction_consistency(self, synthetic_predictor: FakeNewsPredictor):
        text = "President signed executive order regarding federal operations."
        r1 = synthetic_predictor.predict(text)
        r2 = synthetic_predictor.predict(text)

        assert r1.label == r2.label
        assert r1.confidence == r2.confidence
        assert r1.real_probability == r2.real_probability

    def test_no_retraining_during_inference(self, synthetic_predictor: FakeNewsPredictor):
        vocab_before = synthetic_predictor.feature_count
        weights_before = synthetic_predictor.pipeline.classifier.coef_.copy()

        # Run several predictions
        synthetic_predictor.predict("Senate debated new federal legislation today.")
        synthetic_predictor.predict("Unseen rare alien words extraterrestrial galaxy.")

        # Ensure model vocabulary and weights remained unchanged
        assert synthetic_predictor.feature_count == vocab_before
        np.testing.assert_array_equal(
            synthetic_predictor.pipeline.classifier.coef_, weights_before
        )

    def test_prediction_result_to_dict(self, synthetic_predictor: FakeNewsPredictor):
        text = "Senate passed the bipartisan government funding legislation today."
        res = synthetic_predictor.predict(text)
        d = res.to_dict()

        assert isinstance(d, dict)
        assert d["label"] == 1
        assert d["label_name"] == "Real"
        assert "confidence" in d
        assert "fake_probability" in d
        assert "real_probability" in d
        assert "model_name" in d
        assert "cleaned_preview" in d
        assert "disclaimer" in d
        assert "Educational Notice" in d["disclaimer"]


class TestModelLoadingAndCaching:
    """Tests for loading model from disk and predictor caching."""

    def test_missing_model_file_raises_error(self, tmp_path: Path):
        bad_path = tmp_path / "non_existent.joblib"
        with pytest.raises(FileNotFoundError, match="Champion model not found"):
            FakeNewsPredictor(model_path=bad_path)

    def test_corrupted_model_file_raises_error(self, tmp_path: Path):
        corrupt_path = tmp_path / "corrupt.joblib"
        corrupt_path.write_text("random bad bytes", encoding="utf-8")
        with pytest.raises(ValueError, match="Failed to load champion model"):
            FakeNewsPredictor(model_path=corrupt_path)

    def test_singleton_get_and_reset_predictor(self, synthetic_pipeline: FakeNewsPipeline, tmp_path: Path):
        from src.model_utils import save_model_pipeline

        model_file = tmp_path / "fake_news_model.joblib"
        save_model_pipeline(synthetic_pipeline, model_file)

        reset_default_predictor()
        pred1 = get_default_predictor(model_path=model_file)
        pred2 = get_default_predictor(model_path=model_file)

        # Caching returns the same instance
        assert pred1 is pred2

        reset_default_predictor()
        # After reset, next call creates fresh instance
        pred3 = get_default_predictor(model_path=model_file)
        assert pred3 is not pred1

    def test_convenience_predict_functions(self, synthetic_pipeline: FakeNewsPipeline, tmp_path: Path):
        from src.model_utils import save_model_pipeline

        model_file = tmp_path / "fake_news_model.joblib"
        save_model_pipeline(synthetic_pipeline, model_file)

        reset_default_predictor()
        res = predict_news(
            "Senate passed bipartisan legislation today.", model_path=model_file
        )
        assert isinstance(res, PredictionResult)
        assert res.label == 1

        batch_res = predict_news_batch(
            ["Senate passed legislation today.", "Secret conspiracy revealed video."],
            model_path=model_file,
        )
        assert len(batch_res) == 2


class TestRealChampionModelIntegration:
    """Integration test with the actual saved champion model artifact."""

    def test_real_champion_model_prediction(self):
        champion_path = Path(__file__).resolve().parent.parent / "models" / "fake_news_model.joblib"
        if not champion_path.exists():
            pytest.skip("Champion model not found at expected path.")

        predictor = FakeNewsPredictor(model_path=champion_path)
        assert predictor.model_name == "Logistic Regression"
        assert predictor.feature_count == 5000

        # Real news sample
        real_text = "The Senate voted to approve the federal budget bill on Thursday after negotiations."
        real_res = predictor.predict(real_text)
        assert real_res.label == 1
        assert real_res.label_name == "Real"
        assert real_res.confidence > 0.5

        # Fake news sample
        fake_text = "SHOCKING BREAKING: Secret hidden cabal stole the election, anonymous insider leaks proof [VIDEO]!!"
        fake_res = predictor.predict(fake_text)
        assert fake_res.label == 0
        assert fake_res.label_name == "Fake"
        assert fake_res.confidence > 0.5
