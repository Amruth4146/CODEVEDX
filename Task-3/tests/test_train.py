"""Unit tests for train.py in Task-3."""

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import issparse
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from src.feature_extraction import TFIDFExtractor
from src.train import (
    calculate_evaluation_metrics,
    select_best_model,
    train_and_compare_models,
    train_logistic_regression,
    train_multinomial_nb,
)


@pytest.fixture
def synthetic_train_val_dfs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fixture generating balanced synthetic training and validation DataFrames."""
    train_data = {
        "content": [
            "WASHINGTON (Reuters) - Federal officials passed the budget bill.",
            "White House announced economic reform package today.",
            "Senate voted to confirm the judicial nomination in Washington.",
            "Treasury department published quarterly GDP statistics.",
            "SHOCKING: Secret memo reveals hidden election fraud plot!",
            "WATCH: Unverified viral video claims scandalous conspiracy [VIDEO]",
            "BREAKING: Anonymous blog exposes fabricated political story!",
            "Outrageous scandal uncovered by alternative media outlet!",
        ],
        "label": [1, 1, 1, 1, 0, 0, 0, 0],
    }
    val_data = {
        "content": [
            "Congress discussed federal budget appropriations.",
            "Diplomats met for economic reform talks.",
            "SHOCKING: Viral blog alleges secret election plot!",
            "WATCH: Fabricated video surfaces online scandal!",
        ],
        "label": [1, 1, 0, 0],
    }
    return pd.DataFrame(train_data), pd.DataFrame(val_data)


class TestMetricsCalculation:
    """Tests for calculate_evaluation_metrics."""

    def test_perfect_predictions_metrics(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0])
        metrics = calculate_evaluation_metrics(y_true, y_pred)

        assert metrics["accuracy"] == 1.0
        assert metrics["f1_weighted"] == 1.0
        assert metrics["fake_f1"] == 1.0
        assert metrics["real_f1"] == 1.0
        assert metrics["fake_precision"] == 1.0
        assert metrics["fake_recall"] == 1.0
        assert metrics["true_fake"] == 2
        assert metrics["true_real"] == 2
        assert metrics["false_fake"] == 0
        assert metrics["false_real"] == 0

    def test_imperfect_predictions_metrics(self):
        # 1 False Positive, 0 False Negatives
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])  # One fake predicted as real
        metrics = calculate_evaluation_metrics(y_true, y_pred)

        assert metrics["accuracy"] == 0.75
        assert metrics["true_fake"] == 1
        assert metrics["false_real"] == 1
        assert metrics["true_real"] == 2
        assert metrics["false_fake"] == 0


class TestModelTrainingFunctions:
    """Tests for train_logistic_regression and train_multinomial_nb."""

    def test_train_logistic_regression(self):
        extractor = TFIDFExtractor(max_features=20, min_df=1)
        texts = ["real news article one", "real news article two", "fake conspiracy blog", "fake hoax claim"]
        y = np.array([1, 1, 0, 0])
        X = extractor.fit_transform(texts)

        clf = train_logistic_regression(X, y)
        assert isinstance(clf, LogisticRegression)
        assert hasattr(clf, "predict")
        assert hasattr(clf, "predict_proba")
        preds = clf.predict(X)
        assert len(preds) == 4

    def test_train_multinomial_nb(self):
        extractor = TFIDFExtractor(max_features=20, min_df=1)
        texts = ["real news article one", "real news article two", "fake conspiracy blog", "fake hoax claim"]
        y = np.array([1, 1, 0, 0])
        X = extractor.fit_transform(texts)

        clf = train_multinomial_nb(X, y)
        assert isinstance(clf, MultinomialNB)
        assert hasattr(clf, "predict")
        assert hasattr(clf, "predict_proba")
        preds = clf.predict(X)
        assert len(preds) == 4


class TestModelSelection:
    """Tests for select_best_model logic."""

    def test_select_model_with_higher_f1(self):
        metrics_by_model = {
            "Model A": {"f1_weighted": 0.90, "fake_f1": 0.89, "accuracy": 0.90},
            "Model B": {"f1_weighted": 0.96, "fake_f1": 0.95, "accuracy": 0.96},
        }
        best = select_best_model(metrics_by_model)
        assert best == "Model B"

    def test_empty_metrics_raises_error(self):
        with pytest.raises(ValueError, match="Cannot select best model"):
            select_best_model({})


class TestTrainAndComparePipeline:
    """Tests for end-to-end train_and_compare_models."""

    def test_train_and_compare_end_to_end(
        self, synthetic_train_val_dfs: tuple[pd.DataFrame, pd.DataFrame]
    ):
        train_df, val_df = synthetic_train_val_dfs

        results = train_and_compare_models(
            train_df=train_df,
            val_df=val_df,
            preprocessing_mode="standard",
            max_features=50,
            random_state=42,
        )

        assert "metrics" in results
        assert "Logistic Regression" in results["metrics"]
        assert "Multinomial Naive Bayes" in results["metrics"]
        assert "best_model_name" in results
        assert results["best_model_name"] in ("Logistic Regression", "Multinomial Naive Bayes")
        assert results["best_pipeline"].classifier is not None
        assert results["best_pipeline"].feature_extractor.is_fitted is True

        # Test inference from best pipeline
        test_pred = results["best_pipeline"].predict(["Congress discussed federal budget reform."])
        assert test_pred[0] in (0, 1)

    def test_empty_inputs_raise_error(self):
        empty_df = pd.DataFrame(columns=["content", "label"])
        valid_df = pd.DataFrame({"content": ["text"], "label": [1]})
        with pytest.raises(ValueError, match="must be non-empty"):
            train_and_compare_models(empty_df, valid_df)
