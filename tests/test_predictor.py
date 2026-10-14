"""Unit tests for src/predictor.py."""

from datetime import datetime
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from src.predictor import ConsumptionPredictor, FEATURE_NAME, MIN_UNIQUE_DATES_REQUIRED
from src.utils import DATE_COLUMN, TARGET_COLUMN


@pytest.fixture
def sample_chronological_dataset() -> pd.DataFrame:
    """Fixture providing a realistic chronological electricity usage DataFrame."""
    dates = pd.date_range("2024-01-01", periods=30, freq="D").strftime("%Y-%m-%d")
    # Simulate a gentle upward linear trend + small variation
    usage = [20.0 + 0.1 * i + (1.0 if i % 7 in (5, 6) else 0.0) for i in range(30)]
    return pd.DataFrame({DATE_COLUMN: dates, TARGET_COLUMN: usage})


def test_predictor_initialization():
    """Verify default initialization attributes and model algorithm."""
    predictor = ConsumptionPredictor()
    assert predictor.is_trained is False
    assert isinstance(predictor.model, LinearRegression)
    assert predictor.feature_names == [FEATURE_NAME]
    assert predictor.evaluation_metrics == {}


def test_derive_day_index():
    """Verify numeric day index calculation relative to reference date."""
    ref_date = datetime(2024, 1, 1)
    dates = ["2024-01-01", "2024-01-05", "2024-01-15"]
    day_indices = ConsumptionPredictor.derive_day_index(dates, ref_date)

    assert list(day_indices) == [0.0, 4.0, 14.0]


def test_train_chronological_success(sample_chronological_dataset):
    """Verify chronological training and forward out-of-sample evaluation."""
    predictor = ConsumptionPredictor()
    metrics = predictor.train_chronological(sample_chronological_dataset, test_size=0.2)

    assert predictor.is_trained is True
    assert predictor.reference_date == datetime(2024, 1, 1)
    assert predictor.trained_at is not None

    # Check metrics existence
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2_score" in metrics
    assert metrics["mae"] >= 0.0
    assert metrics["rmse"] >= 0.0

    # Verify training metadata
    meta = predictor.training_metadata
    assert meta["model_type"] == "LinearRegression"
    assert "slope_kwh_per_day" in meta
    assert "intercept_kwh" in meta
    assert meta["train_records"] + meta["test_records"] == 30


def test_train_chronological_insufficient_records():
    """Verify training reports an error when historical dates are too few."""
    short_df = pd.DataFrame({
        DATE_COLUMN: ["2024-01-01", "2024-01-02", "2024-01-03"],
        TARGET_COLUMN: [20.0, 22.0, 21.0],
    })
    predictor = ConsumptionPredictor()

    with pytest.raises(ValueError, match="Insufficient historical data"):
        predictor.train_chronological(short_df, min_required_dates=MIN_UNIQUE_DATES_REQUIRED)


def test_predict_for_date(sample_chronological_dataset):
    """Verify predict_for_date computes correct future day index and forecasts."""
    predictor = ConsumptionPredictor()
    predictor.train_chronological(sample_chronological_dataset, test_size=0.2)

    # Predict for a date within dataset and a future date
    pred_current = predictor.predict_for_date("2024-01-10")
    pred_future = predictor.predict_for_date("2024-02-15")

    assert isinstance(pred_current, float)
    assert isinstance(pred_future, float)
    assert pred_current > 0.0
    assert pred_future > 0.0


def test_predict_before_training_raises_error():
    """Verify attempting to predict before training raises RuntimeError."""
    predictor = ConsumptionPredictor()
    with pytest.raises(RuntimeError, match="Model must be trained"):
        predictor.predict(5.0)

    with pytest.raises(RuntimeError, match="Model must be trained"):
        predictor.predict_for_date("2024-05-01")


def test_model_save_and_load(sample_chronological_dataset):
    """Verify serialization and deserialization of model and metadata via joblib."""
    predictor = ConsumptionPredictor()
    orig_metrics = predictor.train_chronological(sample_chronological_dataset, test_size=0.2)
    orig_pred = predictor.predict_for_date("2024-03-01")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_model_file = Path(tmp_dir) / "test_baseline_model.joblib"
        saved_path = predictor.save(tmp_model_file)
        assert saved_path.is_file()

        loaded_predictor = ConsumptionPredictor.load(tmp_model_file)
        assert loaded_predictor.is_trained is True
        assert loaded_predictor.evaluation_metrics == orig_metrics
        assert loaded_predictor.training_metadata["model_type"] == "LinearRegression"

        loaded_pred = loaded_predictor.predict_for_date("2024-03-01")
        assert loaded_pred == orig_pred


def test_r2_score_undefined_on_constant_test_targets():
    """Verify R2 score returns None when test targets have zero variance."""
    # 8 records: first 6 with trend, last 2 with identical constant value
    dates = pd.date_range("2024-01-01", periods=8, freq="D").strftime("%Y-%m-%d")
    values = [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 25.0, 25.0]  # last 2 are constant
    df = pd.DataFrame({DATE_COLUMN: dates, TARGET_COLUMN: values})

    predictor = ConsumptionPredictor()
    metrics = predictor.train_chronological(df, test_size=0.25)  # 2 test samples

    assert "r2_score" in metrics
    # Since variance of last 2 samples is 0, R2 is undefined (None)
    assert metrics["r2_score"] is None


def test_check_model_staleness(sample_chronological_dataset):
    """Verify check_model_staleness detects when dataset has changed since training."""
    predictor = ConsumptionPredictor()
    predictor.train_chronological(sample_chronological_dataset, test_size=0.2)

    # Initially fresh
    is_stale, reason = predictor.check_model_staleness(sample_chronological_dataset)
    assert is_stale is False
    assert "up-to-date" in reason

    # Simulate updated dataset with an additional future record
    new_row = pd.DataFrame({
        DATE_COLUMN: ["2024-02-05"],
        TARGET_COLUMN: [25.0],
    })
    modified_df = pd.concat([sample_chronological_dataset, new_row], ignore_index=True)

    is_stale_updated, stale_reason = predictor.check_model_staleness(modified_df)
    assert is_stale_updated is True
    assert "changed" in stale_reason.lower() or "2024-02-05" in stale_reason


def test_negative_prediction_detection():
    """Verify steep downward trend yields negative prediction for caller to flag."""
    dates = pd.date_range("2024-01-01", periods=10, freq="D").strftime("%Y-%m-%d")
    # Steep decline: 50, 45, 40, ...
    declining_usage = [50.0 - 5.0 * i for i in range(10)]
    df = pd.DataFrame({DATE_COLUMN: dates, TARGET_COLUMN: declining_usage})

    predictor = ConsumptionPredictor()
    predictor.train_chronological(df, test_size=0.2)

    # 100 days into the future: 50 - 5 * 100 = -450
    future_pred = predictor.predict_for_date("2024-05-01")
    assert future_pred < 0.0
