"""Predictor module for daily electricity consumption forecasting.

IMPORTANT ARCHITECTURAL & METHODOLOGICAL NOTE:
-----------------------------------------------------------------------------
This module implements a simple trend-based baseline using scikit-learn's
LinearRegression. It models daily electricity usage (usage_kwh) as a linear function
of an integer day index derived from chronological dates:
    usage_kwh = intercept + slope * day_index

This is an interpretable, foundational baseline designed to capture macro linear
drift over time. It is NOT an accurate seasonal forecasting system, as it does not
account for cyclical weather patterns, summer cooling/winter heating waves, or
day-of-week intra-week variance.

Model accuracy metrics (MAE, RMSE, R2) are reported strictly as computed on unseen
chronological test data. Accuracy numbers are NEVER fabricated, exaggerated, or invented.
-----------------------------------------------------------------------------
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.utils import DATE_COLUMN, DEFAULT_MODEL_PATH, TARGET_COLUMN

# Minimum required unique dates to train a meaningful chronological model
MIN_UNIQUE_DATES_REQUIRED = 5
FEATURE_NAME = "day_index"


class ConsumptionPredictor:
    """Baseline linear trend predictor for daily electricity consumption (kWh).

    Uses a scikit-learn LinearRegression model fitted on a numeric day index
    derived from chronological dates. Evaluated strictly on forward out-of-sample
    test observations to prevent future-data leakage.
    """

    def __init__(self, model: Optional[LinearRegression] = None) -> None:
        """Initialize the ConsumptionPredictor.

        Args:
            model: Optional pre-configured LinearRegression instance.
        """
        self.model: LinearRegression = model or LinearRegression()
        self.feature_names: List[str] = [FEATURE_NAME]
        self.target_name: str = TARGET_COLUMN
        self.reference_date: Optional[datetime] = None
        self.is_trained: bool = False
        self.trained_at: Optional[str] = None
        self.training_metadata: Dict[str, Any] = {}
        self.evaluation_metrics: Dict[str, Optional[float]] = {}

    @staticmethod
    def derive_day_index(
        dates: Union[pd.Series, List[Union[str, datetime]]],
        reference_date: datetime,
    ) -> np.ndarray:
        """Derive integer elapsed days from a reference baseline date.

        Args:
            dates: Sequence of date strings (YYYY-MM-DD) or datetime objects.
            reference_date: Reference base datetime representing day index 0.

        Returns:
            1D NumPy array of integer day indices.
        """
        dt_series = pd.Series(pd.to_datetime(dates))
        ref_ts = pd.Timestamp(reference_date)
        day_indices = (dt_series - ref_ts).dt.days.to_numpy(dtype=float)
        return day_indices

    def train_chronological(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        min_required_dates: int = MIN_UNIQUE_DATES_REQUIRED,
    ) -> Dict[str, Optional[float]]:
        """Train the model on earlier chronological data and evaluate on later data.

        To prevent future-data leakage, records are sorted strictly chronologically
        and split without shuffling:
            Train set: earlier (1 - test_size) portion of observations.
            Test set : later test_size portion of observations.

        Args:
            df: DataFrame containing DATE_COLUMN and TARGET_COLUMN.
            test_size: Fraction of chronological observations reserved for testing (0.0 < test_size < 1.0).
            min_required_dates: Minimum unique dates required for training.

        Returns:
            Dictionary of evaluation metrics on unseen test observations (MAE, RMSE, R2).

        Raises:
            ValueError: If dataset is too small, missing columns, or test_size is invalid.
        """
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("Cannot train model: provided dataset is empty.")

        if DATE_COLUMN not in df.columns:
            raise ValueError(f"Missing required date column: '{DATE_COLUMN}'")
        if TARGET_COLUMN not in df.columns:
            raise ValueError(f"Missing required target column: '{TARGET_COLUMN}'")

        if not (0.0 < test_size < 1.0):
            raise ValueError(f"test_size must be between 0.0 and 1.0, received: {test_size}")

        # Clean and sort chronologically
        clean_df = df.copy()
        clean_df[DATE_COLUMN] = pd.to_datetime(clean_df[DATE_COLUMN], errors="coerce")
        clean_df[TARGET_COLUMN] = pd.to_numeric(clean_df[TARGET_COLUMN], errors="coerce")
        clean_df = clean_df.dropna(subset=[DATE_COLUMN, TARGET_COLUMN])
        clean_df = clean_df.drop_duplicates(subset=[DATE_COLUMN], keep="first")
        clean_df = clean_df.sort_values(by=DATE_COLUMN).reset_index(drop=True)

        unique_dates_count = len(clean_df)
        if unique_dates_count < min_required_dates:
            raise ValueError(
                f"Insufficient historical data: {unique_dates_count} unique date(s) found, "
                f"but at least {min_required_dates} are required to train a chronological model."
            )

        # Reference baseline date is the earliest recorded date (day_index = 0)
        self.reference_date = clean_df[DATE_COLUMN].min().to_pydatetime()
        clean_df[FEATURE_NAME] = self.derive_day_index(
            clean_df[DATE_COLUMN], self.reference_date
        )

        # Chronological train/test split (no random shuffle!)
        split_idx = int(unique_dates_count * (1.0 - test_size))
        # Ensure at least 1 training sample and at least 1 test sample
        split_idx = max(1, min(split_idx, unique_dates_count - 1))

        train_df = clean_df.iloc[:split_idx]
        test_df = clean_df.iloc[split_idx:]

        X_train = train_df[[FEATURE_NAME]]
        y_train = train_df[TARGET_COLUMN].to_numpy()

        X_test = test_df[[FEATURE_NAME]]
        y_test = test_df[TARGET_COLUMN].to_numpy()

        # Fit linear regression model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        self.trained_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Evaluate on later (unseen) test observations
        self.evaluation_metrics = self._calculate_metrics(X_test, y_test)

        # Record comprehensive training metadata
        slope = float(self.model.coef_[0])
        intercept = float(self.model.intercept_)
        self.training_metadata = {
            "model_type": "LinearRegression",
            "model_description": "Baseline linear trend model (usage_kwh = intercept + slope * day_index)",
            "reference_date": self.reference_date.strftime("%Y-%m-%d"),
            "feature_name": FEATURE_NAME,
            "target_name": self.target_name,
            "slope_kwh_per_day": slope,
            "intercept_kwh": intercept,
            "trained_at": self.trained_at,
            "total_records": unique_dates_count,
            "train_records": len(train_df),
            "test_records": len(test_df),
            "train_date_range": (
                f"{train_df[DATE_COLUMN].min().strftime('%Y-%m-%d')} to "
                f"{train_df[DATE_COLUMN].max().strftime('%Y-%m-%d')}"
            ),
            "test_date_range": (
                f"{test_df[DATE_COLUMN].min().strftime('%Y-%m-%d')} to "
                f"{test_df[DATE_COLUMN].max().strftime('%Y-%m-%d')}"
            ),
        }

        return self.evaluation_metrics

    def _calculate_metrics(
        self,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
    ) -> Dict[str, Optional[float]]:
        """Calculate MAE, RMSE, and R2 (when defined) on test observations.

        Args:
            X_test: Test features DataFrame.
            y_test: Actual test consumption values.

        Returns:
            Dictionary with MAE, MSE, RMSE, and R2.
        """
        predictions = self.model.predict(X_test)

        mae = float(mean_absolute_error(y_test, predictions))
        mse = float(mean_squared_error(y_test, predictions))
        rmse = float(np.sqrt(mse))

        # R2 score is defined only when test set has >= 2 samples and non-zero variance
        if len(y_test) >= 2 and float(np.var(y_test)) > 1e-9:
            r2 = float(r2_score(y_test, predictions))
        else:
            r2 = None

        return {
            "mae": round(mae, 4),
            "mse": round(mse, 4),
            "rmse": round(rmse, 4),
            "r2_score": round(r2, 4) if r2 is not None else None,
        }

    def train(self, X_train: pd.DataFrame, y_train: Union[pd.Series, np.ndarray]) -> "ConsumptionPredictor":
        """Generic train interface for feature matrix input.

        Args:
            X_train: Feature DataFrame containing day_index or single numeric column.
            y_train: Target usage values.

        Returns:
            Self instance.
        """
        if isinstance(X_train, pd.DataFrame):
            if FEATURE_NAME in X_train.columns:
                X_fit = X_train[[FEATURE_NAME]]
            else:
                X_fit = X_train.iloc[:, :1]
        else:
            X_arr = np.asarray(X_train)
            X_fit = X_arr.reshape(-1, 1) if X_arr.ndim == 1 else X_arr

        y_fit = np.asarray(y_train)
        self.model.fit(X_fit, y_fit)
        self.is_trained = True
        self.trained_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return self

    def evaluate(self, X_test: pd.DataFrame, y_test: Union[pd.Series, np.ndarray]) -> Dict[str, Optional[float]]:
        """Evaluate the fitted model on test inputs.

        Args:
            X_test: Test feature matrix.
            y_test: Actual test targets.

        Returns:
            Dictionary containing MAE, RMSE, and R2.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before running evaluation.")

        if isinstance(X_test, pd.DataFrame):
            if FEATURE_NAME in X_test.columns:
                X_eval = X_test[[FEATURE_NAME]]
            else:
                X_eval = X_test.iloc[:, :1]
        else:
            X_arr = np.asarray(X_test)
            X_eval = X_arr.reshape(-1, 1) if X_arr.ndim == 1 else X_arr

        y_arr = np.asarray(y_test)
        self.evaluation_metrics = self._calculate_metrics(X_eval, y_arr)
        return self.evaluation_metrics

    def predict(
        self,
        features: Union[pd.DataFrame, Dict[str, Any], np.ndarray, float, int],
    ) -> Union[float, np.ndarray]:
        """Predict electricity consumption in kWh from day index input.

        Args:
            features: Numeric day index (float/int), dictionary, array, or DataFrame.

        Returns:
            Predicted consumption in kWh.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions.")

        if isinstance(features, (int, float, np.integer, np.floating)):
            X_df = pd.DataFrame([[float(features)]], columns=self.feature_names)
            pred = float(self.model.predict(X_df)[0])
            return round(pred, 2)

        if isinstance(features, dict):
            if FEATURE_NAME in features:
                val = float(features[FEATURE_NAME])
            else:
                val = float(list(features.values())[0])
            X_df = pd.DataFrame([[val]], columns=self.feature_names)
            pred = float(self.model.predict(X_df)[0])
            return round(pred, 2)

        if isinstance(features, pd.DataFrame):
            if FEATURE_NAME in features.columns:
                X_in = features[[FEATURE_NAME]]
            else:
                X_in = features.iloc[:, :1].copy()
                X_in.columns = self.feature_names
            preds = self.model.predict(X_in)
            return np.round(preds, 2)

        X_arr = np.asarray(features)
        X_in = X_arr.reshape(-1, 1) if X_arr.ndim == 1 else X_arr
        X_df = pd.DataFrame(X_in, columns=self.feature_names)
        preds = self.model.predict(X_df)
        return np.round(preds, 2)

    def predict_for_date(
        self,
        date_input: Union[datetime, str],
        **kwargs: Any,
    ) -> float:
        """Forecast electricity consumption for a specific calendar date.

        Computes the target day index relative to the reference date established
        during training:
            day_index = (target_date - reference_date).days

        Args:
            date_input: datetime object or ISO date string (YYYY-MM-DD).
            **kwargs: Ignored keyword arguments for backwards compatibility.

        Returns:
            Forecasted electricity consumption in kWh.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before predicting for a date.")

        if self.reference_date is None:
            # Fallback reference date if loaded without metadata
            self.reference_date = datetime(2023, 1, 1)

        if isinstance(date_input, str):
            dt = datetime.strptime(date_input.strip()[:10], "%Y-%m-%d")
        elif isinstance(date_input, datetime):
            dt = date_input
        else:
            raise TypeError(f"Expected datetime or string date, received {type(date_input).__name__}")

        day_index = float((dt.date() - self.reference_date.date()).days)
        prediction = self.predict(day_index)
        return float(prediction)

    def save(self, model_path: Union[str, Path] = DEFAULT_MODEL_PATH) -> Path:
        """Serialize and save the trained model, training metadata, and metrics using joblib.

        Args:
            model_path: Destination path for the .joblib file.

        Returns:
            Path object of the saved model artifact.

        Raises:
            RuntimeError: If attempting to save an untrained model.
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save an untrained model.")

        path = Path(model_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "model": self.model,
                "feature_names": self.feature_names,
                "target_name": self.target_name,
                "reference_date": self.reference_date,
                "training_metadata": self.training_metadata,
                "evaluation_metrics": self.evaluation_metrics,
                "trained_at": self.trained_at,
            }
            joblib.dump(payload, path)
            return path
        except Exception as exc:
            raise IOError(f"Failed to save model artifact to '{path}': {exc}") from exc

    @classmethod
    def load(cls, model_path: Union[str, Path] = DEFAULT_MODEL_PATH) -> "ConsumptionPredictor":
        """Load a previously serialized model and metadata from disk.

        Args:
            model_path: Path to the .joblib artifact.

        Returns:
            Restored ConsumptionPredictor instance.

        Raises:
            FileNotFoundError: If model file does not exist.
            IOError: If deserialization fails.
        """
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found at: '{path.resolve()}'")

        try:
            payload = joblib.load(path)
            instance = cls(model=payload["model"])
            instance.feature_names = payload.get("feature_names", [FEATURE_NAME])
            instance.target_name = payload.get("target_name", TARGET_COLUMN)
            instance.reference_date = payload.get("reference_date", None)
            instance.training_metadata = payload.get("training_metadata", {})
            instance.evaluation_metrics = payload.get("evaluation_metrics", {})
            instance.trained_at = payload.get("trained_at", None)
            instance.is_trained = True
            return instance
        except Exception as exc:
            raise IOError(f"Failed to load model artifact from '{path}': {exc}") from exc
