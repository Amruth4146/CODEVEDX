"""
Unit tests for the Model Evaluation & Diagnostic Visualizations Module (src/evaluation.py).
Tests:
- Regression metrics calculation
- Diagnostic plot generation (Actual vs Pred, Residual Hist, Residuals vs Pred)
- Model comparison and CV distribution charts
- Permutation & native feature importance generation
- Educational metric explanations & error interpretations
- Dataset cryptographic fingerprinting
"""

import pytest
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from src.evaluation import (
    calculate_regression_metrics,
    plot_actual_vs_predicted,
    plot_residual_distribution,
    plot_residuals_vs_predicted,
    plot_residuals,
    plot_model_comparison,
    plot_cv_score_distribution,
    plot_feature_importance_visualization,
    get_metric_explanations,
    get_error_interpretation,
    calculate_dataset_fingerprint,
)


@pytest.fixture
def sample_regression_data():
    np.random.seed(42)
    y_true = np.random.uniform(40, 95, size=50)
    noise = np.random.normal(0, 3, size=50)
    y_pred = np.clip(y_true + noise, 0, 100)
    return y_true, y_pred


@pytest.fixture
def sample_feature_matrix():
    np.random.seed(42)
    n = 60
    X = pd.DataFrame({
        "attendance_percentage": np.random.uniform(50, 100, n),
        "previous_exam_marks": np.random.uniform(40, 95, n),
        "daily_study_hours": np.random.uniform(1, 10, n),
    })
    y = (
        0.4 * X["attendance_percentage"]
        + 0.5 * X["previous_exam_marks"]
        + 2.0 * X["daily_study_hours"]
        + np.random.normal(0, 2, n)
    )
    return X, y


def test_calculate_regression_metrics(sample_regression_data):
    y_true, y_pred = sample_regression_data
    metrics = calculate_regression_metrics(y_true, y_pred)

    assert "r2" in metrics
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "mse" in metrics
    assert "mape" in metrics

    assert metrics["r2"] > 0.7
    assert metrics["rmse"] > 0.0
    assert metrics["mae"] > 0.0
    assert metrics["mse"] > 0.0
    assert metrics["rmse"] <= np.sqrt(metrics["mse"]) + 0.05  # Mathematical relationship


def test_plot_actual_vs_predicted(sample_regression_data):
    y_true, y_pred = sample_regression_data
    fig = plot_actual_vs_predicted(y_true, y_pred, model_name="Test Model")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 2  # Scatter points + parity line


def test_plot_residual_distribution(sample_regression_data):
    y_true, y_pred = sample_regression_data
    fig = plot_residual_distribution(y_true, y_pred, model_name="Test Model")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1


def test_plot_residuals_vs_predicted(sample_regression_data):
    y_true, y_pred = sample_regression_data
    fig = plot_residuals_vs_predicted(y_true, y_pred, model_name="Test Model")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1


def test_plot_residuals_backward_compatibility(sample_regression_data):
    y_true, y_pred = sample_regression_data
    fig = plot_residuals(y_true, y_pred)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2


def test_plot_model_comparison():
    mock_results = {
        "Linear Regression": {
            "cv_rmse": 3.15,
            "test_metrics": {"rmse": 3.42, "mae": 2.78, "mse": 11.72, "r2": 0.864},
        },
        "Random Forest Regressor": {
            "cv_rmse": 4.03,
            "test_metrics": {"rmse": 4.50, "mae": 3.63, "mse": 20.25, "r2": 0.765},
        },
    }
    fig = plot_model_comparison(mock_results)
    assert isinstance(fig, go.Figure)


def test_plot_cv_score_distribution():
    mock_results = {
        "Linear Regression": {
            "cv_rmse": 3.15,
            "cv_fold_rmse": [3.10, 3.25, 3.05, 3.18, 3.17],
        },
        "Random Forest Regressor": {
            "cv_rmse": 4.03,
            "cv_fold_rmse": [4.10, 3.95, 4.05, 4.02, 4.03],
        },
    }
    fig = plot_cv_score_distribution(mock_results)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2  # 2 algorithms


def test_plot_feature_importance_linear_permutation(sample_feature_matrix):
    X, y = sample_feature_matrix
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression()),
    ])
    pipe.fit(X, y)

    fig, explanation = plot_feature_importance_visualization(
        pipeline=pipe,
        X_test=X,
        y_test=y,
        feature_names=list(X.columns),
        model_name="Linear Regression",
    )
    assert isinstance(fig, go.Figure)
    assert "Permutation Importance" in explanation
    assert len(fig.data) >= 1


def test_plot_feature_importance_random_forest_native(sample_feature_matrix):
    X, y = sample_feature_matrix
    rf_pipe = Pipeline([
        ("regressor", RandomForestRegressor(n_estimators=10, random_state=42)),
    ])
    rf_pipe.fit(X, y)

    fig, explanation = plot_feature_importance_visualization(
        pipeline=rf_pipe,
        X_test=X,
        y_test=y,
        feature_names=list(X.columns),
        model_name="Random Forest",
    )
    assert isinstance(fig, go.Figure)
    assert "Native Random Forest Feature Importance" in explanation


def test_get_metric_explanations():
    defs = get_metric_explanations()
    assert "MAE" in defs
    assert "MSE" in defs
    assert "RMSE" in defs
    assert "R2" in defs
    for m in ["MAE", "MSE", "RMSE", "R2"]:
        assert "formula" in defs[m]
        assert "unit" in defs[m]
        assert "interpretation" in defs[m]


def test_get_error_interpretation():
    interp = get_error_interpretation(mae=2.78, rmse=3.42, r2=0.864)
    assert "summary" in interp
    assert "point_impact" in interp
    assert "outlier_risk" in interp
    assert "confidence_rule" in interp
    assert "2.78" in interp["point_impact"]
    assert "3.42" in interp["confidence_rule"]


def test_calculate_dataset_fingerprint():
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [4.0, 5.0, 6.0],
    })
    fp = calculate_dataset_fingerprint(df)
    assert "sha256" in fp
    assert len(fp["sha256"]) == 64
    assert fp["total_records"] == 3
    assert fp["total_columns"] == 2
