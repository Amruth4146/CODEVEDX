"""
Unit tests for Machine Learning Training Workflow Module.
Student Performance Prediction System - CODEVEDX Task 2
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Ensure project root in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.train_model import (
    validate_dataset_schema,
    load_and_prepare_dataset,
    build_pipeline_architectures,
    get_rf_search_space,
    evaluate_predictions,
    train_and_select_models,
    save_model_artifacts,
    load_model_bundle,
    validate_loaded_model_schema,
)


@pytest.fixture
def sample_training_data():
    """Generates synthetic student dataset for fast, isolated test execution."""
    np.random.seed(42)
    n = 100
    att = np.random.uniform(50.0, 100.0, n)
    prev = np.random.uniform(30.0, 95.0, n)
    hours = np.random.uniform(1.0, 8.0, n)
    # Target linearly correlated with features plus slight noise
    final = 0.25 * att + 0.45 * prev + 2.5 * hours + np.random.normal(0, 2.0, n)
    final = np.clip(final, 0.0, 100.0)

    df = pd.DataFrame({
        "student_id": [f"STU_{i:04d}" for i in range(n)],
        "attendance_percentage": att,
        "previous_exam_marks": prev,
        "daily_study_hours": hours,
        "assignments_completed": np.random.randint(4, 11, n),
        "extracurricular_activities": np.random.choice(["Yes", "No"], n),
        "sleep_hours": np.random.uniform(5.5, 9.0, n),
        "final_exam_marks": final,
    })
    return df


def test_validate_dataset_schema_valid(sample_training_data):
    valid, err = validate_dataset_schema(sample_training_data)
    assert valid is True
    assert err is None


def test_validate_dataset_schema_missing_target(sample_training_data):
    df_no_target = sample_training_data.drop(columns=["final_exam_marks"])
    valid, err = validate_dataset_schema(df_no_target)
    assert valid is False
    assert "Target column not found" in err


def test_validate_dataset_schema_too_few_rows():
    tiny_df = pd.DataFrame({
        "attendance_percentage": [90.0],
        "daily_study_hours": [4.0],
        "final_exam_marks": [85.0],
    })
    valid, err = validate_dataset_schema(tiny_df)
    assert valid is False
    assert "insufficient samples" in err.lower()


def test_validate_dataset_schema_missing_predictor(sample_training_data):
    df_no_pred = sample_training_data.drop(columns=["attendance_percentage"])
    valid, err = validate_dataset_schema(df_no_pred)
    assert valid is False
    assert "attendance_percentage" in err


def test_build_pipeline_architectures():
    pipes = build_pipeline_architectures(random_state=42)
    assert "Linear Regression" in pipes
    assert "Random Forest Regressor" in pipes

    lr_pipe = pipes["Linear Regression"]
    assert isinstance(lr_pipe, Pipeline)
    assert "imputer" in lr_pipe.named_steps
    assert "scaler" in lr_pipe.named_steps
    assert isinstance(lr_pipe.named_steps["regressor"], LinearRegression)

    rf_pipe = pipes["Random Forest Regressor"]
    assert isinstance(rf_pipe, Pipeline)
    assert "imputer" in rf_pipe.named_steps
    assert isinstance(rf_pipe.named_steps["regressor"], RandomForestRegressor)


def test_evaluate_predictions_metrics():
    y_true = np.array([50.0, 70.0, 85.0, 90.0])
    y_pred = np.array([52.0, 68.0, 87.0, 91.0])

    metrics = evaluate_predictions(y_true, y_pred)
    assert "mae" in metrics
    assert "mse" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics
    # Verify values are valid floats
    assert metrics["mae"] > 0.0
    assert metrics["rmse"] > 0.0
    assert 0.0 <= metrics["r2"] <= 1.0


def test_train_and_select_models_execution(sample_training_data):
    # Prepare features and target
    y = sample_training_data["final_exam_marks"]
    X = sample_training_data[["attendance_percentage", "previous_exam_marks", "daily_study_hours"]].copy()

    X_train, X_test = X.iloc[:80], X.iloc[80:]
    y_train, y_test = y.iloc[:80], y.iloc[80:]

    best_name, best_pipe, best_params, model_results, test_preds = train_and_select_models(
        X_train, y_train, X_test, y_test, random_state=42, cv_folds=3
    )

    assert best_name in ["Linear Regression", "Random Forest Regressor"]
    assert isinstance(best_pipe, Pipeline)
    assert "Linear Regression" in model_results
    assert "Random Forest Regressor" in model_results

    # Winning model must strictly have lowest or equal CV RMSE
    winner_cv_rmse = model_results[best_name]["cv_rmse"]
    other_name = "Random Forest Regressor" if best_name == "Linear Regression" else "Linear Regression"
    other_cv_rmse = model_results[other_name]["cv_rmse"]
    assert winner_cv_rmse <= other_cv_rmse

    # Ensure test predictions match test set size
    assert len(test_preds[best_name]) == len(X_test)


def test_schema_validation_prevents_stale_model(tmp_path, sample_training_data):
    # Create and save a small pipeline
    y = sample_training_data["final_exam_marks"]
    X = sample_training_data[["attendance_percentage", "previous_exam_marks", "daily_study_hours"]]
    feature_names = list(X.columns)

    pipes = build_pipeline_architectures(random_state=42)
    lr = pipes["Linear Regression"]
    lr.fit(X, y)

    test_metrics = {"mae": 2.1, "mse": 6.2, "rmse": 2.49, "r2": 0.91}
    model_results = {"Linear Regression": {"cv_rmse": 2.45, "test_metrics": test_metrics}}

    artifact_path = tmp_path / "test_model.joblib"
    saved_path = save_model_artifacts(
        pipeline=lr,
        model_name="Linear Regression",
        feature_names=feature_names,
        test_metrics=test_metrics,
        model_results=model_results,
        best_params="OLS",
        training_samples=len(X),
        filepath=artifact_path,
    )
    assert saved_path.exists()

    bundle = load_model_bundle(artifact_path)
    assert bundle["model_name"] == "Linear Regression"
    assert bundle["feature_names"] == feature_names

    # Test compatible dataframe
    valid_df = pd.DataFrame({
        "attendance_percentage": [80.0],
        "previous_exam_marks": [75.0],
        "daily_study_hours": [4.0],
    })
    is_valid, err = validate_loaded_model_schema(bundle, valid_df)
    assert is_valid is True
    assert err is None

    # Test incompatible dataframe (missing required predictor)
    incompatible_df = pd.DataFrame({
        "attendance_percentage": [80.0],
        "unknown_column": [123],
    })
    is_valid, err = validate_loaded_model_schema(bundle, incompatible_df)
    assert is_valid is False
    assert "Incompatible feature schema" in err
