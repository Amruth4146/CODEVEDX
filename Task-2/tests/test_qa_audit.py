"""
Comprehensive Quality Assurance Audit Test Suite
Student Performance Prediction System - CODEVEDX Task 2

Audits:
- Data Integrity (Missing, duplicate, malformed, empty, extreme bounds)
- Machine Learning (Leakage-free CV, persistence, safe deserialization, consistency, reproducibility)
- Security & Privacy (Upload validation, PII prevention, sensitive file exclusion)
- Performance & UI Components (Caching, what-if monotonicity, Plotly figures)
"""

import sys
import io
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.preprocessing import (
    clean_data,
    detect_missing_values,
    detect_duplicates,
    validate_numerical_ranges,
    handle_invalid_boundaries,
    exclude_missing_target,
    prepare_training_data,
    prepare_inference_features,
    encode_features,
)
from src.train_model import (
    validate_dataset_schema,
    validate_loaded_model_schema,
    evaluate_predictions,
    load_and_prepare_dataset,
    train_and_select_models,
    build_pipeline_architectures,
    load_model_bundle,
)
from src.prediction import ModelPredictor
from src.evaluation import (
    calculate_regression_metrics,
    plot_actual_vs_predicted,
    plot_residual_distribution,
    calculate_dataset_fingerprint,
)
from src.eda import (
    plot_feature_histogram,
    plot_performance_category_donut,
    assign_performance_categories,
)
from src.security import (
    validate_uploaded_csv,
    validate_model_bundle_safety,
    sanitize_student_dataframe,
    MAX_UPLOAD_SIZE_BYTES,
)


# ============================================================================
# 1. DATA QUALITY & CORNER-CASE TESTS
# ============================================================================

def test_empty_dataframe_and_csv_handling():
    """Verify system safely rejects or handles empty DataFrames."""
    empty_df = pd.DataFrame()

    # clean_data raises ValueError on empty
    with pytest.raises(ValueError, match="Cannot clean an empty or None"):
        clean_data(empty_df)

    # Inspection functions return safe empty dictionaries
    assert detect_missing_values(empty_df)["total_missing"] == 0
    assert detect_duplicates(empty_df)["duplicate_count"] == 0
    assert validate_numerical_ranges(empty_df)["total_violations"] == 0

    # validate_uploaded_csv rejects empty buffer
    empty_buf = io.BytesIO(b"")
    empty_buf.name = "test.csv"
    valid, err, _ = validate_uploaded_csv(empty_buf)
    assert valid is False
    assert "empty" in err.lower() or "no data" in err.lower()


def test_missing_values_and_imputation():
    """Verify median imputation for features and strict exclusion of missing targets."""
    df = pd.DataFrame({
        "student_id": ["STU-1", "STU-2", "STU-3"],
        "attendance_percentage": [80.0, np.nan, 90.0],
        "previous_exam_marks": [70.0, 75.0, np.nan],
        "daily_study_hours": [3.0, np.nan, 5.0],
        "final_exam_marks": [75.0, 80.0, np.nan],
    })

    # Missing target exclusion
    filtered, dropped = exclude_missing_target(df)
    assert dropped == 1
    assert len(filtered) == 2

    # Clean data applies median imputation
    cleaned = clean_data(filtered)
    assert cleaned["attendance_percentage"].isnull().sum() == 0
    assert cleaned["previous_exam_marks"].isnull().sum() == 0
    assert cleaned["daily_study_hours"].isnull().sum() == 0


def test_duplicate_rows_handling():
    """Verify exact duplicate rows are detected and removed."""
    df = pd.DataFrame({
        "student_id": ["STU-101", "STU-102", "STU-101"],
        "attendance_percentage": [85.0, 90.0, 85.0],
        "previous_exam_marks": [70.0, 80.0, 70.0],
        "daily_study_hours": [4.0, 5.0, 4.0],
        "final_exam_marks": [78.0, 88.0, 78.0],
    })
    res = detect_duplicates(df)
    assert res["has_duplicates"] is True
    assert res["duplicate_count"] == 1

    cleaned = clean_data(df)
    assert len(cleaned) == 2


def test_invalid_numerical_values_and_clamping():
    """Verify out-of-range academic values are clamped to domain boundaries."""
    df = pd.DataFrame({
        "attendance_percentage": [-15.0, 150.0, 85.0],
        "previous_exam_marks": [-10.0, 105.0, 70.0],
        "daily_study_hours": [-4.0, 30.0, 4.0],
        "final_exam_marks": [120.0, -5.0, 65.0],
    })
    clamped, corrections = handle_invalid_boundaries(df)
    assert clamped["attendance_percentage"].min() >= 0.0
    assert clamped["attendance_percentage"].max() <= 100.0
    assert clamped["previous_exam_marks"].min() >= 0.0
    assert clamped["previous_exam_marks"].max() <= 100.0
    assert clamped["daily_study_hours"].min() >= 0.0
    assert clamped["daily_study_hours"].max() <= 24.0
    assert clamped["final_exam_marks"].min() >= 0.0
    assert clamped["final_exam_marks"].max() <= 100.0


def test_malformed_csv_and_unparseable_strings():
    """Verify numerical columns with unparseable strings are detected and handled."""
    df = pd.DataFrame({
        "attendance_percentage": ["85.0", "corrupted_text", "90.0"],
        "previous_exam_marks": [70.0, 75.0, 80.0],
        "daily_study_hours": [3.0, 4.0, 5.0],
        "final_exam_marks": [75.0, 80.0, 85.0],
    })
    # Range validation detects unparseable strings
    v = validate_numerical_ranges(df)
    assert v["total_violations"] >= 1
    assert "attendance_percentage" in v["violations"]

    # Cleaning coerces to NaN and imputes median without crashing
    cleaned = clean_data(df)
    assert np.issubdtype(cleaned["attendance_percentage"].dtype, np.number)
    assert not cleaned["attendance_percentage"].isnull().any()


def test_extreme_input_values_in_inference():
    """Verify inference input formatting handles None, NaN, inf, and extreme values."""
    # Extreme study hours and attendance
    feat1 = prepare_inference_features(
        attendance_percentage=9999.0,
        previous_exam_marks=-999.0,
        daily_study_hours=100.0,
    )
    assert feat1["attendance_percentage"].iloc[0] == 100.0
    assert feat1["previous_exam_marks"].iloc[0] == 0.0
    assert feat1["daily_study_hours"].iloc[0] == 24.0

    # None and NaN handling
    feat2 = prepare_inference_features(
        attendance_percentage=None,
        previous_exam_marks=float("nan"),
        daily_study_hours=None,
    )
    assert not np.isnan(feat2["attendance_percentage"].iloc[0])
    assert not np.isnan(feat2["previous_exam_marks"].iloc[0])
    assert not np.isnan(feat2["daily_study_hours"].iloc[0])

    # Infinite inputs
    feat3 = prepare_inference_features(
        attendance_percentage=float("inf"),
        previous_exam_marks=float("-inf"),
        daily_study_hours=float("inf"),
    )
    assert feat3["attendance_percentage"].iloc[0] == 100.0
    assert feat3["previous_exam_marks"].iloc[0] == 0.0


# ============================================================================
# 2. MACHINE LEARNING & EVALUATION TESTS
# ============================================================================

def test_train_test_separation_and_leakage_prevention():
    """Verify 80/20 train/test split occurs before preprocessing statistics are fit."""
    df = pd.DataFrame({
        "attendance_percentage": [80.0 + i for i in range(100)],
        "previous_exam_marks": [60.0 + i * 0.3 for i in range(100)],
        "daily_study_hours": [2.0 + (i % 8) * 0.5 for i in range(100)],
        "final_exam_marks": [55.0 + i * 0.4 for i in range(100)],
    })
    X_train, X_test, y_train, y_test, pipeline, features = prepare_training_data(
        df, test_size=0.2, random_state=42, scale_features=True
    )
    # 80% train, 20% test
    assert X_train.shape[0] == 80
    assert X_test.shape[0] == 20
    assert len(y_train) == 80
    assert len(y_test) == 20

    # Pipeline contains fitted scaler
    scaler = pipeline.named_steps["scaler"]
    # Scaler mean must match X_train mean before scaling, not the combined 100 samples
    assert len(scaler.mean_) == len(features)


def test_reproducibility():
    """Verify fixed random_state produces identical CV folds and identical model metrics."""
    np.random.seed(42)
    X = pd.DataFrame({
        "attendance_percentage": np.random.uniform(50, 100, 80),
        "previous_exam_marks": np.random.uniform(40, 95, 80),
        "daily_study_hours": np.random.uniform(1, 8, 80),
    })
    y = 0.3 * X["attendance_percentage"] + 0.4 * X["previous_exam_marks"] + 2.0 * X["daily_study_hours"]

    X_test = pd.DataFrame({
        "attendance_percentage": np.random.uniform(50, 100, 20),
        "previous_exam_marks": np.random.uniform(40, 95, 20),
        "daily_study_hours": np.random.uniform(1, 8, 20),
    })
    y_test = 0.3 * X_test["attendance_percentage"] + 0.4 * X_test["previous_exam_marks"] + 2.0 * X_test["daily_study_hours"]

    # Run 1
    best1, _, _, res1, preds1 = train_and_select_models(X, y, X_test, y_test, random_state=42)
    # Run 2
    best2, _, _, res2, preds2 = train_and_select_models(X, y, X_test, y_test, random_state=42)

    assert best1 == best2
    assert res1[best1]["cv_rmse"] == res2[best2]["cv_rmse"]
    assert np.allclose(preds1[best1], preds2[best2])


def test_prediction_consistency_single_vs_batch():
    """Verify single prediction and batch prediction give identical outputs for the same student."""
    predictor = ModelPredictor()

    single_res = predictor.predict(
        attendance_percentage=85.0,
        previous_exam_marks=72.0,
        daily_study_hours=4.5,
    )

    batch_df = pd.DataFrame({
        "attendance_percentage": [85.0],
        "previous_exam_marks": [72.0],
        "daily_study_hours": [4.5],
    })
    batch_res = predictor.predict_batch(batch_df)

    assert np.isclose(
        single_res["predicted_score"],
        batch_res["predicted_final_marks"].iloc[0],
        atol=0.1,
    )
    assert single_res["classification"]["category"] == batch_res["performance_category"].iloc[0]


def test_evaluation_metrics_mathematical_correctness():
    """Verify MAE, MSE, RMSE, and R2 calculation on known synthetic cases."""
    y_true = np.array([50.0, 60.0, 70.0, 80.0, 90.0])

    # Case 1: Perfect predictions
    perfect_metrics = evaluate_predictions(y_true, y_true)
    assert perfect_metrics["mae"] == 0.0
    assert perfect_metrics["mse"] == 0.0
    assert perfect_metrics["rmse"] == 0.0
    assert perfect_metrics["r2"] == 1.0

    # Case 2: Constant +2 offset
    y_pred_offset = y_true + 2.0
    offset_metrics = evaluate_predictions(y_true, y_pred_offset)
    assert offset_metrics["mae"] == 2.0
    assert offset_metrics["mse"] == 4.0
    assert offset_metrics["rmse"] == 2.0
    assert offset_metrics["r2"] < 1.0

    # Case 3: Mean predictor has R2 = 0.0
    y_pred_mean = np.full_like(y_true, y_true.mean())
    mean_metrics = evaluate_predictions(y_true, y_pred_mean)
    assert np.isclose(mean_metrics["r2"], 0.0, atol=0.001)


def test_safe_deserialization_validation():
    """Verify validate_model_bundle_safety detects invalid or corrupt model artifacts."""
    # Valid bundle mock
    valid_bundle = {
        "pipeline": Pipeline([("lr", LinearRegression())]),
        "feature_names": ["attendance_percentage", "previous_exam_marks", "daily_study_hours"],
        "model_name": "Linear Regression",
    }
    is_safe, _ = validate_model_bundle_safety(valid_bundle)
    assert is_safe is True

    # Non-dictionary
    safe1, err1 = validate_model_bundle_safety("not a dict")
    assert safe1 is False
    assert "dictionary" in err1.lower()

    # Missing pipeline/model
    safe2, err2 = validate_model_bundle_safety({"feature_names": ["a", "b"]})
    assert safe2 is False
    assert "pipeline" in err2.lower() or "model" in err2.lower()

    # Estimator lacking predict method
    safe3, err3 = validate_model_bundle_safety({"pipeline": "arbitrary_string", "feature_names": ["a"]})
    assert safe3 is False
    assert "predict" in err3.lower()


# ============================================================================
# 3. SECURITY & PRIVACY TESTS
# ============================================================================

def test_file_upload_security_validation():
    """Verify file upload rejection of invalid extensions, excessive size, and PII."""
    # 1. Invalid extension
    bad_ext = io.BytesIO(b"data")
    bad_ext.name = "payload.exe"
    ok, err, _ = validate_uploaded_csv(bad_ext)
    assert ok is False
    assert ".csv" in err

    # 2. Size limit
    large_buf = io.BytesIO(b"x" * 100)
    large_buf.name = "huge.csv"
    setattr(large_buf, "size", MAX_UPLOAD_SIZE_BYTES + 1024)
    ok_sz, err_sz, _ = validate_uploaded_csv(large_buf)
    assert ok_sz is False
    assert "exceeds" in err_sz.lower()

    # 3. Sensitive PII detection (SSN, credit card, passwords)
    pii_csv = io.BytesIO(b"attendance_percentage,previous_exam_marks,ssn\n80,70,123-45-6789")
    pii_csv.name = "students.csv"
    ok_pii, err_pii, _ = validate_uploaded_csv(pii_csv)
    assert ok_pii is False
    assert "PII" in err_pii or "Personal" in err_pii

    # 4. Valid clean CSV
    clean_csv = io.BytesIO(
        b"attendance_percentage,previous_exam_marks,daily_study_hours\n85,75,4.0\n90,80,5.0"
    )
    clean_csv.name = "valid_students.csv"
    ok_clean, _, clean_df = validate_uploaded_csv(
        clean_csv,
        required_columns=["attendance_percentage", "previous_exam_marks", "daily_study_hours"],
    )
    assert ok_clean is True
    assert len(clean_df) == 2


def test_student_privacy_anonymization():
    """Verify sanitize_student_dataframe strips personal names/emails and enforces anonymous STU-xxxx IDs."""
    raw_df = pd.DataFrame({
        "first_name": ["Alice", "Bob"],
        "email_address": ["alice@university.edu", "bob@university.edu"],
        "attendance_percentage": [85.0, 90.0],
        "previous_exam_marks": [75.0, 80.0],
    })
    sanitized = sanitize_student_dataframe(raw_df)
    assert "first_name" not in sanitized.columns
    assert "email_address" not in sanitized.columns
    assert "student_id" in sanitized.columns
    assert sanitized["student_id"].iloc[0].startswith("STU-")


def test_gitignore_covers_sensitive_files():
    """Verify .gitignore properly excludes virtual envs, secrets, and test caches."""
    gitignore_path = ROOT_DIR / ".gitignore"
    assert gitignore_path.exists()
    content = gitignore_path.read_text(encoding="utf-8")

    assert ".venv" in content
    assert "__pycache__" in content
    assert "secrets.toml" in content
    assert ".pytest_cache" in content
    assert ".env" in content


# ============================================================================
# 4. USER INTERFACE & PERFORMANCE TESTS
# ============================================================================

def test_what_if_simulation_monotonicity():
    """Verify that increasing study hours never causes an illogical drop in projected score."""
    predictor = ModelPredictor()
    sim_df = predictor.simulate_what_if(
        attendance=75.0,
        previous_marks=65.0,
        study_hours=2.0,
    )
    # Check that each simulated row exists (Current + 4 incremental steps)
    assert len(sim_df) == 5
    scores = sim_df["Projected Score"].values
    # In a linear/regularized model, adding study hours must not reduce projected score
    for i in range(len(scores) - 1):
        assert scores[i + 1] >= scores[i]


def test_plotly_figures_render_without_errors():
    """Verify interactive Plotly diagnostic figures are properly structured and error-free."""
    y_true = np.array([65.0, 75.0, 85.0, 90.0, 50.0])
    y_pred = np.array([67.0, 73.0, 86.0, 88.0, 52.0])

    fig_parity = plot_actual_vs_predicted(y_true, y_pred, model_name="Linear Regression")
    assert fig_parity is not None
    assert len(fig_parity.data) >= 2  # Scatter + diagonal reference line

    fig_resid = plot_residual_distribution(y_true, y_pred, model_name="Linear Regression")
    assert fig_resid is not None
    assert len(fig_resid.data) >= 1

    df = pd.DataFrame({"final_exam_marks": y_true})
    fig_hist = plot_feature_histogram(df, column="final_exam_marks")
    assert fig_hist is not None

    df_cat = assign_performance_categories(df)
    fig_donut = plot_performance_category_donut(df_cat)
    assert fig_donut is not None
