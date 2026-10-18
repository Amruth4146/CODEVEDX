"""
Unit tests for data preprocessing, quality inspection, and leakage-free pipeline.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.preprocessing import (
    detect_missing_values,
    detect_duplicates,
    validate_numerical_ranges,
    detect_outliers_iqr,
    inspect_data_quality,
    remove_duplicates,
    handle_invalid_boundaries,
    exclude_missing_target,
    clean_data,
    generate_cleaning_report,
    encode_features,
    split_features_target,
    build_preprocessing_pipeline,
    prepare_training_data,
    prepare_inference_features,
)


# ============================================================================
# 1. DATA QUALITY INSPECTION TESTS
# ============================================================================

def test_detect_missing_values():
    df = pd.DataFrame({
        "attendance_percentage": [85.0, np.nan, 90.0, np.nan],
        "previous_exam_marks": [70.0, 75.0, np.nan, 80.0],
        "daily_study_hours": [3.0, 4.0, 5.0, 6.0],
    })
    res = detect_missing_values(df)
    assert res["has_missing"] is True
    assert res["total_missing"] == 3
    assert res["columns"]["attendance_percentage"]["missing_count"] == 2
    assert res["columns"]["previous_exam_marks"]["missing_count"] == 1
    assert res["columns"]["daily_study_hours"]["missing_count"] == 0


def test_detect_duplicates():
    df = pd.DataFrame({
        "student_id": ["STU-1", "STU-2", "STU-1"],
        "attendance_percentage": [80.0, 90.0, 80.0],
        "daily_study_hours": [4.0, 5.0, 4.0],
    })
    res = detect_duplicates(df)
    assert res["has_duplicates"] is True
    assert res["duplicate_count"] == 1
    assert 2 in res["duplicate_indices"]


def test_validate_numerical_ranges_detects_violations():
    df = pd.DataFrame({
        "attendance_percentage": [110.0, -5.0, 85.0],   # 2 invalid
        "previous_exam_marks": [105.0, 50.0, -10.0],    # 2 invalid
        "daily_study_hours": [-2.0, 4.5, 6.0],          # 1 negative
        "final_exam_marks": [120.0, -15.0, 75.0],       # 2 invalid
    })
    res = validate_numerical_ranges(df)
    assert res["total_violations"] == 7
    assert res["violations"]["attendance_percentage"]["count"] == 2
    assert res["violations"]["previous_exam_marks"]["count"] == 2
    assert res["violations"]["daily_study_hours"]["count"] == 1
    assert res["violations"]["final_exam_marks"]["count"] == 2


def test_detect_outliers_iqr():
    # Regular values around 50, with extreme outliers 10 and 200
    df = pd.DataFrame({
        "attendance_percentage": [50.0, 52.0, 51.0, 49.0, 53.0, 50.0, 51.0, 200.0, 5.0]
    })
    res = detect_outliers_iqr(df, factor=1.5)
    att_res = res["columns"]["attendance_percentage"]
    assert att_res["outlier_count"] >= 1
    assert "summary_df" in res
    assert not res["summary_df"].empty


def test_inspect_data_quality():
    df = pd.DataFrame({
        "attendance_percentage": [85.0, 90.0, 85.0],
        "previous_exam_marks": [70.0, np.nan, 70.0],
        "daily_study_hours": [-1.0, 4.0, 3.0],
    })
    report = inspect_data_quality(df)
    assert "missing_report" in report
    assert "duplicate_report" in report
    assert "range_report" in report
    assert "outlier_report" in report
    assert 0.0 <= report["health_score"] <= 100.0


# ============================================================================
# 2. MISSING VALUE & DUPLICATE HANDLING TESTS
# ============================================================================

def test_remove_duplicates():
    df = pd.DataFrame({
        "student_id": ["STU-1", "STU-2", "STU-1"],
        "score": [90, 80, 90],
    })
    cleaned, dropped = remove_duplicates(df)
    assert len(cleaned) == 2
    assert dropped == 1


def test_handle_invalid_boundaries():
    df = pd.DataFrame({
        "attendance_percentage": [120.0, -10.0, 85.0],
        "daily_study_hours": [-3.0, 5.0, 30.0],
        "previous_exam_marks": [150.0, 40.0, -5.0],
    })
    cleaned, corrections = handle_invalid_boundaries(df)
    assert cleaned["attendance_percentage"].max() <= 100.0
    assert cleaned["attendance_percentage"].min() >= 0.0
    assert cleaned["daily_study_hours"].min() >= 0.0
    assert cleaned["daily_study_hours"].max() <= 24.0
    assert cleaned["previous_exam_marks"].max() <= 100.0
    assert cleaned["previous_exam_marks"].min() >= 0.0
    assert corrections["attendance_percentage"] == 2
    assert corrections["daily_study_hours"] == 2


def test_exclude_missing_target_never_imputes_target():
    df = pd.DataFrame({
        "attendance_percentage": [80.0, 85.0, 90.0, 95.0],
        "previous_exam_marks": [70.0, 75.0, 80.0, 85.0],
        "daily_study_hours": [3.0, 4.0, 5.0, 6.0],
        "final_exam_marks": [65.0, np.nan, 85.0, np.nan],
    })
    filtered, dropped = exclude_missing_target(df, target_col="final_exam_marks")
    assert dropped == 2
    assert len(filtered) == 2
    assert filtered["final_exam_marks"].isnull().sum() == 0


def test_clean_data_uses_median_imputation_for_features():
    df = pd.DataFrame({
        "attendance_percentage": [80.0, np.nan, 90.0],
        "previous_exam_marks": [60.0, 70.0, np.nan],
        "daily_study_hours": [3.0, 4.0, 5.0],
        "final_exam_marks": [65.0, 75.0, 85.0],
    })
    cleaned = clean_data(df)
    assert cleaned.isnull().sum().sum() == 0
    # Median of [80.0, 90.0] is 85.0
    assert cleaned.loc[1, "attendance_percentage"] == 85.0
    # Median of [60.0, 70.0] is 65.0
    assert cleaned.loc[2, "previous_exam_marks"] == 65.0


def test_generate_cleaning_report():
    raw_df = pd.DataFrame({
        "col1": [1.0, np.nan, 1.0],
        "col2": [10.0, 20.0, 10.0],
    })
    cleaned_df = clean_data(raw_df)
    rep = generate_cleaning_report(raw_df, cleaned_df)
    assert rep["initial_rows"] == 3
    assert rep["cleaned_rows"] == 2
    assert rep["duplicates_removed"] == 1
    assert rep["total_missing_before"] == 1
    assert rep["total_missing_after"] == 0


# ============================================================================
# 3. FEATURE SCALING & DATA LEAKAGE PREVENTION TESTS
# ============================================================================

def test_build_preprocessing_pipeline():
    pipeline = build_preprocessing_pipeline(scale_features=True)
    assert isinstance(pipeline, Pipeline)
    assert "imputer" in pipeline.named_steps
    assert "scaler" in pipeline.named_steps


def test_prepare_training_data_prevents_data_leakage():
    # Construct synthetic dataset with intentional missing values and out-of-range target
    df = pd.DataFrame({
        "student_id": [f"STU-{i}" for i in range(20)],
        "attendance_percentage": [80.0 + (i % 15) for i in range(20)],
        "previous_exam_marks": [65.0 + (i % 25) for i in range(20)],
        "daily_study_hours": [3.0 + (i * 0.2) for i in range(20)],
        "assignments_completed": [8 for _ in range(20)],
        "extracurricular_activities": ["Yes" if i % 2 == 0 else "No" for i in range(20)],
        "sleep_hours": [7.0 for _ in range(20)],
        "final_exam_marks": [60.0 + (i * 1.5) for i in range(20)],
    })
    # Set one target value to NaN to ensure row exclusion
    df.loc[19, "final_exam_marks"] = np.nan

    X_train, X_test, y_train, y_test, pipeline, features = prepare_training_data(
        df, test_size=0.25, random_state=42, scale_features=True
    )

    # Missing target row (19) was excluded, so 19 rows total
    assert len(X_train) + len(X_test) == 19
    # Verify pipeline was fitted
    assert hasattr(pipeline.named_steps["scaler"], "mean_")
    # Features match expected feature names
    assert "attendance_percentage" in features
    assert "student_id" not in features
    # Check transformed train data has mean close to 0 and std close to 1
    train_mean = np.mean(X_train, axis=0)
    assert np.all(np.abs(train_mean) < 1e-10)


def test_prepare_inference_features():
    row = prepare_inference_features(
        attendance_percentage=95.0,
        previous_exam_marks=85.0,
        daily_study_hours=5.5,
        assignments_completed=10,
        extracurricular_activities="Yes",
        sleep_hours=8.0,
    )
    assert isinstance(row, pd.DataFrame)
    assert len(row) == 1
    assert row["attendance_percentage"].iloc[0] == 95.0
    assert row["extracurricular_activities"].iloc[0] == 1
