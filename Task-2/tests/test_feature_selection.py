"""
Unit tests for Feature Selection and Feature Importance Analysis Module.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.feature_selection import (
    resolve_target_and_features,
    compute_correlation_analysis,
    compute_mutual_information,
    compute_rf_importance,
    generate_feature_comparison_table,
    plot_feature_importance_comparison_chart,
    cross_validate_feature_selection,
    get_selected_feature_summary,
    get_educational_explanation,
)


@pytest.fixture
def mock_dataset():
    np.random.seed(42)
    n = 60
    study = np.random.uniform(1.0, 9.0, n)
    prev = np.random.uniform(40.0, 95.0, n)
    att = np.random.uniform(50.0, 100.0, n)
    extra = np.random.choice(["Yes", "No"], n)
    # Target linearly dependent on study, prev, att with small noise
    target = 0.4 * prev + 0.3 * att + 3.0 * study + np.random.normal(0, 2.0, n)

    return pd.DataFrame({
        "student_id": [f"TEST-{i}" for i in range(n)],
        "attendance_percentage": att,
        "previous_exam_marks": prev,
        "daily_study_hours": study,
        "extracurricular_activities": extra,
        "final_exam_marks": np.clip(target, 0.0, 100.0),
    })


def test_resolve_target_and_features_strictly_excludes_target(mock_dataset):
    X, y, target_col, features = resolve_target_and_features(mock_dataset)
    assert target_col == "final_exam_marks"
    # Target must never be in input features
    assert "final_exam_marks" not in X.columns
    assert "final_exam_score" not in X.columns
    assert "student_id" not in X.columns
    assert len(X) == len(mock_dataset)
    assert len(y) == len(mock_dataset)


def test_compute_correlation_analysis(mock_dataset):
    X, y, _, _ = resolve_target_and_features(mock_dataset)
    corr_df = compute_correlation_analysis(X, y)
    assert "pearson_r" in corr_df.columns
    assert "spearman_rho" in corr_df.columns
    assert "abs_pearson" in corr_df.columns
    assert "correlation_rank" in corr_df.columns
    # Check descending order
    assert corr_df["abs_pearson"].is_monotonic_decreasing


def test_compute_mutual_information(mock_dataset):
    X, y, _, _ = resolve_target_and_features(mock_dataset)
    mi_df = compute_mutual_information(X, y, random_state=42)
    assert "mi_score" in mi_df.columns
    assert "mi_percentage" in mi_df.columns
    assert "mi_rank" in mi_df.columns
    assert (mi_df["mi_score"] >= 0).all()
    # Sum of percentages should be ~ 100
    assert np.isclose(mi_df["mi_percentage"].sum(), 100.0, atol=1.0)


def test_compute_rf_importance(mock_dataset):
    X, y, _, _ = resolve_target_and_features(mock_dataset)
    rf_df = compute_rf_importance(X, y, random_state=42)
    assert "rf_importance" in rf_df.columns
    assert "rf_percentage" in rf_df.columns
    assert (rf_df["rf_importance"] >= 0).all()
    assert np.isclose(rf_df["rf_percentage"].sum(), 100.0, atol=1.0)


def test_generate_feature_comparison_table(mock_dataset):
    X, y, _, _ = resolve_target_and_features(mock_dataset)
    comp_df = generate_feature_comparison_table(X, y, random_state=42)
    assert "avg_rank" in comp_df.columns
    assert "recommendation" in comp_df.columns
    assert len(comp_df) == X.shape[1]
    # Check that recommendation contains Retain or Optional
    assert comp_df["recommendation"].str.contains("Retain|Optional").all()


def test_plot_feature_importance_comparison_chart(mock_dataset):
    X, y, _, _ = resolve_target_and_features(mock_dataset)
    comp_df = generate_feature_comparison_table(X, y, random_state=42)
    fig = plot_feature_importance_comparison_chart(comp_df)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 3  # RF, MI, Pearson traces


def test_cross_validate_feature_selection_leakage_free(mock_dataset):
    X, y, _, _ = resolve_target_and_features(mock_dataset)
    cv_res = cross_validate_feature_selection(X, y, k_features=2, n_splits=3, random_state=42)
    assert "baseline" in cv_res
    assert "selected" in cv_res
    assert "r2_mean" in cv_res["baseline"]
    assert "r2_mean" in cv_res["selected"]
    assert cv_res["k_features"] == 2


def test_get_educational_explanation():
    edu = get_educational_explanation()
    assert "what_is_it" in edu
    assert "why_it_matters" in edu
    assert "non_linear_note" in edu
    assert "never_discard_blindly" in edu
    assert "leakage_warning" in edu
