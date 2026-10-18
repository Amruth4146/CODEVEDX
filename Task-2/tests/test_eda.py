"""
Unit tests for Exploratory Data Analysis (EDA) module.
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

from src.eda import (
    assign_performance_categories,
    compute_kpis,
    compute_summary_stats,
    plot_feature_histogram,
    plot_scatter_with_trend,
    create_correlation_heatmap,
    plot_all_boxplots,
    plot_performance_category_donut,
    generate_statistical_insights,
    CATEGORY_LABELS,
)


@pytest.fixture
def sample_academic_df():
    return pd.DataFrame({
        "student_id": ["STU-1", "STU-2", "STU-3", "STU-4"],
        "attendance_percentage": [92.0, 78.0, 65.0, 48.0],
        "previous_exam_marks": [88.0, 72.0, 58.0, 39.0],
        "daily_study_hours": [6.0, 4.5, 3.0, 1.5],
        "assignments_completed": [10, 8, 6, 3],
        "extracurricular_activities": ["Yes", "Yes", "No", "No"],
        "sleep_hours": [7.5, 7.0, 6.5, 6.0],
        "final_exam_marks": [91.0, 76.5, 54.0, 42.0],
    })


def test_assign_performance_categories(sample_academic_df):
    df_cat = assign_performance_categories(sample_academic_df)
    assert "performance_category" in df_cat.columns
    cats = df_cat["performance_category"].tolist()
    # 91.0 -> Excellent (85–100)
    assert cats[0] == "Excellent (85–100)"
    # 76.5 -> Good (70–84.99)
    assert cats[1] == "Good (70–84.99)"
    # 54.0 -> Average (50–69.99)
    assert cats[2] == "Average (50–69.99)"
    # 42.0 -> Needs Improvement (<50)
    assert cats[3] == "Needs Improvement (<50)"


def test_compute_kpis(sample_academic_df):
    kpis = compute_kpis(sample_academic_df)
    assert kpis["total_students"] == 4
    assert kpis["avg_attendance"] == round(sample_academic_df["attendance_percentage"].mean(), 1)
    assert kpis["avg_previous_marks"] == round(sample_academic_df["previous_exam_marks"].mean(), 1)
    assert kpis["avg_study_hours"] == round(sample_academic_df["daily_study_hours"].mean(), 1)
    assert kpis["avg_final_score"] == round(sample_academic_df["final_exam_marks"].mean(), 1)


def test_compute_summary_stats(sample_academic_df):
    stats = compute_summary_stats(sample_academic_df)
    assert isinstance(stats, pd.DataFrame)
    assert "mean" in stats.columns
    assert "std" in stats.columns
    assert "skewness" in stats.columns


def test_plot_feature_histogram(sample_academic_df):
    fig = plot_feature_histogram(sample_academic_df, "attendance_percentage")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


def test_plot_scatter_with_trend(sample_academic_df):
    fig = plot_scatter_with_trend(sample_academic_df, x_col="daily_study_hours", y_col="final_exam_marks")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


def test_create_correlation_heatmap(sample_academic_df):
    fig = create_correlation_heatmap(sample_academic_df)
    assert isinstance(fig, go.Figure)


def test_plot_all_boxplots(sample_academic_df):
    fig = plot_all_boxplots(sample_academic_df)
    assert isinstance(fig, go.Figure)


def test_plot_performance_category_donut(sample_academic_df):
    fig = plot_performance_category_donut(sample_academic_df)
    assert isinstance(fig, go.Figure)


def test_generate_statistical_insights_includes_causation_advisory(sample_academic_df):
    insights = generate_statistical_insights(sample_academic_df)
    assert len(insights) >= 3
    # Verify presence of causation advisory
    advisory_found = any("Causation" in item["title"] or "advisory" in item.get("type", "") for item in insights)
    assert advisory_found is True
