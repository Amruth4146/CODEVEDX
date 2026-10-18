"""
Exploratory Data Analysis (EDA) module for Student Performance Prediction System.
Provides interactive Plotly and Seaborn/Matplotlib visualizations, cohort statistical summaries,
configurable performance categorization, and educational insights on correlation vs. causation.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

# Category definitions specified in requirements
CATEGORY_BINS = [-np.inf, 49.999, 69.999, 84.999, np.inf]
CATEGORY_LABELS = [
    "Needs Improvement (<50)",
    "Average (50–69.99)",
    "Good (70–84.99)",
    "Excellent (85–100)",
]

CATEGORY_COLORS = {
    "Excellent (85–100)": "#10B981",       # Emerald Green
    "Good (70–84.99)": "#06B6D4",          # Cyan / Sky
    "Average (50–69.99)": "#F59E0B",       # Amber / Orange
    "Needs Improvement (<50)": "#EF4444",  # Crimson / Red
}

THEME_TEMPLATES = {
    "light": {
        "plotly_template": "plotly_white",
        "bg_paper": "#FFFFFF",
        "bg_plot": "#F8FAFC",
        "font_color": "#0F172A",
        "grid_color": "#E2E8F0",
    },
    "dark": {
        "plotly_template": "plotly_dark",
        "bg_paper": "#1E293B",
        "bg_plot": "#0F172A",
        "font_color": "#F8FAFC",
        "grid_color": "rgba(148, 163, 184, 0.15)",
    },
}


def get_target_col(df: pd.DataFrame) -> str:
    """Helper to detect target column name."""
    if "final_exam_marks" in df.columns:
        return "final_exam_marks"
    if "final_exam_score" in df.columns:
        return "final_exam_score"
    return "final_exam_marks"


def assign_performance_categories(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Assigns application-defined academic performance categories to the DataFrame:
    - Excellent: 85–100
    - Good: 70–84.99
    - Average: 50–69.99
    - Needs Improvement: Below 50
    """
    col = target_col or get_target_col(df)
    res_df = df.copy()

    if col in res_df.columns:
        res_df["performance_category"] = pd.cut(
            res_df[col],
            bins=CATEGORY_BINS,
            labels=CATEGORY_LABELS,
            right=True,
        )
    else:
        res_df["performance_category"] = "Average (50–69.99)"

    return res_df


def compute_kpis(df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, float]:
    """
    Computes summary KPIs for the filtered dataset:
    - Total Students
    - Average Attendance (%)
    - Average Previous Marks
    - Average Study Hours
    - Average Final Exam Score
    """
    col = target_col or get_target_col(df)
    n = len(df)
    if n == 0:
        return {
            "total_students": 0,
            "avg_attendance": 0.0,
            "avg_previous_marks": 0.0,
            "avg_study_hours": 0.0,
            "avg_final_score": 0.0,
        }

    return {
        "total_students": int(n),
        "avg_attendance": round(float(df["attendance_percentage"].mean()), 1) if "attendance_percentage" in df.columns else 0.0,
        "avg_previous_marks": round(float(df["previous_exam_marks"].mean()), 1) if "previous_exam_marks" in df.columns else 0.0,
        "avg_study_hours": round(float(df["daily_study_hours"].mean()), 1) if "daily_study_hours" in df.columns else 0.0,
        "avg_final_score": round(float(df[col].mean()), 1) if col in df.columns else 0.0,
    }


def compute_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes statistical metrics (mean, std, min, quartiles, max, skewness) for numeric features.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    stats = numeric_df.describe().T
    stats["skewness"] = numeric_df.skew()
    stats["kurtosis"] = numeric_df.kurtosis()
    stats = stats.round(2)
    stats.index = [idx.replace("_", " ").title() for idx in stats.index]
    return stats


def plot_feature_histogram(
    df: pd.DataFrame,
    column: str,
    color: str = "#6366F1",
    theme: str = "light",
    nbins: int = 30,
) -> go.Figure:
    """
    Plots an interactive histogram with marginal box plot for a specified feature.
    """
    theme_cfg = THEME_TEMPLATES.get(theme, THEME_TEMPLATES["light"])
    col_label = column.replace("_", " ").title()

    fig = px.histogram(
        df,
        x=column,
        marginal="box",
        nbins=nbins,
        color_discrete_sequence=[color],
        title=f"Distribution of {col_label}",
    )
    fig.update_layout(
        template=theme_cfg["plotly_template"],
        paper_bgcolor=theme_cfg["bg_paper"],
        plot_bgcolor=theme_cfg["bg_plot"],
        font=dict(color=theme_cfg["font_color"]),
        xaxis_title=col_label,
        yaxis_title="Student Count",
        margin=dict(l=40, r=40, t=50, b=40),
        bargap=0.08,
    )
    return fig


def plot_scatter_with_trend(
    df: pd.DataFrame,
    x_col: str,
    y_col: Optional[str] = None,
    color_by_category: bool = True,
    theme: str = "light",
) -> go.Figure:
    """
    Creates an interactive scatter plot with an OLS regression trendline,
    displaying Pearson correlation coefficient, data point tooltips, and regression stats.
    """
    target = y_col or get_target_col(df)
    theme_cfg = THEME_TEMPLATES.get(theme, THEME_TEMPLATES["light"])
    x_label = x_col.replace("_", " ").title()
    y_label = target.replace("_", " ").title()

    # Calculate Pearson r
    r_val = df[x_col].corr(df[target]) if len(df) > 1 else 0.0

    if color_by_category and "performance_category" in df.columns:
        fig = px.scatter(
            df,
            x=x_col,
            y=target,
            color="performance_category",
            color_discrete_map=CATEGORY_COLORS,
            trendline="ols",
            opacity=0.75,
            title=f"{x_label} vs. {y_label} (Pearson r = {r_val:.2f})",
            hover_data=["student_id"] if "student_id" in df.columns else None,
        )
    else:
        fig = px.scatter(
            df,
            x=x_col,
            y=target,
            trendline="ols",
            opacity=0.75,
            color_discrete_sequence=["#6366F1"],
            title=f"{x_label} vs. {y_label} (Pearson r = {r_val:.2f})",
            hover_data=["student_id"] if "student_id" in df.columns else None,
        )

    fig.update_layout(
        template=theme_cfg["plotly_template"],
        paper_bgcolor=theme_cfg["bg_paper"],
        plot_bgcolor=theme_cfg["bg_plot"],
        font=dict(color=theme_cfg["font_color"]),
        xaxis_title=x_label,
        yaxis_title=y_label,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
    )
    return fig


def create_correlation_heatmap(df: pd.DataFrame, theme: str = "light") -> go.Figure:
    """
    Generates an interactive Plotly heatmap showing Pearson correlation coefficients between numeric features.
    """
    theme_cfg = THEME_TEMPLATES.get(theme, THEME_TEMPLATES["light"])
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr().round(2)

    col_labels = [c.replace("_", " ").title() for c in corr.columns]

    fig = px.imshow(
        corr,
        x=col_labels,
        y=col_labels,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Purples",
        title="Pearson Correlation Heatmap",
        labels=dict(color="Correlation"),
    )
    fig.update_layout(
        template=theme_cfg["plotly_template"],
        paper_bgcolor=theme_cfg["bg_paper"],
        plot_bgcolor=theme_cfg["bg_plot"],
        font=dict(color=theme_cfg["font_color"]),
        title_font_size=16,
        margin=dict(l=40, r=40, t=50, b=40),
        coloraxis_colorbar=dict(title="r"),
    )
    return fig


def plot_all_boxplots(df: pd.DataFrame, theme: str = "light") -> go.Figure:
    """
    Renders box plots for core features side by side for visual outlier inspection.
    """
    theme_cfg = THEME_TEMPLATES.get(theme, THEME_TEMPLATES["light"])
    target = get_target_col(df)
    box_cols = ["attendance_percentage", "previous_exam_marks", "daily_study_hours"]
    if target in df.columns:
        box_cols.append(target)

    # Normalized comparison or individual subplots
    fig = make_subplots(
        rows=1,
        cols=len(box_cols),
        subplot_titles=[c.replace("_", " ").title() for c in box_cols],
    )

    palette = ["#6366F1", "#06B6D4", "#F59E0B", "#10B981"]

    for idx, col in enumerate(box_cols, start=1):
        fig.add_trace(
            go.Box(
                y=df[col],
                name=col.replace("_", " ").title(),
                marker_color=palette[idx - 1],
                boxpoints="outliers",
                jitter=0.2,
                pointpos=-1.8,
            ),
            row=1,
            col=idx,
        )

    fig.update_layout(
        template=theme_cfg["plotly_template"],
        paper_bgcolor=theme_cfg["bg_paper"],
        plot_bgcolor=theme_cfg["bg_plot"],
        font=dict(color=theme_cfg["font_color"]),
        title="Outlier & Quartile Box Plot Matrix",
        showlegend=False,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def plot_performance_category_donut(df: pd.DataFrame, theme: str = "light") -> go.Figure:
    """
    Plots an interactive donut chart of application-defined performance tiers.
    """
    theme_cfg = THEME_TEMPLATES.get(theme, THEME_TEMPLATES["light"])
    cat_df = assign_performance_categories(df) if "performance_category" not in df.columns else df

    counts = cat_df["performance_category"].value_counts()
    # Ensure ordered labels
    counts = counts.reindex(CATEGORY_LABELS, fill_value=0)

    fig = px.pie(
        values=counts.values,
        names=counts.index,
        hole=0.55,
        title="Application-Defined Performance Category Distribution",
        color=counts.index,
        color_discrete_map=CATEGORY_COLORS,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(
        template=theme_cfg["plotly_template"],
        paper_bgcolor=theme_cfg["bg_paper"],
        plot_bgcolor=theme_cfg["bg_plot"],
        font=dict(color=theme_cfg["font_color"]),
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
    )
    return fig


def generate_seaborn_figure(df: pd.DataFrame) -> plt.Figure:
    """
    Generates a multi-panel Seaborn / Matplotlib static dashboard figure
    suitable for export or detailed static reporting.
    """
    target = get_target_col(df)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    sns.set_theme(style="whitegrid", palette="muted")

    # 1. Attendance Distribution (KDE + Hist)
    sns.histplot(df["attendance_percentage"], kde=True, ax=axes[0, 0], color="#6366F1")
    axes[0, 0].set_title("Attendance Percentage Distribution", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Attendance (%)")

    # 2. Previous Marks vs Final Marks (Regression)
    sns.regplot(data=df, x="previous_exam_marks", y=target, ax=axes[0, 1], color="#06B6D4", scatter_kws={"alpha": 0.6})
    axes[0, 1].set_title(f"Previous Marks vs {target.replace('_', ' ').title()}", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Previous Marks")
    axes[0, 1].set_ylabel("Final Score")

    # 3. Daily Study Hours vs Final Marks
    sns.regplot(data=df, x="daily_study_hours", y=target, ax=axes[1, 0], color="#10B981", scatter_kws={"alpha": 0.6})
    axes[1, 0].set_title(f"Daily Study Hours vs {target.replace('_', ' ').title()}", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Daily Study Hours")
    axes[1, 0].set_ylabel("Final Score")

    # 4. Correlation Heatmap
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="Purples", ax=axes[1, 1], cbar=False)
    axes[1, 1].set_title("Correlation Matrix", fontsize=12, fontweight="bold")

    plt.tight_layout()
    return fig


def generate_statistical_insights(df: pd.DataFrame) -> List[Dict[str, str]]:
    """
    Extracts statistical findings, dataset metrics, correlation observations,
    and includes explicit pedagogical guidance that correlation does not imply causation.
    """
    insights = []
    target = get_target_col(df)
    if target not in df.columns or len(df) == 0:
        return insights

    # Correlation checks
    r_att = df["attendance_percentage"].corr(df[target]) if "attendance_percentage" in df.columns else 0.0
    r_prev = df["previous_exam_marks"].corr(df[target]) if "previous_exam_marks" in df.columns else 0.0
    r_study = df["daily_study_hours"].corr(df[target]) if "daily_study_hours" in df.columns else 0.0

    # 1. Study Hours
    insights.append({
        "title": "Study Commitment & Performance",
        "metric": f"r = {r_study:.2f}",
        "text": f"Study hours demonstrate a correlation of r = {r_study:.2f} with final examination marks. Students dedicating above-average study hours show consistently higher grade concentrations.",
        "type": "positive",
    })

    # 2. Previous Marks
    insights.append({
        "title": "Academic Foundation Impact",
        "metric": f"r = {r_prev:.2f}",
        "text": f"Prior academic achievement exhibits a correlation of r = {r_prev:.2f} with semester outcomes, highlighting the predictive value of historical competency.",
        "type": "info",
    })

    # 3. Attendance
    insights.append({
        "title": "Attendance & Engagement",
        "metric": f"r = {r_att:.2f}",
        "text": f"Classroom attendance (r = {r_att:.2f}) correlates positively with performance, emphasizing the role of continuous instructional engagement.",
        "type": "accent",
    })

    # 4. Scientific Advisory: Correlation vs Causation
    insights.append({
        "title": "Methodological Note: Correlation vs. Causation",
        "metric": "Scientific Notice",
        "text": "Observed correlations indicate linear statistical co-movements between features, not direct single-variable causal proofs. Confounding student factors (study discipline, baseline knowledge, resource access) simultaneously shape outcomes.",
        "type": "advisory",
    })

    return insights


# Backward compatibility aliases
plot_performance_categories = plot_performance_category_donut
plot_bivariate_scatter = plot_scatter_with_trend
plot_feature_distribution = plot_feature_histogram
get_academic_insights = generate_statistical_insights

