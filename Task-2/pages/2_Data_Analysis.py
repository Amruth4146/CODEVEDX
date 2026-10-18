"""
Page 2: Comprehensive Exploratory Data Analysis & Preprocessing Dashboard
Student Performance Prediction System - CODEVEDX Task 2
"""

import sys
from pathlib import Path
import io

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
import numpy as np

from src.data_loader import load_raw_data
from src.eda import (
    assign_performance_categories,
    compute_kpis,
    compute_summary_stats,
    plot_feature_histogram,
    plot_scatter_with_trend,
    create_correlation_heatmap,
    plot_all_boxplots,
    plot_performance_category_donut,
    generate_seaborn_figure,
    generate_statistical_insights,
    CATEGORY_LABELS,
    get_target_col,
)
from src.preprocessing import (
    inspect_data_quality,
    clean_data,
    generate_cleaning_report,
)
from src.feature_selection import calculate_feature_correlations

st.set_page_config(
    page_title="Exploratory Data Analysis | Student Performance System",
    page_icon="🔬",
    layout="wide",
)

# Apply SaaS Styling
from src.ui_components import inject_custom_css, render_header, render_kpi_card
inject_custom_css()


@st.cache_data
def get_data():
    raw_df = load_raw_data()
    return assign_performance_categories(raw_df)


base_df = get_data()
target_column = get_target_col(base_df)

# ============================================================================
# SIDEBAR: BRANDING & INTERACTIVE FILTERS
# ============================================================================
st.sidebar.markdown("""
<div class="sidebar-branding">
    <div style="display: flex; align-items: center; gap: 0.6rem;">
        <div style="background: linear-gradient(135deg, #4F46E5 0%, #06B6D4 100%); width: 34px; height: 34px; border-radius: 9px; display: flex; align-items: center; justify-content: center; color: white; font-size: 1.1rem; font-weight: 800;">
            ✦
        </div>
        <div>
            <div class="sidebar-title">EduPredict AI</div>
            <div class="sidebar-subtitle">EDA & Statistical Lab</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("### 🎛️ **Cohort Filter Controls**")
st.sidebar.caption("Filter the academic cohort dynamically across all dashboard charts.")

# Theme selector for charts (Clean white/light by default as requested)
chart_theme = st.sidebar.radio(
    "🎨 Chart Visual Theme:",
    options=["light", "dark"],
    index=0,
    format_func=lambda x: "☀️ Clean SaaS White" if x == "light" else "🌙 Slate Dark",
)

# 1. Attendance Range
att_min = float(base_df["attendance_percentage"].min())
att_max = 100.0
att_range = st.sidebar.slider(
    "📅 Attendance Range (%)",
    min_value=att_min,
    max_value=att_max,
    value=(att_min, att_max),
    step=1.0,
    help="Filter students by classroom attendance percentage.",
)

# 2. Previous Marks Range
prev_min = float(base_df["previous_exam_marks"].min())
prev_max = 100.0
prev_range = st.sidebar.slider(
    "📝 Previous Marks Range (0–100)",
    min_value=prev_min,
    max_value=prev_max,
    value=(prev_min, prev_max),
    step=1.0,
    help="Filter students by prior examination performance.",
)

# 3. Study Hours Range
study_min = float(base_df["daily_study_hours"].min())
study_max = float(base_df["daily_study_hours"].max())
study_range = st.sidebar.slider(
    "⏱️ Daily Study Hours Range",
    min_value=study_min,
    max_value=study_max,
    value=(study_min, study_max),
    step=0.5,
    help="Filter students by reported daily self-study hours.",
)

# 4. Performance Category Filter
selected_cats = st.sidebar.multiselect(
    "🏷️ Performance Categories:",
    options=CATEGORY_LABELS,
    default=CATEGORY_LABELS,
    help="Select one or more performance standing bands.",
)

# Apply Filters
filtered_df = base_df[
    (base_df["attendance_percentage"] >= att_range[0])
    & (base_df["attendance_percentage"] <= att_range[1])
    & (base_df["previous_exam_marks"] >= prev_range[0])
    & (base_df["previous_exam_marks"] <= prev_range[1])
    & (base_df["daily_study_hours"] >= study_range[0])
    & (base_df["daily_study_hours"] <= study_range[1])
    & (base_df["performance_category"].isin(selected_cats))
].copy()

# Cohort Filter Stats in Sidebar
st.sidebar.markdown("---")
cohort_pct = (len(filtered_df) / max(1, len(base_df))) * 100
st.sidebar.markdown(f"**Cohort Filtered:** `{len(filtered_df)}` of `{len(base_df)}` ({cohort_pct:.1f}%)")

if st.sidebar.button("🔄 Reset All Filters", use_container_width=True):
    st.rerun()

# ============================================================================
# MAIN DASHBOARD HEADER & KPI CARDS
# ============================================================================
render_header(
    title="Exploratory Data Analysis & Statistical Lab",
    subtitle="Multi-dimensional statistical insights, correlation matrices, feature distributions, and student academic standing.",
    badge_text="STATISTICAL EXPLORATION & EDA",
    badge_variant="cyan",
    status_pill=f"Active Cohort: {len(filtered_df)} / {len(base_df)} Students",
)

# Performance Category Disclaimer
st.info(
    "💡 **Methodological Note on Categories:** The performance tiers (*Excellent: 85–100*, *Good: 70–84.99*, "
    "*Average: 50–69.99*, *Needs Improvement: <50*) are application-defined diagnostic benchmarks established for this analysis "
    "and do not constitute official institutional grading classifications."
)

if filtered_df.empty:
    st.warning("⚠️ No student records match the active filter criteria. Please broaden your filter sliders in the sidebar.")
    st.stop()

# 1. KPI CARDS ROW (Reflecting Filtered Dataset)
kpis = compute_kpis(filtered_df, target_col=target_column)

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(
        render_kpi_card(
            label="Total Students",
            value=f"{kpis['total_students']:,}",
            subtitle="Filtered Sample Size",
            color="#4F46E5",
            icon="👥",
            badge_chip="Sample",
        ),
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        render_kpi_card(
            label="Average Attendance",
            value=f"{kpis['avg_attendance']:.1f}%",
            subtitle="Campus Presence Mean",
            color="#06B6D4",
            icon="📅",
            badge_chip="Presence",
        ),
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        render_kpi_card(
            label="Avg Previous Marks",
            value=f"{kpis['avg_previous_marks']:.1f}",
            subtitle="Historical Score Average",
            color="#10B981",
            icon="📝",
            badge_chip="Historical",
        ),
        unsafe_allow_html=True,
    )

with c4:
    st.markdown(
        render_kpi_card(
            label="Avg Study Hours",
            value=f"{kpis['avg_study_hours']:.1f} hrs",
            subtitle="Daily Study Dedication",
            color="#F59E0B",
            icon="⏱️",
            badge_chip="Dedication",
        ),
        unsafe_allow_html=True,
    )

with c5:
    st.markdown(
        render_kpi_card(
            label="Avg Final Score",
            value=f"{kpis['avg_final_score']:.1f}",
            subtitle="Exam Score Benchmark",
            color="#10B981",
            icon="🎯",
            badge_chip="Outcome",
        ),
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)


# ============================================================================
# MULTI-TAB EXPLORATION DASHBOARD
# ============================================================================
tab_overview, tab_scatter, tab_corr_box, tab_insights, tab_quality = st.tabs([
    "📊 Distributions & Categorization",
    "📈 Bivariate Relationships (Scatter)",
    "🔥 Correlation & Outlier Box Plots",
    "💡 Statistical Insights & Findings",
    "🧹 Preprocessing & Quality Pipeline",
])


# ----------------------------------------------------------------------------
# TAB 1: DISTRIBUTIONS & PERFORMANCE CATEGORIES
# ----------------------------------------------------------------------------
with tab_overview:
    st.markdown("### 📊 **Cohort Distributions & Performance Standing**")

    # Donut chart for performance category
    col_donut, col_desc = st.columns([1.3, 1.2], gap="large")
    with col_donut:
        fig_cat = plot_performance_category_donut(filtered_df, theme=chart_theme)
        st.plotly_chart(fig_cat, use_container_width=True, key="eda_donut_cat")

    with col_desc:
        st.markdown("#### 📋 **Category Breakdown Summary**")
        cat_counts = filtered_df["performance_category"].value_counts().reindex(CATEGORY_LABELS, fill_value=0)
        breakdown_rows = []
        for cat in CATEGORY_LABELS:
            cnt = int(cat_counts[cat])
            pct = (cnt / len(filtered_df)) * 100 if len(filtered_df) > 0 else 0.0
            breakdown_rows.append({
                "Performance Category": cat,
                "Student Count": cnt,
                "Proportion (%)": f"{pct:.1f}%",
            })
        st.dataframe(pd.DataFrame(breakdown_rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### 📦 **Univariate Feature Distributions**")
    st.caption("Marginal box plots display quartiles, medians, and whisker bounds for each metric.")

    d_col1, d_col2 = st.columns(2)
    with d_col1:
        fig_att = plot_feature_histogram(filtered_df, "attendance_percentage", color="#6366F1", theme=chart_theme)
        st.plotly_chart(fig_att, use_container_width=True, key="eda_hist_att")

        fig_study = plot_feature_histogram(filtered_df, "daily_study_hours", color="#F59E0B", theme=chart_theme)
        st.plotly_chart(fig_study, use_container_width=True, key="eda_hist_study")

    with d_col2:
        fig_prev = plot_feature_histogram(filtered_df, "previous_exam_marks", color="#06B6D4", theme=chart_theme)
        st.plotly_chart(fig_prev, use_container_width=True, key="eda_hist_prev")

        fig_final = plot_feature_histogram(filtered_df, target_column, color="#10B981", theme=chart_theme)
        st.plotly_chart(fig_final, use_container_width=True, key="eda_hist_final")


# ----------------------------------------------------------------------------
# TAB 2: BIVARIATE SCATTER PLOTS & TRENDLINES
# ----------------------------------------------------------------------------
with tab_scatter:
    st.markdown("### 📈 **Feature vs. Final Examination Score Relationships**")
    st.caption("Interactive scatter plots featuring OLS regression trendlines and Pearson correlation coefficients.")

    s_col1, s_col2 = st.columns(2)
    with s_col1:
        # Attendance vs Final
        fig_s1 = plot_scatter_with_trend(filtered_df, x_col="attendance_percentage", y_col=target_column, theme=chart_theme)
        st.plotly_chart(fig_s1, use_container_width=True, key="eda_scatter_att")

        # Previous Marks vs Final
        fig_s2 = plot_scatter_with_trend(filtered_df, x_col="previous_exam_marks", y_col=target_column, theme=chart_theme)
        st.plotly_chart(fig_s2, use_container_width=True, key="eda_scatter_prev")

    with s_col2:
        # Study Hours vs Final
        fig_s3 = plot_scatter_with_trend(filtered_df, x_col="daily_study_hours", y_col=target_column, theme=chart_theme)
        st.plotly_chart(fig_s3, use_container_width=True, key="eda_scatter_study")

        # Assignments Completed vs Final
        if "assignments_completed" in filtered_df.columns:
            fig_s4 = plot_scatter_with_trend(filtered_df, x_col="assignments_completed", y_col=target_column, theme=chart_theme)
            st.plotly_chart(fig_s4, use_container_width=True, key="eda_scatter_assign")

    st.markdown("---")
    # Custom Dynamic Bivariate Explorer
    st.markdown("#### 🔍 **Dynamic Bivariate Pair Explorer**")
    num_cols = [c for c in filtered_df.select_dtypes(include=[np.number]).columns if c != "student_id"]
    bx1, bx2 = st.columns(2)
    with bx1:
        sel_x = st.selectbox("Select Independent Variable (X-Axis):", num_cols, index=0, key="eda_sel_x")
    with bx2:
        sel_y = st.selectbox("Select Dependent Variable (Y-Axis):", num_cols, index=len(num_cols) - 1, key="eda_sel_y")

    fig_custom = plot_scatter_with_trend(filtered_df, x_col=sel_x, y_col=sel_y, theme=chart_theme)
    st.plotly_chart(fig_custom, use_container_width=True, key="eda_scatter_custom")


# ----------------------------------------------------------------------------
# TAB 3: CORRELATION HEATMAP & OUTLIER BOX PLOTS
# ----------------------------------------------------------------------------
with tab_corr_box:
    st.markdown("### 🔥 **Feature Correlation Matrix & Outlier Diagnostics**")

    c_heat, c_rank = st.columns([1.5, 1], gap="large")
    with c_heat:
        fig_heat = create_correlation_heatmap(filtered_df, theme=chart_theme)
        st.plotly_chart(fig_heat, use_container_width=True, key="eda_heatmap_matrix")

    with c_rank:
        st.markdown("#### 🏆 **Target Correlation Rankings**")
        corr_rank_df = calculate_feature_correlations(filtered_df, target_col=target_column)
        st.dataframe(
            corr_rank_df[["formatted_feature", "correlation"]].rename(
                columns={"formatted_feature": "Academic Feature", "correlation": "Pearson r"}
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("---")
    st.markdown("#### 📦 **Side-by-Side Outlier Detection Box Plots**")
    st.caption("Whiskers show 1.5×IQR boundaries; individual points denote statistical outliers.")
    fig_boxes = plot_all_boxplots(filtered_df, theme=chart_theme)
    st.plotly_chart(fig_boxes, use_container_width=True, key="eda_boxplots_matrix")

    # Static Seaborn/Matplotlib Summary
    with st.expander("🎨 **Static Publication-Quality Seaborn / Matplotlib Multi-Panel Figure**", expanded=False):
        st.caption("High-resolution static rendering using Seaborn with Matplotlib layout:")
        fig_sns = generate_seaborn_figure(filtered_df)
        st.pyplot(fig_sns)


# ----------------------------------------------------------------------------
# TAB 4: STATISTICAL INSIGHTS & CAUSATION ADVISORY
# ----------------------------------------------------------------------------
with tab_insights:
    st.markdown("### 💡 **Statistical Insights & Feature Dynamics**")

    insights_list = generate_statistical_insights(filtered_df)
    in1, in2 = st.columns(2, gap="large")

    for idx, item in enumerate(insights_list):
        target_col_box = in1 if idx % 2 == 0 else in2
        with target_col_box:
            is_advisory = item.get("type") == "advisory"
            card_border = "#EF4444" if is_advisory else "#6366F1"
            st.markdown(f"""
            <div class="glass-card" style="border-left: 4px solid {card_border};">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <h4 style="margin: 0; color: #F8FAFC;">{item['title']}</h4>
                    <span class="badge badge-model">{item['metric']}</span>
                </div>
                <p style="color: #CBD5E1; font-size: 0.95rem; line-height: 1.6; margin: 0;">
                    {item['text']}
                </p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📋 **Descriptive Cohort Statistics Table**")
    stats_df = compute_summary_stats(filtered_df)
    st.dataframe(stats_df, use_container_width=True)


# ----------------------------------------------------------------------------
# TAB 5: PREPROCESSING & QUALITY PIPELINE (PRESERVED FUNCTIONALITY)
# ----------------------------------------------------------------------------
with tab_quality:
    st.markdown("### 🔍 **Data Quality Inspection & Preprocessing Engine**")
    st.caption("Inspect raw data integrity, detect missing values/outliers, and generate leakage-free preprocessed datasets.")

    quality_report = inspect_data_quality(filtered_df)
    cleaned_df = clean_data(filtered_df)
    cleaning_rep = generate_cleaning_report(filtered_df, cleaned_df)

    # Health Metric Summary Cards
    h1, h2, h3, h4, h5 = st.columns(5)
    with h1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-val" style="color: #6366F1;">{quality_report['health_score']}%</div>
            <div class="metric-label">Data Health Score</div>
        </div>
        """, unsafe_allow_html=True)

    with h2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-val">{quality_report['total_records']}</div>
            <div class="metric-label">Analyzed Rows</div>
        </div>
        """, unsafe_allow_html=True)

    with h3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-val" style="color: {'#EF4444' if quality_report['missing_report']['has_missing'] else '#10B981'};">
                {quality_report['missing_report']['total_missing']}
            </div>
            <div class="metric-label">Missing Values</div>
        </div>
        """, unsafe_allow_html=True)

    with h4:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-val" style="color: {'#F59E0B' if quality_report['duplicate_report']['has_duplicates'] else '#10B981'};">
                {quality_report['duplicate_report']['duplicate_count']}
            </div>
            <div class="metric-label">Duplicate Records</div>
        </div>
        """, unsafe_allow_html=True)

    with h5:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-val" style="color: {'#EF4444' if quality_report['range_report']['total_violations'] > 0 else '#10B981'};">
                {quality_report['range_report']['total_violations']}
            </div>
            <div class="metric-label">Range Violations</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Download Cleaned Data
    down_buf = io.StringIO()
    cleaned_df.to_csv(down_buf, index=False)
    st.download_button(
        label="📥 Download Cleaned Filtered Dataset (CSV)",
        data=down_buf.getvalue(),
        file_name="cleaned_student_academic_records.csv",
        mime="text/csv",
        use_container_width=True,
    )
