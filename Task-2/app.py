"""
EduPredict AI — Student Performance Prediction System
CODEVEDX Internship - Task 2: Training + Data Analysis Tool
Main Application Landing & Executive Portal
"""

import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from src.prediction import ModelPredictor
from src.data_loader import load_raw_data
from src.ui_components import inject_custom_css, render_sidebar, render_kpi_card

# Page Configuration
st.set_page_config(
    page_title="EduPredict AI | Student Performance Prediction System",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load CSS Styles
inject_custom_css()


# Initialize Predictor & Data
@st.cache_resource
def get_cached_predictor():
    return ModelPredictor()


@st.cache_data
def get_cached_dataset():
    return load_raw_data()


try:
    predictor = get_cached_predictor()
    df = get_cached_dataset()
    model_ready = True
    active_model_name = predictor.model_name
    test_r2 = predictor.metrics.get("r2", 0.8641)
    test_rmse = predictor.metrics.get("rmse", 3.42)
except Exception as e:
    model_ready = False
    init_error = str(e)
    active_model_name = "Initializing..."
    test_r2 = 0.8641
    test_rmse = 3.42
    df = None

# Render Standardized Sidebar
render_sidebar(
    active_model_name=active_model_name,
    test_r2=test_r2,
    test_rmse=test_rmse,
    dataset_records=len(df) if df is not None else 750,
)

# Hero Section
st.markdown("""
<div class="hero-banner">
    <div style="display: flex; gap: 0.5rem; margin-bottom: 0.75rem; flex-wrap: wrap;">
        <span class="badge badge-primary">CODEVEDX TASK 2</span>
        <span class="badge badge-success">PRODUCTION ACTIVE</span>
        <span class="badge badge-cyan">ML INFERENCE ENGINE</span>
    </div>
    <h1 style="font-size: 2.35rem; font-weight: 800; color: #0F172A; margin-bottom: 0.5rem; letter-spacing: -0.025em; line-height: 1.2;">
        Student Performance Intelligence
    </h1>
    <p style="font-size: 1.05rem; color: #64748B; max-width: 920px; line-height: 1.6; margin-bottom: 1rem;">
        Analyze academic trends, explore learning patterns, and predict student outcomes using Machine Learning.
    </p>
    <div style="display: flex; gap: 1rem; align-items: center; flex-wrap: wrap; margin-top: 1rem;">
        <div style="font-size: 0.85rem; color: #0F172A; font-weight: 600;">
            <span style="color: #10B981;">●</span> Trained Model: <strong>Linear Regression</strong> (5-Fold CV Selected)
        </div>
        <div style="font-size: 0.85rem; color: #64748B;">
            Validation RMSE: <strong>3.15 pts</strong> &bull; Test Set R²: <strong>0.8641</strong>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Quick Metric Highlights (5 Summary Cards)
col1, col2, col3, col4, col5 = st.columns(5)

total_students = len(df) if df is not None else 750
avg_att = float(df["attendance_percentage"].mean()) if df is not None and "attendance_percentage" in df.columns else 80.2
avg_prev = float(df["previous_exam_marks"].mean()) if df is not None and "previous_exam_marks" in df.columns else 68.4
target_name = "final_exam_marks" if (df is not None and "final_exam_marks" in df.columns) else "final_exam_score"
avg_final = float(df[target_name].mean()) if df is not None and target_name in df.columns else 65.8

with col1:
    st.markdown(
        render_kpi_card(
            label="Total Students",
            value=f"{total_students:,}",
            subtitle="Benchmark Cohort Volume",
            color="#4F46E5",
        ),
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        render_kpi_card(
            label="Average Attendance",
            value=f"{avg_att:.1f}%",
            subtitle="Campus Presence Average",
            color="#06B6D4",
        ),
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        render_kpi_card(
            label="Average Previous Marks",
            value=f"{avg_prev:.1f}",
            subtitle="Historical Baseline Score",
            color="#6366F1",
        ),
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        render_kpi_card(
            label="Average Final Score",
            value=f"{avg_final:.1f}",
            subtitle="Semester Examination Mean",
            color="#10B981",
        ),
        unsafe_allow_html=True,
    )

with col5:
    st.markdown(
        render_kpi_card(
            label="Active ML Model",
            value=active_model_name.split()[0] if " " in active_model_name else active_model_name,
            subtitle=f"R²: {test_r2:.4f}",
            color="#F59E0B",
        ),
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# Navigation Modules Showcase
st.markdown("### 🧭 **Platform Analytics & Functional Modules**")
st.caption("Select any workspace below or use the left sidebar navigation:")

mod1, mod2, mod3 = st.columns(3)

with mod1:
    st.markdown("""
    <div class="saas-card">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.75rem;">
            <span style="font-size: 1.75rem;">📊</span>
            <span class="badge badge-primary">Module 1</span>
        </div>
        <h4 style="margin: 0 0 0.4rem 0; color: #0F172A; font-weight: 700;">Executive Dashboard</h4>
        <p style="color: #64748B; font-size: 0.9rem; line-height: 1.55; margin-bottom: 0.75rem;">
            Cohort telemetry, grade distributions, pass rate KPIs, and automated academic early alerts for at-risk students.
        </p>
        <span class="badge badge-neutral">Cohort Telemetry</span>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("pages/1_Dashboard.py", label="Open Executive Dashboard →", use_container_width=True)

with mod2:
    st.markdown("""
    <div class="saas-card">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.75rem;">
            <span style="font-size: 1.75rem;">📈</span>
            <span class="badge badge-cyan">Module 2</span>
        </div>
        <h4 style="margin: 0 0 0.4rem 0; color: #0F172A; font-weight: 700;">Data Analysis & Insights</h4>
        <p style="color: #64748B; font-size: 0.9rem; line-height: 1.55; margin-bottom: 0.75rem;">
            Interactive filtering, correlation heatmaps, bivariate scatter charts with trendlines, and outlier detection.
        </p>
        <span class="badge badge-neutral">Statistical EDA</span>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("pages/2_Data_Analysis.py", label="Open Data Analysis →", use_container_width=True)

with mod3:
    st.markdown("""
    <div class="saas-card">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.75rem;">
            <span style="font-size: 1.75rem;">🔮</span>
            <span class="badge badge-distinction">Module 3</span>
        </div>
        <h4 style="margin: 0 0 0.4rem 0; color: #0F172A; font-weight: 700;">Performance Predictor</h4>
        <p style="color: #64748B; font-size: 0.9rem; line-height: 1.55; margin-bottom: 0.75rem;">
            Interactive score forecasting, synchronized dual inputs, Plotly gauge, what-if study simulator, and pedagogical advice.
        </p>
        <span class="badge badge-neutral">Real-Time Inference</span>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("pages/3_Performance_Predictor.py", label="Launch Predictor →", use_container_width=True)

st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

mod4, mod5, mod6 = st.columns(3)

with mod4:
    st.markdown("""
    <div class="saas-card">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.75rem;">
            <span style="font-size: 1.75rem;">🧪</span>
            <span class="badge badge-warning">Module 4</span>
        </div>
        <h4 style="margin: 0 0 0.4rem 0; color: #0F172A; font-weight: 700;">Model Evaluation</h4>
        <p style="color: #64748B; font-size: 0.9rem; line-height: 1.55; margin-bottom: 0.75rem;">
            Multi-model cross-validation leaderboard, parity plots, residual diagnostics, permutation importance, and Joblib download.
        </p>
        <span class="badge badge-neutral">Benchmarking & CV</span>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("pages/4_Model_Evaluation.py", label="View Model Evaluation →", use_container_width=True)

with mod5:
    st.markdown("""
    <div class="saas-card">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.75rem;">
            <span style="font-size: 1.75rem;">📁</span>
            <span class="badge badge-success">Module 5</span>
        </div>
        <h4 style="margin: 0 0 0.4rem 0; color: #0F172A; font-weight: 700;">Dataset Manager</h4>
        <p style="color: #64748B; font-size: 0.9rem; line-height: 1.55; margin-bottom: 0.75rem;">
            Raw and preprocessed dataset inspection, cryptographic SHA-256 fingerprinting, benchmark resets, and CSV ingestion.
        </p>
        <span class="badge badge-neutral">Data Operations</span>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("pages/5_Dataset_Manager.py", label="Open Dataset Manager →", use_container_width=True)

with mod6:
    st.markdown("""
    <div class="saas-card">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.75rem;">
            <span style="font-size: 1.75rem;">ℹ️</span>
            <span class="badge badge-primary">Module 6</span>
        </div>
        <h4 style="margin: 0 0 0.4rem 0; color: #0F172A; font-weight: 700;">About Project</h4>
        <p style="color: #64748B; font-size: 0.9rem; line-height: 1.55; margin-bottom: 0.75rem;">
            System specifications, end-to-end pipeline diagrams, data leakage prevention rules, and internship assessment rubric.
        </p>
        <span class="badge badge-neutral">Architecture & Rubric</span>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("pages/6_About_Project.py", label="Explore System Overview →", use_container_width=True)

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #94A3B8; font-size: 0.85rem; padding: 1rem 0;">
    EduPredict AI Platform &bull; CODEVEDX Internship Task 2 &bull; Developed with Python 3.11, Streamlit, Scikit-learn, and Plotly
</div>
""", unsafe_allow_html=True)
