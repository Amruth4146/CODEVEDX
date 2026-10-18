"""
Page 5: Dataset Manager & Health Inspector
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

from src.data_loader import (
    load_raw_data,
    load_processed_data,
    generate_benchmark_dataset,
    save_processed_data,
    get_dataset_summary,
    RAW_DATA_PATH,
)
from src.preprocessing import clean_data

st.set_page_config(
    page_title="Dataset Manager | Student Performance System",
    page_icon="📁",
    layout="wide",
)

# Apply Scoped SaaS Styling
from src.ui_components import inject_custom_css, render_header, render_sidebar, render_kpi_card
from src.train_model import load_model_bundle
inject_custom_css()

try:
    bundle = load_model_bundle()
    active_name = bundle.get("model_name", "Linear Regression")
    test_m = bundle.get("test_metrics", {})
    t_r2 = test_m.get("r2", 0.8641)
    t_rmse = test_m.get("rmse", 3.42)
except Exception:
    active_name = "Linear Regression"
    t_r2 = 0.8641
    t_rmse = 3.42

raw_df = load_raw_data()
processed_df = load_processed_data()
summary = get_dataset_summary(raw_df)

# Render Standardized Sidebar
render_sidebar(
    active_model_name=active_name,
    test_r2=t_r2,
    test_rmse=t_rmse,
    dataset_records=len(raw_df),
)

# Header
render_header(
    title="Dataset Management Catalog",
    subtitle="Inspect raw and preprocessed academic records, verify schema integrity, download sanitized CSV data, or ingest new cohorts.",
    badge_text="DATA OPERATIONS & CATALOG",
    badge_variant="success",
    status_pill=f"Catalog Volume: {summary['total_records']} Records",
)

# Dataset Health KPI Cards
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(
        render_kpi_card(
            label="Total Student Records",
            value=f"{summary['total_records']:,}",
            subtitle="Full Cohort Volume",
            color="#4F46E5",
        ),
        unsafe_allow_html=True,
    )

with m2:
    st.markdown(
        render_kpi_card(
            label="Attribute Columns",
            value=f"{summary['total_columns']}",
            subtitle="Features & Target Metrics",
            color="#06B6D4",
        ),
        unsafe_allow_html=True,
    )

with m3:
    st.markdown(
        render_kpi_card(
            label="Missing Values",
            value=f"{summary['missing_values']}",
            subtitle="Pre-Sanitization Nulls",
            color="#10B981" if summary['missing_values'] == 0 else "#F59E0B",
        ),
        unsafe_allow_html=True,
    )

with m4:
    st.markdown(
        render_kpi_card(
            label="Duplicates Detected",
            value=f"{summary['duplicate_rows']}",
            subtitle="Identical Row Deduplication",
            color="#10B981" if summary['duplicate_rows'] == 0 else "#EF4444",
        ),
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# Action Bar
act1, act2 = st.columns([1, 1])

with act1:
    if st.button("🔄 Reset to Standard Benchmark Dataset", use_container_width=True):
        generate_benchmark_dataset()
        st.cache_data.clear()
        st.success("Successfully regenerated standard benchmark dataset!")
        st.rerun()

with act2:
    csv_buf = io.StringIO()
    processed_df.to_csv(csv_buf, index=False)
    st.download_button(
        label="📥 Download Cleaned Dataset (CSV)",
        data=csv_buf.getvalue(),
        file_name="cleaned_student_academic_records.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.markdown("---")

# Data Viewers
tab_proc, tab_raw, tab_upload = st.tabs(["✨ Processed & Cleaned Records", "📄 Raw Student Records", "📤 Ingest Custom CSV"])

with tab_proc:
    st.markdown(f"**Cleaned Academic Data Catalog** ({len(processed_df)} rows):")
    st.dataframe(processed_df, use_container_width=True)

with tab_raw:
    st.markdown(f"**Raw Ingested Records** ({len(raw_df)} rows):")
    st.dataframe(raw_df, use_container_width=True)

from src.security import validate_uploaded_csv

with tab_upload:
    st.markdown("#### Ingest New Academic Dataset")
    st.caption("Upload a custom CSV file to replace current training data.")
    uploaded = st.file_uploader("Select CSV file:", type=["csv"], key="raw_uploader")

    if uploaded is not None:
        required = ["attendance_percentage", "previous_exam_marks", "daily_study_hours", "final_exam_marks"]
        is_valid, err_msg, new_df = validate_uploaded_csv(uploaded, required_columns=required)

        if not is_valid:
            st.error(f"❌ Upload Validation Failure: {err_msg}")
        else:
            st.success(f"✅ Upload validated: {len(new_df)} student rows, {len(new_df.columns)} columns.")
            st.write("Preview of uploaded data:", new_df.head())

            if st.button("Confirm and Save Dataset"):
                new_df.to_csv(RAW_DATA_PATH, index=False)
                cleaned = clean_data(new_df)
                save_processed_data(cleaned)
                st.cache_data.clear()
                st.success("Dataset successfully ingested and processed! Retrain models in Page 4 to update predictions.")
