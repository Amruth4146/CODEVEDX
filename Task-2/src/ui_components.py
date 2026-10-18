"""
UI Components & SaaS Design System Module.
Student Performance Prediction System - CODEVEDX Task 2

Provides reusable, responsive layout components:
- inject_custom_css: Loads scoped SaaS stylesheet
- render_header: Clean SaaS header with breadcrumbs and status indicators
- render_sidebar: Standardized sidebar branding, telemetry, and navigation status
- render_kpi_card: Standardized metric card with hover micro-interactions
- apply_saas_plotly_theme: Applies consistent light SaaS styling to Plotly figures
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any
import streamlit as st
import plotly.graph_objects as go

# Root Path
BASE_DIR = Path(__file__).resolve().parent.parent
CSS_PATH = BASE_DIR / "assets" / "styles.css"


def inject_custom_css() -> None:
    """Injects the production SaaS stylesheet into the active Streamlit page."""
    if CSS_PATH.exists():
        with open(CSS_PATH, "r", encoding="utf-8") as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)


def render_header(
    title: str,
    subtitle: str,
    badge_text: str = "PRODUCTION AI SYSTEM",
    badge_variant: str = "primary",
    status_pill: Optional[str] = None,
) -> None:
    """
    Renders a modern, responsive SaaS page header with hierarchy, badges, and metadata.
    """
    badge_class = f"badge badge-{badge_variant}"
    status_html = ""
    if status_pill:
        status_html = f'<span class="badge badge-success" style="margin-left: 0.5rem;">{status_pill}</span>'

    header_html = f"""
    <div style="margin-bottom: 1.75rem; border-bottom: 1px solid #E2E8F0; padding-bottom: 1.25rem;">
        <div style="display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 0.4rem;">
            <span class="{badge_class}">{badge_text}</span>
            {status_html}
        </div>
        <h1 style="font-size: 2.15rem; font-weight: 800; color: #0F172A; margin: 0.2rem 0 0.35rem 0; letter-spacing: -0.025em; line-height: 1.2;">
            {title}
        </h1>
        <p style="color: #64748B; font-size: 1.02rem; margin: 0; line-height: 1.5; max-width: 900px;">
            {subtitle}
        </p>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)


def render_sidebar(
    active_model_name: str = "Linear Regression",
    test_r2: float = 0.8641,
    test_rmse: float = 3.42,
    dataset_records: int = 750,
) -> None:
    """
    Renders the standardized SaaS sidebar across all pages.
    Features:
    - Application branding
    - Live model production status
    - Navigation context
    - System metadata
    """
    with st.sidebar:
        # Branding
        st.markdown("""
        <div class="sidebar-branding">
            <div style="display: flex; align-items: center; gap: 0.6rem;">
                <div style="background: linear-gradient(135deg, #4F46E5 0%, #06B6D4 100%); width: 36px; height: 36px; border-radius: 10px; display: flex; align-items: center; justify-content: center; color: white; font-size: 1.2rem; font-weight: 800;">
                    ✦
                </div>
                <div>
                    <div class="sidebar-title">EduPredict AI</div>
                    <div class="sidebar-subtitle">Student Performance System</div>
                </div>
            </div>
            <div style="margin-top: 0.6rem;">
                <span class="badge badge-primary" style="font-size: 0.7rem;">CODEVEDX TASK 2</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Telemetry Card
        st.markdown(f"""
        <div class="sidebar-status-card">
            <div style="font-weight: 700; color: #0F172A; margin-bottom: 0.4rem; display: flex; align-items: center; gap: 0.35rem;">
                <span style="color: #10B981;">●</span> Production Telemetry
            </div>
            <div style="color: #64748B; line-height: 1.6;">
                <strong>Active Model:</strong> <span style="color: #4F46E5;">{active_model_name}</span><br>
                <strong>Test R² Score:</strong> <span style="color: #0F172A; font-weight: 600;">{test_r2:.4f}</span><br>
                <strong>Test RMSE:</strong> <span style="color: #0F172A; font-weight: 600;">{test_rmse:.2f} pts</span><br>
                <strong>Cohort Volume:</strong> <span style="color: #0F172A; font-weight: 600;">{dataset_records} students</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Quick Navigation Summary
        st.markdown("""
        <div style="color: #64748B; font-size: 0.78rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;">
            System Modules
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        - 📊 **Dashboard** (Cohort Overview)
        - 📈 **Data Analysis** (EDA & Feature Heatmap)
        - 🔮 **Performance Predictor** (Inference)
        - 🧪 **Model Evaluation** (Benchmarking & CV)
        - 📁 **Dataset Manager** (Catalog & Health)
        - ℹ️ **About Project** (System Architecture)
        """)

        st.markdown("---")

        # Footer Version Tag
        st.markdown("""
        <div style="text-align: center; color: #94A3B8; font-size: 0.75rem; padding-top: 0.5rem;">
            v2.4.2 • Production SaaS Edition<br>
            CODEVEDX Internship Assessment
        </div>
        """, unsafe_allow_html=True)


def render_kpi_card(
    label: str,
    value: str,
    subtitle: Optional[str] = None,
    color: str = "#4F46E5",
) -> str:
    """Returns HTML for a premium SaaS KPI card."""
    sub_html = f'<div class="metric-sub">{subtitle}</div>' if subtitle else ""
    return f"""
    <div class="metric-box">
        <div class="metric-val" style="color: {color};">{value}</div>
        <div class="metric-label">{label}</div>
        {sub_html}
    </div>
    """


def apply_saas_plotly_theme(
    fig: go.Figure,
    title: Optional[str] = None,
    xaxis_title: Optional[str] = None,
    yaxis_title: Optional[str] = None,
) -> go.Figure:
    """
    Applies the standardized SaaS light aesthetic to any Plotly figure:
    - Pure white paper background
    - Subtle slate-100 plot canvas
    - Slate-200 grid lines
    - High-contrast Slate-900 typography (Inter)
    """
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Inter, -apple-system, sans-serif", color="#0F172A", size=12),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#F8FAFC",
        title=dict(
            text=title or fig.layout.title.text,
            font=dict(color="#0F172A", size=15, weight=700),
            x=0.01,
            xanchor="left",
        ),
        margin=dict(l=45, r=30, t=55, b=45),
        xaxis=dict(
            title=xaxis_title or fig.layout.xaxis.title.text,
            gridcolor="#E2E8F0",
            linecolor="#CBD5E1",
            tickfont=dict(color="#64748B", size=11),
            zerolinecolor="#E2E8F0",
        ),
        yaxis=dict(
            title=yaxis_title or fig.layout.yaxis.title.text,
            gridcolor="#E2E8F0",
            linecolor="#CBD5E1",
            tickfont=dict(color="#64748B", size=11),
            zerolinecolor="#E2E8F0",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#475569", size=11),
        ),
    )
    return fig
