"""
Feature Selection and Predictive Importance Analysis Module.
Student Performance Prediction System - CODEVEDX Task 2

Implements:
- Correlation-based feature analysis (Pearson & Spearman)
- Mutual Information regression (non-linear dependency capture)
- Random Forest feature importance (Mean Decrease in Impurity)
- Comparative feature importance visualizers
- Leakage-free cross-validated feature selection
- Educational pedagogical guide on feature selection theory and practice
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_selection import (
    f_regression,
    mutual_info_regression,
    SelectKBest,
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_validate

logger = logging.getLogger(__name__)

# Target column variations
TARGET_COLUMNS = ["final_exam_score", "final_exam_marks"]

# Predictor feature mapping / aliases
FEATURE_ALIASES = {
    "attendance_percentage": "attendance_percentage",
    "previous_marks": "previous_exam_marks",
    "previous_exam_marks": "previous_marks",
    "study_hours_per_day": "daily_study_hours",
    "daily_study_hours": "study_hours_per_day",
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


def resolve_target_and_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, str, List[str]]:
    """
    Identifies target column and extracts predictor features without leakage.
    Ensures final_exam_score / final_exam_marks is strictly excluded from predictor X.
    """
    target_col = None
    for col in TARGET_COLUMNS:
        if col in df.columns:
            target_col = col
            break

    if target_col is None:
        raise KeyError(f"Target column not found. Expected one of {TARGET_COLUMNS}")

    # Exclude ID columns and target column
    ignored_cols = set(TARGET_COLUMNS + ["student_id", "performance_category"])
    feature_cols = [c for c in df.columns if c not in ignored_cols]

    # Convert categoricals if any
    X = df[feature_cols].copy()
    if "extracurricular_activities" in X.columns and X["extracurricular_activities"].dtype == "object":
        X["extracurricular_activities"] = (
            X["extracurricular_activities"].str.lower().map({"yes": 1, "no": 0, "true": 1, "false": 0}).fillna(0).astype(int)
        )

    y = df[target_col].copy()

    # Integrity verification: target must never be an input feature
    assert target_col not in X.columns, f"Data leakage error: Target '{target_col}' detected in predictor inputs!"

    return X, y, target_col, feature_cols


# ============================================================================
# 1. CORRELATION-BASED FEATURE ANALYSIS
# ============================================================================

def compute_correlation_analysis(X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    """
    Calculates Pearson linear correlation and Spearman rank-order correlation
    between each predictor feature and the target variable.
    """
    features = list(X_train.columns)
    rows = []

    for feat in features:
        s_feat = pd.to_numeric(X_train[feat], errors="coerce")
        s_target = pd.to_numeric(y_train, errors="coerce")

        # Pearson correlation (linear relationship)
        pearson_r = float(s_feat.corr(s_target, method="pearson"))
        # Spearman correlation (monotonic rank relationship)
        spearman_rho = float(s_feat.corr(s_target, method="spearman"))

        rows.append({
            "feature": feat,
            "formatted_feature": feat.replace("_", " ").title(),
            "pearson_r": round(pearson_r, 4),
            "abs_pearson": round(abs(pearson_r), 4),
            "spearman_rho": round(spearman_rho, 4),
        })

    df_corr = pd.DataFrame(rows).sort_values(by="abs_pearson", ascending=False).reset_index(drop=True)
    df_corr["correlation_rank"] = range(1, len(df_corr) + 1)
    return df_corr


# ============================================================================
# 2. MUTUAL INFORMATION REGRESSION
# ============================================================================

def compute_mutual_information(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Computes Mutual Information (MI) between each predictor and the continuous target.
    Mutual Information quantifies both linear and non-linear dependencies.
    """
    features = list(X_train.columns)
    mi_scores = mutual_info_regression(X_train, y_train, random_state=random_state)

    total_mi = float(np.sum(mi_scores))
    mi_pct = (mi_scores / total_mi * 100) if total_mi > 0 else np.zeros(len(features))

    df_mi = pd.DataFrame({
        "feature": features,
        "formatted_feature": [f.replace("_", " ").title() for f in features],
        "mi_score": np.round(mi_scores, 4),
        "mi_percentage": np.round(mi_pct, 2),
    }).sort_values(by="mi_score", ascending=False).reset_index(drop=True)

    df_mi["mi_rank"] = range(1, len(df_mi) + 1)
    return df_mi


# ============================================================================
# 3. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================================

def compute_rf_importance(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Computes Gini / Mean Decrease in Impurity (MDI) feature importances
    using an ensemble of Random Forest decision trees.
    """
    features = list(X_train.columns)
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=6,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    importances = rf.feature_importances_
    imp_pct = (importances / importances.sum()) * 100

    df_rf = pd.DataFrame({
        "feature": features,
        "formatted_feature": [f.replace("_", " ").title() for f in features],
        "rf_importance": np.round(importances, 4),
        "rf_percentage": np.round(imp_pct, 2),
    }).sort_values(by="rf_importance", ascending=False).reset_index(drop=True)

    df_rf["rf_rank"] = range(1, len(df_rf) + 1)
    return df_rf


# ============================================================================
# 4. FEATURE SELECTION COMPARISON TABLE
# ============================================================================

def generate_feature_comparison_table(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Builds a unified comparison matrix summarizing:
    - Pearson linear correlation ($r$)
    - Spearman rank correlation ($\rho$)
    - Mutual Information score & relative %
    - Random Forest importance %
    - Average rank across methods
    - Diagnostic retention recommendation
    """
    df_corr = compute_correlation_analysis(X_train, y_train)
    df_mi = compute_mutual_information(X_train, y_train, random_state=random_state)
    df_rf = compute_rf_importance(X_train, y_train, random_state=random_state)

    merged = df_corr.merge(df_mi, on=["feature", "formatted_feature"]).merge(
        df_rf, on=["feature", "formatted_feature"]
    )

    # Calculate average rank across all three paradigms
    merged["avg_rank"] = (
        (merged["correlation_rank"] + merged["mi_rank"] + merged["rf_rank"]) / 3.0
    ).round(2)

    # Sort by consensus rank
    merged = merged.sort_values(by="avg_rank", ascending=True).reset_index(drop=True)

    # Generate recommendation without prematurely discarding low-correlation features
    def make_recommendation(row: pd.Series) -> str:
        if row["avg_rank"] <= 2.0:
            return "Retain (Core Predictor)"
        elif row["avg_rank"] <= 3.5:
            return "Retain (Strong Impact)"
        elif row["mi_score"] > 0.05 or row["abs_pearson"] > 0.2:
            return "Retain (Contextual Feature)"
        else:
            return "Optional / Regularized"

    merged["recommendation"] = merged.apply(make_recommendation, axis=1)
    return merged


# ============================================================================
# 5. FEATURE IMPORTANCE VISUALIZERS
# ============================================================================

def plot_feature_importance_comparison_chart(
    comparison_df: pd.DataFrame,
    theme: str = "light",
) -> go.Figure:
    """
    Renders an interactive multi-bar comparison chart displaying:
    - Pearson Correlation (%)
    - Mutual Information (%)
    - Random Forest Importance (%)
    for each evaluated feature.
    """
    theme_cfg = THEME_TEMPLATES.get(theme, THEME_TEMPLATES["light"])

    # Prepare normalized comparison metrics
    df_plot = comparison_df.copy()
    corr_pct = (df_plot["abs_pearson"] / df_plot["abs_pearson"].sum()) * 100

    fig = go.Figure()

    # Trace 1: Random Forest
    fig.add_trace(
        go.Bar(
            name="Random Forest (%)",
            x=df_plot["formatted_feature"],
            y=df_plot["rf_percentage"],
            marker_color="#6366F1",
            text=df_plot["rf_percentage"].apply(lambda v: f"{v:.1f}%"),
            textposition="outside",
        )
    )

    # Trace 2: Mutual Information
    fig.add_trace(
        go.Bar(
            name="Mutual Information (%)",
            x=df_plot["formatted_feature"],
            y=df_plot["mi_percentage"],
            marker_color="#06B6D4",
            text=df_plot["mi_percentage"].apply(lambda v: f"{v:.1f}%"),
            textposition="outside",
        )
    )

    # Trace 3: Pearson Linear Correlation (Normalized %)
    fig.add_trace(
        go.Bar(
            name="Correlation Relative (%)",
            x=df_plot["formatted_feature"],
            y=np.round(corr_pct, 2),
            marker_color="#10B981",
            text=corr_pct.apply(lambda v: f"{v:.1f}%"),
            textposition="outside",
        )
    )

    fig.update_layout(
        template=theme_cfg["plotly_template"],
        barmode="group",
        title="Multi-Method Feature Importance & Predictive Contribution",
        paper_bgcolor=theme_cfg["bg_paper"],
        plot_bgcolor=theme_cfg["bg_plot"],
        font=dict(color=theme_cfg["font_color"]),
        xaxis_title="Evaluated Features",
        yaxis_title="Relative Predictive Contribution (%)",
        yaxis=dict(range=[0, max(df_plot["rf_percentage"].max(), df_plot["mi_percentage"].max()) * 1.25]),
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
    )
    return fig


# ============================================================================
# 6. LEAKAGE-FREE CROSS-VALIDATION PIPELINE
# ============================================================================

def cross_validate_feature_selection(
    X: pd.DataFrame,
    y: pd.Series,
    k_features: int = 3,
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Performs k-fold cross-validation with feature selection strictly nested
    inside the Pipeline to guarantee zero data leakage between train and test folds.
    Compares baseline model (all features) vs selected-feature model (top-k features).
    """
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Baseline Pipeline (All Features)
    base_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0)),
    ])

    # Feature-Selection Pipeline (Feature selection learned only inside train fold)
    k_actual = min(k_features, X.shape[1])
    sel_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("selector", SelectKBest(score_func=f_regression, k=k_actual)),
        ("model", Ridge(alpha=1.0)),
    ])

    scoring = ["r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"]

    cv_base = cross_validate(base_pipe, X, y, cv=cv, scoring=scoring)
    cv_sel = cross_validate(sel_pipe, X, y, cv=cv, scoring=scoring)

    return {
        "k_features": k_actual,
        "n_splits": n_splits,
        "baseline": {
            "r2_mean": float(np.mean(cv_base["test_r2"])),
            "r2_std": float(np.std(cv_base["test_r2"])),
            "rmse_mean": float(abs(np.mean(cv_base["test_neg_root_mean_squared_error"]))),
            "mae_mean": float(abs(np.mean(cv_base["test_neg_mean_absolute_error"]))),
        },
        "selected": {
            "r2_mean": float(np.mean(cv_sel["test_r2"])),
            "r2_std": float(np.std(cv_sel["test_r2"])),
            "rmse_mean": float(abs(np.mean(cv_sel["test_neg_root_mean_squared_error"]))),
            "mae_mean": float(abs(np.mean(cv_sel["test_neg_mean_absolute_error"]))),
        },
    }


# ============================================================================
# 7. SELECTED FEATURE SUMMARY & EDUCATIONAL GUIDE
# ============================================================================

def get_selected_feature_summary(comparison_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Returns an executive digest of the top predictive contributors.
    """
    top_feature = comparison_df.iloc[0]["formatted_feature"]
    top_rf_pct = comparison_df.iloc[0]["rf_percentage"]
    core_features = comparison_df[comparison_df["recommendation"].str.contains("Core|Strong")]["formatted_feature"].tolist()

    return {
        "top_contributor": top_feature,
        "top_rf_pct": f"{top_rf_pct:.1f}%",
        "core_predictors": core_features,
        "total_evaluated": len(comparison_df),
    }


def get_educational_explanation() -> Dict[str, str]:
    """
    Returns detailed pedagogical guidance on feature selection concepts.
    """
    return {
        "what_is_it": (
            "Feature selection is the rigorous process of isolating the most informative, relevant input features "
            "to build machine learning models that are more accurate, generalizable, and interpretable."
        ),
        "why_it_matters": (
            "1. Mitigates Overfitting: Redundant or noisy attributes cause models to memorize spurious patterns.\n"
            "2. Curbs the Curse of Dimensionality: High-dimensional spaces require exponentially more data to sample effectively.\n"
            "3. Improves Model Interpretability: Stakeholders can understand the actual pedagogical drivers of student outcomes.\n"
            "4. Reduces Inference Latency: Fewer features result in faster data collection, cleaning, and model prediction."
        ),
        "non_linear_note": (
            "Linear correlation (Pearson r) only captures straight-line associations. Mutual Information and Tree-based "
            "ensembles (Random Forest) capture complex non-linear thresholds and variable interactions that correlation misses entirely."
        ),
        "never_discard_blindly": (
            "Crucial Principle: Never automatically discard a feature simply because its bivariate linear correlation is low! "
            "A feature with weak marginal correlation may exhibit significant predictive synergy when combined with other features "
            "(interaction effects) or may act non-linearly."
        ),
        "leakage_warning": (
            "Leakage-Free Execution: All feature selection rankings, transformations, and scalings must be fitted exclusively "
            "on training splits or strictly within cross-validation folds. Fitting feature selectors on the entire dataset "
            "before evaluation leaks validation data distribution into training, yielding overly optimistic, invalid performance estimates."
        ),
    }


# ============================================================================
# 8. BACKWARD COMPATIBILITY EXPORTS
# ============================================================================

def calculate_feature_correlations(
    df: pd.DataFrame,
    target_col: str = "final_exam_marks",
) -> pd.DataFrame:
    """Backward compatibility function for Pearson correlation ranking."""
    numeric_df = df.select_dtypes(include=[np.number])
    target = target_col if target_col in numeric_df.columns else get_target_column(df)
    if not target or target not in numeric_df.columns:
        return pd.DataFrame()

    corr_series = numeric_df.corr()[target].drop(index=target)
    corr_df = pd.DataFrame({
        "feature": corr_series.index,
        "correlation": corr_series.values,
        "abs_correlation": corr_series.abs().values,
    }).sort_values(by="abs_correlation", ascending=False).reset_index(drop=True)

    corr_df["formatted_feature"] = corr_df["feature"].str.replace("_", " ").str.title()
    return corr_df


def get_feature_importances(model: Any, feature_names: List[str]) -> pd.DataFrame:
    """Backward compatibility function for extracting trained model importances."""
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        coefs = np.abs(model.coef_)
        if coefs.ndim > 1:
            coefs = coefs.flatten()
        importances = coefs / coefs.sum() if coefs.sum() > 0 else coefs

    if importances is None:
        importances = np.ones(len(feature_names)) / max(1, len(feature_names))

    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values(by="importance", ascending=True).reset_index(drop=True)

    imp_df["formatted_feature"] = imp_df["feature"].str.replace("_", " ").str.title()
    imp_df["importance_percentage"] = (imp_df["importance"] * 100).round(2)
    return imp_df


def plot_feature_importance_bar(importance_df: pd.DataFrame) -> go.Figure:
    """Backward compatibility function for horizontal bar chart."""
    fig = px.bar(
        importance_df,
        x="importance_percentage",
        y="formatted_feature",
        orientation="h",
        text="importance_percentage",
        title="Predictive Feature Importance Breakdown (%)",
        labels={"importance_percentage": "Importance (%)", "formatted_feature": "Feature"},
        color="importance_percentage",
        color_continuous_scale="Purples",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis_title="Relative Impact (%)",
        yaxis_title="",
        coloraxis_showscale=False,
    )
    return fig


def select_top_features(X: pd.DataFrame, y: pd.Series, k: int = 3) -> List[str]:
    """Backward compatibility function for top-k feature selection."""
    selector = SelectKBest(score_func=f_regression, k=min(k, X.shape[1]))
    selector.fit(X, y)
    top_indices = selector.get_support(indices=True)
    return [X.columns[i] for i in top_indices]


def get_target_column(df: pd.DataFrame) -> Optional[str]:
    """Helper to detect target column name."""
    for col in TARGET_COLUMNS:
        if col in df.columns:
            return col
    return None
