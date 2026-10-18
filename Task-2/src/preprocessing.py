"""
Preprocessing and Data Quality Pipeline for Student Performance Prediction System.
Provides rigorous data quality inspection, duplicate detection, range validation,
IQR outlier analysis, median imputation, and leakage-free Scikit-learn Pipeline scaling.
"""

import logging
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Supported target column variants
TARGET_COLS = ["final_exam_marks", "final_exam_score"]
PRIMARY_TARGET = "final_exam_marks"

# Academic feature definitions
CORE_FEATURE_COLS = [
    "attendance_percentage",
    "previous_exam_marks",
    "daily_study_hours",
]

ALL_FEATURE_COLS = [
    "attendance_percentage",
    "previous_exam_marks",
    "daily_study_hours",
    "assignments_completed",
    "extracurricular_activities",
    "sleep_hours",
]

# Physical and logical academic parameter boundaries
VALID_BOUNDS = {
    "attendance_percentage": (0.0, 100.0),
    "previous_exam_marks": (0.0, 100.0),
    "final_exam_marks": (0.0, 100.0),
    "final_exam_score": (0.0, 100.0),
    "daily_study_hours": (0.0, 24.0),
    "assignments_completed": (0, 10),
    "sleep_hours": (0.0, 24.0),
}


# ============================================================================
# 1. DATA QUALITY INSPECTION (Leakage-Free Inspection Only)
# ============================================================================

def detect_missing_values(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detects missing / NaN values per column and across the entire DataFrame.
    """
    if df is None or df.empty:
        return {"total_missing": 0, "columns": {}, "has_missing": False}

    missing_counts = df.isnull().sum()
    total_missing = int(missing_counts.sum())
    total_cells = int(df.size)
    total_rows = len(df)

    col_details = {}
    for col in df.columns:
        cnt = int(missing_counts[col])
        pct = round((cnt / total_rows) * 100, 2) if total_rows > 0 else 0.0
        col_details[col] = {"missing_count": cnt, "missing_percentage": pct}

    return {
        "total_missing": total_missing,
        "missing_percentage": round((total_missing / total_cells) * 100, 2) if total_cells > 0 else 0.0,
        "has_missing": total_missing > 0,
        "columns": col_details,
    }


def detect_duplicates(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Identifies exact duplicate records in the DataFrame.
    """
    if df is None or df.empty:
        return {"duplicate_count": 0, "duplicate_percentage": 0.0, "has_duplicates": False}

    dup_mask = df.duplicated()
    dup_count = int(dup_mask.sum())
    total_rows = len(df)
    dup_pct = round((dup_count / total_rows) * 100, 2) if total_rows > 0 else 0.0

    return {
        "duplicate_count": dup_count,
        "duplicate_percentage": dup_pct,
        "has_duplicates": dup_count > 0,
        "duplicate_indices": df.index[dup_mask].tolist(),
    }


def validate_numerical_ranges(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validates numerical values against logical academic constraints:
    - Attendance percentage: [0, 100]
    - Examination marks (previous & final): [0, 100]
    - Daily study hours: >= 0 (detects negative values)
    - Assignments completed: [0, 10]
    - Sleep hours: [0, 24]
    Detects both range violations and unparseable non-numeric values in numeric columns.
    """
    violations: Dict[str, Dict[str, Any]] = {}
    total_violations = 0

    if df is None or df.empty:
        return {"total_violations": 0, "violations": violations}

    # 1. Attendance Percentage
    if "attendance_percentage" in df.columns:
        col = pd.to_numeric(df["attendance_percentage"], errors="coerce")
        unparseable = df["attendance_percentage"].notna() & col.isna()
        invalid_mask = (col < 0.0) | (col > 100.0) | unparseable
        cnt = int(invalid_mask.sum())
        if cnt > 0:
            violations["attendance_percentage"] = {
                "issue": "Attendance percentage out of range [0, 100]% or unparseable",
                "count": cnt,
                "indices": df.index[invalid_mask].tolist(),
            }
            total_violations += cnt

    # 2. Previous Exam Marks
    if "previous_exam_marks" in df.columns:
        col = pd.to_numeric(df["previous_exam_marks"], errors="coerce")
        unparseable = df["previous_exam_marks"].notna() & col.isna()
        invalid_mask = (col < 0.0) | (col > 100.0) | unparseable
        cnt = int(invalid_mask.sum())
        if cnt > 0:
            violations["previous_exam_marks"] = {
                "issue": "Previous marks out of range [0, 100] or unparseable",
                "count": cnt,
                "indices": df.index[invalid_mask].tolist(),
            }
            total_violations += cnt

    # 3. Final Exam Marks / Score
    target_col = get_target_column(df)
    if target_col and target_col in df.columns:
        col = pd.to_numeric(df[target_col], errors="coerce")
        invalid_mask = (col < 0.0) | (col > 100.0)
        cnt = int(invalid_mask.sum())
        if cnt > 0:
            violations[target_col] = {
                "issue": f"{target_col} out of range [0, 100]",
                "count": cnt,
                "indices": df.index[invalid_mask].tolist(),
            }
            total_violations += cnt

    # 4. Daily Study Hours (Negative Detection)
    if "daily_study_hours" in df.columns:
        col = pd.to_numeric(df["daily_study_hours"], errors="coerce")
        invalid_mask = col < 0.0
        cnt = int(invalid_mask.sum())
        if cnt > 0:
            violations["daily_study_hours"] = {
                "issue": "Negative daily study hours (< 0.0 hrs)",
                "count": cnt,
                "indices": df.index[invalid_mask].tolist(),
            }
            total_violations += cnt

    # 5. Assignments Completed
    if "assignments_completed" in df.columns:
        col = pd.to_numeric(df["assignments_completed"], errors="coerce")
        invalid_mask = (col < 0) | (col > 10)
        cnt = int(invalid_mask.sum())
        if cnt > 0:
            violations["assignments_completed"] = {
                "issue": "Assignments completed out of range [0, 10]",
                "count": cnt,
                "indices": df.index[invalid_mask].tolist(),
            }
            total_violations += cnt

    return {
        "total_violations": total_violations,
        "violations": violations,
    }


def detect_outliers_iqr(df: pd.DataFrame, factor: float = 1.5) -> Dict[str, Any]:
    """
    Identifies statistical outliers in all numerical columns using the IQR method:
    - Q1 = 25th percentile, Q3 = 75th percentile
    - IQR = Q3 - Q1
    - Lower Bound = Q1 - (factor * IQR)
    - Upper Bound = Q3 + (factor * IQR)
    """
    outlier_results: Dict[str, Dict[str, Any]] = {}
    summary_rows = []

    if df is None or df.empty:
        return {"columns": outlier_results, "summary_df": pd.DataFrame()}

    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "student_id"]

    for col_name in numeric_cols:
        series = df[col_name].dropna()
        if series.empty:
            continue

        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower_bound = q1 - (factor * iqr)
        upper_bound = q3 + (factor * iqr)

        outlier_mask = (series < lower_bound) | (series > upper_bound)
        count = int(outlier_mask.sum())
        pct = round((count / len(series)) * 100, 2) if len(series) > 0 else 0.0
        indices = series.index[outlier_mask].tolist()

        outlier_results[col_name] = {
            "q1": round(q1, 2),
            "q3": round(q3, 2),
            "iqr": round(iqr, 2),
            "lower_bound": round(lower_bound, 2),
            "upper_bound": round(upper_bound, 2),
            "outlier_count": count,
            "outlier_percentage": pct,
            "indices": indices,
        }

        summary_rows.append({
            "Feature": col_name.replace("_", " ").title(),
            "Q1 (25%)": round(q1, 2),
            "Q3 (75%)": round(q3, 2),
            "IQR": round(iqr, 2),
            "Lower Bound": round(lower_bound, 2),
            "Upper Bound": round(upper_bound, 2),
            "Outlier Count": count,
            "Outlier %": f"{pct:.1f}%",
        })

    summary_df = pd.DataFrame(summary_rows) if summary_rows else pd.DataFrame()
    return {
        "columns": outlier_results,
        "summary_df": summary_df,
    }


def inspect_data_quality(df: pd.DataFrame, iqr_factor: float = 1.5) -> Dict[str, Any]:
    """
    Comprehensive, non-destructive data quality audit.
    Evaluates missing values, duplicates, boundary violations, and outliers.
    """
    if df is None or df.empty:
        raise ValueError("Cannot inspect data quality on an empty or None DataFrame.")

    missing = detect_missing_values(df)
    duplicates = detect_duplicates(df)
    ranges = validate_numerical_ranges(df)
    outliers = detect_outliers_iqr(df, factor=iqr_factor)

    # Calculate overall health score (0 to 100%)
    total_cells = df.size
    penalties = (missing["total_missing"] * 2) + (duplicates["duplicate_count"] * 5) + (ranges["total_violations"] * 3)
    health_score = max(0.0, min(100.0, round(100.0 - (penalties / max(1, total_cells)) * 100, 1)))

    return {
        "total_records": len(df),
        "total_columns": len(df.columns),
        "missing_report": missing,
        "duplicate_report": duplicates,
        "range_report": ranges,
        "outlier_report": outliers,
        "health_score": health_score,
    }


# ============================================================================
# 2. DATA CLEANING & TRANSFORMATION (Independent of Model Training)
# ============================================================================

def remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Removes exact duplicate rows from DataFrame.
    Returns: (cleaned_df, num_duplicates_removed)
    """
    if df is None or df.empty:
        return df, 0

    init_len = len(df)
    cleaned = df.drop_duplicates().copy()
    dropped = init_len - len(cleaned)
    if dropped > 0:
        logger.info("Removed %d duplicate records.", dropped)
    return cleaned, dropped


def handle_invalid_boundaries(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Enforces valid mathematical and academic ranges:
    - Attendance: clipped to [0, 100]
    - Previous Marks: clipped to [0, 100]
    - Final Marks: clipped to [0, 100]
    - Study Hours: clipped to [0, 24] (negative study hours set to 0.0)
    - Assignments: clipped to [0, 10]
    - Sleep Hours: clipped to [0, 24]
    """
    cleaned = df.copy()
    corrections = {}

    for col, (low, high) in VALID_BOUNDS.items():
        if col in cleaned.columns:
            if not np.issubdtype(cleaned[col].dtype, np.number):
                cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
            if np.issubdtype(cleaned[col].dtype, np.number):
                out_mask = (cleaned[col] < low) | (cleaned[col] > high)
                num_violations = int(out_mask.sum())
                if num_violations > 0:
                    cleaned[col] = cleaned[col].clip(lower=low, upper=high)
                    corrections[col] = num_violations
                    logger.info("Clamped %d out-of-bounds values in %s to [%s, %s]", num_violations, col, low, high)

    return cleaned, corrections


def exclude_missing_target(df: pd.DataFrame, target_col: Optional[str] = None) -> Tuple[pd.DataFrame, int]:
    """
    Excludes rows where the supervised target column is missing.
    Strictly avoids target imputation for supervised learning integrity.
    """
    col = target_col or get_target_column(df)
    if not col or col not in df.columns:
        return df, 0

    init_len = len(df)
    filtered = df.dropna(subset=[col]).copy()
    dropped = init_len - len(filtered)
    if dropped > 0:
        logger.warning("Excluded %d rows with missing target '%s' from supervised dataset.", dropped, col)
    return filtered, dropped


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Comprehensive data cleaning function:
    - Drops exact duplicate records
    - Trims whitespace from string categories
    - Enforces academic bounds (e.g. non-negative study hours, 0-100% marks/attendance)
    - Imputes numerical input features using column median
    - Imputes categorical features using mode
    """
    if df is None or df.empty:
        raise ValueError("Cannot clean an empty or None DataFrame.")

    cleaned, _ = remove_duplicates(df)
    cleaned, _ = handle_invalid_boundaries(cleaned)

    # Clean whitespace in string columns
    for col in cleaned.select_dtypes(include=["object"]).columns:
        cleaned[col] = cleaned[col].astype(str).str.strip()

    # Median imputation for numerical input features (excluding target if missing)
    target_col = get_target_column(cleaned)
    for col in cleaned.columns:
        if cleaned[col].isnull().sum() > 0:
            if col == target_col:
                # Do NOT impute target in supervised training data
                continue
            if np.issubdtype(cleaned[col].dtype, np.number):
                med = cleaned[col].median()
                cleaned[col] = cleaned[col].fillna(med)
                logger.info("Imputed missing feature %s with median: %s", col, med)
            else:
                mode_val = cleaned[col].mode().iloc[0] if not cleaned[col].mode().empty else "Unknown"
                cleaned[col] = cleaned[col].fillna(mode_val)
                logger.info("Imputed missing categorical %s with mode: %s", col, mode_val)

    return cleaned


def generate_cleaning_report(raw_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generates a before-and-after audit report detailing rows cleaned,
    duplicates removed, and missing value resolutions.
    """
    missing_before = detect_missing_values(raw_df)
    missing_after = detect_missing_values(cleaned_df)
    duplicates_before = detect_duplicates(raw_df)["duplicate_count"]
    duplicates_after = detect_duplicates(cleaned_df)["duplicate_count"]

    column_comparison = []
    for col in raw_df.columns:
        before_cnt = missing_before["columns"].get(col, {}).get("missing_count", 0)
        after_cnt = missing_after["columns"].get(col, {}).get("missing_count", 0)
        column_comparison.append({
            "Column": col,
            "Missing (Before)": before_cnt,
            "Missing (After)": after_cnt,
            "Resolved": before_cnt - after_cnt,
        })

    return {
        "initial_rows": len(raw_df),
        "cleaned_rows": len(cleaned_df),
        "duplicates_removed": duplicates_before - duplicates_after,
        "total_missing_before": missing_before["total_missing"],
        "total_missing_after": missing_after["total_missing"],
        "columns_table": pd.DataFrame(column_comparison),
    }


# ============================================================================
# 3. FEATURE ENCODING & LEAKAGE-FREE PIPELINE PREPARATION
# ============================================================================

def get_target_column(df: pd.DataFrame) -> Optional[str]:
    """
    Identifies the active target column in the DataFrame.
    """
    for col in TARGET_COLS:
        if col in df.columns:
            return col
    return None


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes categorical features for machine learning modeling.
    Maps extracurricular_activities: 'Yes'/'True' -> 1, 'No'/'False' -> 0.
    """
    encoded = df.copy()
    if "extracurricular_activities" in encoded.columns:
        if encoded["extracurricular_activities"].dtype == "object":
            mapping = {"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0}
            encoded["extracurricular_activities"] = (
                encoded["extracurricular_activities"]
                .astype(str)
                .str.strip()
                .str.lower()
                .map(mapping)
                .fillna(0)
                .astype(int)
            )
    return encoded


def split_features_target(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    target_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Splits DataFrame into feature matrix X and target vector y.
    Excludes identifiers such as student_id.
    """
    active_target = target_col or get_target_column(df)
    if not active_target or active_target not in df.columns:
        raise KeyError(f"Target column not found in DataFrame. Expected one of {TARGET_COLS}")

    cols_to_use = feature_cols if feature_cols else [c for c in df.columns if c not in ["student_id", active_target]]
    X = df[cols_to_use].copy()
    y = df[active_target].copy()
    return X, y


def build_preprocessing_pipeline(scale_features: bool = True) -> Pipeline:
    """
    Constructs an explicit Scikit-learn Pipeline:
    1. SimpleImputer with median strategy for numerical features
    2. StandardScaler for feature scaling (if enabled)
    """
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale_features:
        steps.append(("scaler", StandardScaler()))

    return Pipeline(steps=steps)


def prepare_training_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    feature_cols: Optional[List[str]] = None,
    scale_features: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Pipeline, List[str]]:
    """
    Executes a leakage-free preprocessing workflow:
    1. Excludes records with missing target values (never imputes target)
    2. Removes duplicate records and clamps boundary violations
    3. Encodes categorical attributes
    4. Separates X and y
    5. Splits X_train, X_test, y_train, y_test BEFORE fitting transformers
    6. Fits Scikit-learn Pipeline (SimpleImputer + StandardScaler) ONLY on X_train
    7. Transforms X_train and X_test using the learned pipeline statistics
    Returns: X_train_proc, X_test_proc, y_train, y_test, pipeline, feature_names
    """
    # 1. Target exclusion
    active_target = get_target_column(df)
    clean_df, _ = exclude_missing_target(df, active_target)

    # 2. Boundary cleanup & deduplication
    clean_df, _ = remove_duplicates(clean_df)
    clean_df, _ = handle_invalid_boundaries(clean_df)

    # 3. Categorical encoding
    encoded_df = encode_features(clean_df)

    # 4. Feature-target split
    X, y = split_features_target(encoded_df, feature_cols=feature_cols, target_col=active_target)
    feature_names = list(X.columns)

    # 5. DATA LEAKAGE PREVENTION: Split BEFORE fitting any transformers
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 6. Build and fit pipeline ONLY on training data
    pipeline = build_preprocessing_pipeline(scale_features=scale_features)
    X_train_proc = pipeline.fit_transform(X_train_df)

    # 7. Apply learned statistics to test data
    X_test_proc = pipeline.transform(X_test_df)

    return X_train_proc, X_test_proc, y_train.values, y_test.values, pipeline, feature_names


def prepare_inference_features(
    attendance_percentage: float,
    previous_exam_marks: float,
    daily_study_hours: float,
    assignments_completed: int = 8,
    extracurricular_activities: str = "No",
    sleep_hours: float = 7.0,
    feature_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Converts individual user inputs into a structured, validated DataFrame ready for model inference.
    Handles None, NaN, infinite values, and strings gracefully.
    """
    def _safe_float(val, default, low, high):
        if val is None:
            return default
        try:
            f = float(val)
            if np.isnan(f):
                return default
            return float(np.clip(f, low, high))
        except (ValueError, TypeError):
            return default

    att = _safe_float(attendance_percentage, 80.0, 0.0, 100.0)
    prev = _safe_float(previous_exam_marks, 68.0, 0.0, 100.0)
    study = _safe_float(daily_study_hours, 3.5, 0.0, 24.0)
    assign = int(_safe_float(assignments_completed, 8, 0, 10))
    extra = 1 if str(extracurricular_activities).strip().lower() in ["yes", "true", "1"] else 0
    sleep = _safe_float(sleep_hours, 7.0, 0.0, 24.0)

    data = {
        "attendance_percentage": [att],
        "previous_exam_marks": [prev],
        "daily_study_hours": [study],
        "assignments_completed": [assign],
        "extracurricular_activities": [extra],
        "sleep_hours": [sleep],
    }
    df = pd.DataFrame(data)

    if feature_names:
        for f in feature_names:
            if f not in df.columns:
                df[f] = 0
        df = df[feature_names]

    return df
