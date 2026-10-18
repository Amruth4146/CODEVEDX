"""
Complete Machine Learning Training Workflow Module.
Student Performance Prediction System - CODEVEDX Task 2

Implements:
- End-to-end dataset loading, validation, and feature separation
- 80/20 train-test splitting with random_state=42
- Scikit-learn Pipeline construction (imputation + scaling + model)
- Training-only cross-validation (5-fold K-Fold)
- Hyperparameter tuning for Random Forest using GridSearchCV
- Model selection using lowest cross-validation RMSE
- Unbiased evaluation on untouched test set (MAE, MSE, RMSE, R²)
- Strict regression metric labeling (R² is never labeled as accuracy percentage)
- Schema validation to prevent stale model usage
- Joblib serialization of the complete Pipeline artifact
"""

import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV, cross_validate, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Ensure project root in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.data_loader import load_raw_data, save_processed_data
from src.preprocessing import clean_data, exclude_missing_target, encode_features

logger = logging.getLogger(__name__)

# Base Artifact Directory
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "student_performance_model.joblib"

TARGET_COLUMNS = ["final_exam_marks", "final_exam_score"]
PRIMARY_TARGET = "final_exam_marks"


def get_target_col(df: pd.DataFrame) -> str:
    """Identifies the active target column in the dataset."""
    for col in TARGET_COLUMNS:
        if col in df.columns:
            return col
    return PRIMARY_TARGET


def validate_dataset_schema(df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
    """
    Validates dataset integrity before training:
    - Verifies sufficient sample count
    - Verifies presence of required target column
    - Validates presence of core predictors
    """
    if df is None or len(df) < 50:
        return False, "Dataset contains insufficient samples for training (minimum 50 required)."

    target = get_target_col(df)
    if target not in df.columns:
        return False, f"Target column not found. Expected one of {TARGET_COLUMNS}."

    core_predictors = ["attendance_percentage", "daily_study_hours"]
    for cp in core_predictors:
        if cp not in df.columns:
            # Check for alias
            alias = "study_hours_per_day" if cp == "daily_study_hours" else None
            if alias is None or alias not in df.columns:
                return False, f"Mandatory predictor '{cp}' missing from dataset schema."

    return True, None


def load_and_prepare_dataset(
    raw_data_path: Optional[str | Path] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str], str]:
    """
    Loads raw dataset, sanitizes boundaries, removes duplicates, excludes missing targets,
    and performs a leakage-free 80/20 train-test split.
    Returns: X_train, X_test, y_train, y_test, feature_names, target_name
    """
    raw_df = load_raw_data(raw_data_path)

    valid, err_msg = validate_dataset_schema(raw_df)
    if not valid:
        logger.error("Dataset validation error: %s", err_msg)
        raise ValueError(f"Schema validation failure: {err_msg}")

    target_name = get_target_col(raw_df)

    # 1. Exclude rows with missing target (never impute target)
    clean_df, dropped_target = exclude_missing_target(raw_df, target_col=target_name)
    if dropped_target > 0:
        logger.warning("Dropped %d records missing target '%s'.", dropped_target, target_name)

    # 2. Clean boundaries & duplicates
    clean_df = clean_data(clean_df)
    save_processed_data(clean_df)

    # 3. Encode categoricals if present
    encoded_df = encode_features(clean_df)

    # 4. Separate X and y (target strictly excluded from X)
    ignored = set(TARGET_COLUMNS + ["student_id", "performance_category"])
    feature_cols = [c for c in encoded_df.columns if c not in ignored]

    X = encoded_df[feature_cols].copy()
    y = encoded_df[target_name].copy()

    assert target_name not in X.columns, f"Data leakage error: Target '{target_name}' found in feature matrix!"

    # 5. Split into 80% train and 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    logger.info(
        "Prepared dataset: %d train samples (80%%), %d test samples (20%%), %d features.",
        len(X_train), len(X_test), len(feature_cols)
    )

    return X_train, X_test, y_train, y_test, feature_cols, target_name


def build_pipeline_architectures(random_state: int = 42) -> Dict[str, Pipeline]:
    """
    Builds Scikit-learn Pipeline objects pairing preprocessing with regressors:
    1. Linear Regression Pipeline (Imputer + StandardScaler + LinearRegression)
    2. Random Forest Regressor Pipeline (Imputer + RandomForestRegressor)
    """
    return {
        "Linear Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]),
        "Random Forest Regressor": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("regressor", RandomForestRegressor(random_state=random_state, n_jobs=-1)),
        ]),
    }


def get_rf_search_space() -> Dict[str, list]:
    """
    Returns a practical, fast hyperparameter search grid for Random Forest,
    optimized to train quickly on a standard student laptop.
    """
    return {
        "regressor__n_estimators": [60, 100],
        "regressor__max_depth": [4, 6, 8],
        "regressor__min_samples_split": [2, 5],
    }


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes standard regression metrics:
    - MAE: Mean Absolute Error
    - MSE: Mean Squared Error
    - RMSE: Root Mean Squared Error
    - R²: Coefficient of Determination (strictly NOT labeled as accuracy %)
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    # Clamped to realistic examination range
    y_pred_arr = np.clip(y_pred_arr, 0.0, 100.0)

    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    mse = float(mean_squared_error(y_true_arr, y_pred_arr))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true_arr, y_pred_arr))

    return {
        "mae": round(mae, 2),
        "mse": round(mse, 2),
        "rmse": round(rmse, 2),
        "r2": round(r2, 4),
    }


def train_and_select_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int = 42,
    cv_folds: int = 5,
) -> Tuple[str, Pipeline, Dict[str, Any], Dict[str, Dict[str, Any]], Dict[str, np.ndarray]]:
    """
    Executes training workflow:
    1. 5-Fold cross-validation on X_train, y_train for Linear Regression.
    2. 5-Fold GridSearchCV on X_train, y_train for Random Forest.
    3. Selects best model using lowest Cross-Validation RMSE.
    4. Evaluates models on the untouched test set (X_test, y_test).
    Returns: best_name, best_pipeline, best_params, model_results, test_predictions
    """
    pipelines = build_pipeline_architectures(random_state=random_state)
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scoring = ["neg_root_mean_squared_error", "neg_mean_absolute_error", "neg_mean_squared_error", "r2"]

    model_results: Dict[str, Dict[str, Any]] = {}
    fitted_pipelines: Dict[str, Pipeline] = {}
    test_preds: Dict[str, np.ndarray] = {}
    best_params_dict: Dict[str, Any] = {}

    # ------------------------------------------------------------------------
    # 1. Linear Regression
    # ------------------------------------------------------------------------
    logger.info("Executing 5-Fold Cross-Validation for Linear Regression...")
    lr_pipe = pipelines["Linear Regression"]
    lr_cv = cross_validate(lr_pipe, X_train, y_train, cv=kf, scoring=scoring, return_train_score=False)

    lr_cv_rmse = float(abs(np.mean(lr_cv["test_neg_root_mean_squared_error"])))
    lr_cv_mae = float(abs(np.mean(lr_cv["test_neg_mean_absolute_error"])))
    lr_cv_mse = float(abs(np.mean(lr_cv["test_neg_mean_squared_error"])))
    lr_cv_r2 = float(np.mean(lr_cv["test_r2"]))
    lr_fold_rmse = [round(float(abs(x)), 2) for x in lr_cv["test_neg_root_mean_squared_error"]]

    # Fit Linear Regression on full training split
    lr_pipe.fit(X_train, y_train)
    lr_test_pred = np.clip(lr_pipe.predict(X_test), 0.0, 100.0)
    lr_test_metrics = evaluate_predictions(y_test.values, lr_test_pred)

    fitted_pipelines["Linear Regression"] = lr_pipe
    test_preds["Linear Regression"] = lr_test_pred
    model_results["Linear Regression"] = {
        "cv_rmse": round(lr_cv_rmse, 2),
        "cv_mae": round(lr_cv_mae, 2),
        "cv_mse": round(lr_cv_mse, 2),
        "cv_r2": round(lr_cv_r2, 4),
        "cv_fold_rmse": lr_fold_rmse,
        "test_metrics": lr_test_metrics,
        "best_params": "N/A (Standard OLS)",
    }
    logger.info("Linear Regression -> CV RMSE: %.2f | Test RMSE: %.2f | Test R²: %.4f", lr_cv_rmse, lr_test_metrics["rmse"], lr_test_metrics["r2"])

    # ------------------------------------------------------------------------
    # 2. Random Forest Regressor (GridSearchCV Hyperparameter Optimization)
    # ------------------------------------------------------------------------
    logger.info("Executing GridSearchCV optimization for Random Forest Regressor...")
    rf_pipe = pipelines["Random Forest Regressor"]
    rf_param_grid = get_rf_search_space()

    grid_search = GridSearchCV(
        rf_pipe,
        rf_param_grid,
        cv=kf,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        refit=True,
    )
    grid_search.fit(X_train, y_train)

    best_rf_pipe = grid_search.best_estimator_
    best_rf_params = grid_search.best_params_
    rf_cv_rmse = float(abs(grid_search.best_score_))

    # Evaluate CV metrics on best estimator
    rf_cv = cross_validate(best_rf_pipe, X_train, y_train, cv=kf, scoring=scoring)
    rf_cv_mae = float(abs(np.mean(rf_cv["test_neg_mean_absolute_error"])))
    rf_cv_mse = float(abs(np.mean(rf_cv["test_neg_mean_squared_error"])))
    rf_cv_r2 = float(np.mean(rf_cv["test_r2"]))
    rf_fold_rmse = [round(float(abs(x)), 2) for x in rf_cv["test_neg_root_mean_squared_error"]]

    rf_test_pred = np.clip(best_rf_pipe.predict(X_test), 0.0, 100.0)
    rf_test_metrics = evaluate_predictions(y_test.values, rf_test_pred)

    fitted_pipelines["Random Forest Regressor"] = best_rf_pipe
    test_preds["Random Forest Regressor"] = rf_test_pred
    best_params_dict["Random Forest Regressor"] = best_rf_params
    model_results["Random Forest Regressor"] = {
        "cv_rmse": round(rf_cv_rmse, 2),
        "cv_mae": round(rf_cv_mae, 2),
        "cv_mse": round(rf_cv_mse, 2),
        "cv_r2": round(rf_cv_r2, 4),
        "cv_fold_rmse": rf_fold_rmse,
        "test_metrics": rf_test_metrics,
        "best_params": best_rf_params,
    }
    logger.info("Random Forest Regressor -> CV RMSE: %.2f | Test RMSE: %.2f | Test R²: %.4f", rf_cv_rmse, rf_test_metrics["rmse"], rf_test_metrics["r2"])

    # ------------------------------------------------------------------------
    # 3. Model Selection Criterion: Lowest Cross-Validation RMSE
    # ------------------------------------------------------------------------
    if lr_cv_rmse < rf_cv_rmse:
        best_name = "Linear Regression"
        best_pipeline = fitted_pipelines["Linear Regression"]
        best_params = "Standard OLS Closed-Form Solution"
    else:
        best_name = "Random Forest Regressor"
        best_pipeline = fitted_pipelines["Random Forest Regressor"]
        best_params = best_rf_params

    logger.info(
        "Optimal Model Selected: '%s' based on lowest CV RMSE (%.2f vs %.2f).",
        best_name, model_results[best_name]["cv_rmse"],
        model_results["Random Forest Regressor" if best_name == "Linear Regression" else "Linear Regression"]["cv_rmse"]
    )

    return best_name, best_pipeline, best_params, model_results, test_preds


def save_model_artifacts(
    pipeline: Pipeline,
    model_name: str,
    feature_names: List[str],
    test_metrics: Dict[str, float],
    model_results: Dict[str, Dict[str, Any]],
    best_params: Any,
    training_samples: int,
    filepath: Optional[Path] = None,
) -> Path:
    """
    Serializes the complete trained pipeline (including imputer & scaler)
    along with metadata and schema definitions using Joblib.
    """
    target_path = filepath or MODEL_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Feature schema fingerprint to prevent stale model usage
    feature_schema = {f: "float64" for f in feature_names}

    # Dataset fingerprint (SHA-256)
    import hashlib
    fp_str = f"samples={training_samples}|features={sorted(feature_names)}|target={PRIMARY_TARGET}|seed=42"
    sha = hashlib.sha256(fp_str.encode("utf-8")).hexdigest()
    fingerprint_dict = {
        "sha256": sha,
        "short_hash": sha[:12],
        "training_rows": training_samples,
        "features": feature_names,
    }

    bundle = {
        "pipeline": pipeline,
        "model": pipeline,                      # Backward compatibility
        "scaler": pipeline.named_steps.get("scaler"),  # Backward compatibility
        "model_name": model_name,
        "selection_criterion": "Lowest Cross-Validation RMSE",
        "feature_names": feature_names,
        "feature_schema": feature_schema,
        "dataset_fingerprint": fingerprint_dict,
        "metrics": test_metrics,                # Backward compatibility
        "test_metrics": test_metrics,
        "all_models_metrics": {m: v["test_metrics"] for m, v in model_results.items()},
        "cv_results": model_results,
        "best_params": best_params,
        "training_samples": training_samples,
        "training_config": {
            "test_size": 0.2,
            "random_state": 42,
            "cv_folds": 5,
        },
        "trained_at": datetime.now().isoformat(),
        "version": "2.0.0",
    }

    joblib.dump(bundle, target_path)
    logger.info("Successfully serialized trained ML pipeline to %s", target_path)
    return target_path


def load_model_bundle(filepath: Optional[Path] = None) -> Dict[str, Any]:
    """
    Loads serialized model artifact bundle from disk.
    If absent, automatically triggers the training workflow.
    """
    target_path = filepath or MODEL_PATH
    if not target_path.exists():
        logger.warning("Artifact not found at %s. Triggering training workflow...", target_path)
        bundle, _, _ = train_pipeline(save_path=target_path)
        return bundle

    bundle = joblib.load(target_path)
    from src.security import validate_model_bundle_safety
    is_safe, err = validate_model_bundle_safety(bundle)
    if not is_safe:
        logger.error("Loaded model bundle failed integrity check: %s. Re-training pipeline...", err)
        bundle, _, _ = train_pipeline(save_path=target_path)
    return bundle


def validate_loaded_model_schema(bundle: Dict[str, Any], input_df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
    """
    Validates that an incoming dataset matches the schema required by the serialized pipeline.
    Prevents stale models from executing on incompatible features.
    """
    expected_features = bundle.get("feature_names", [])
    missing = [f for f in expected_features if f not in input_df.columns]
    if missing:
        return False, f"Incompatible feature schema. Missing expected predictors: {missing}"
    return True, None


def train_pipeline(
    raw_data_path: Optional[str | Path] = None,
    save_path: Optional[Path] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    cv_folds: int = 5,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """
    High-level training pipeline orchestrator:
    1. Loads, validates, and splits data into 80% train and 20% test
    2. Performs 5-fold cross-validation on Linear Regression
    3. Tunes Random Forest using GridSearchCV
    4. Selects optimal model based on lowest cross-validation RMSE
    5. Evaluates on untouched test split
    6. Persists the winning Scikit-learn Pipeline with Joblib
    """
    try:
        logger.info("Starting complete Machine Learning training workflow...")

        # Step 1: Load, validate, split data (80% train, 20% test)
        X_train, X_test, y_train, y_test, feature_names, target_name = load_and_prepare_dataset(
            raw_data_path=raw_data_path,
            test_size=test_size,
            random_state=random_state,
        )

        # Step 2: Cross-validate, tune, and select best model
        best_name, best_pipeline, best_params, model_results, test_preds = train_and_select_models(
            X_train, y_train, X_test, y_test,
            random_state=random_state,
            cv_folds=cv_folds,
        )

        # Step 3: Serialize pipeline & metadata
        winning_test_metrics = model_results[best_name]["test_metrics"]
        saved_file = save_model_artifacts(
            pipeline=best_pipeline,
            model_name=best_name,
            feature_names=feature_names,
            test_metrics=winning_test_metrics,
            model_results=model_results,
            best_params=best_params,
            training_samples=len(X_train),
            filepath=save_path,
        )

        bundle = {
            "pipeline": best_pipeline,
            "model": best_pipeline,
            "scaler": best_pipeline.named_steps.get("scaler"),
            "model_name": best_name,
            "metrics": winning_test_metrics,
            "test_metrics": winning_test_metrics,
            "cv_results": model_results,
            "all_models_metrics": {m: v["test_metrics"] for m, v in model_results.items()},
            "best_params": best_params,
            "feature_names": feature_names,
            "saved_file": str(saved_file),
            "trained_at": datetime.now().isoformat(),
        }

        logger.info("Training workflow successfully concluded. Deployed model: %s", best_name)
        return bundle, y_test.values, test_preds[best_name]

    except Exception as exc:
        logger.error("Error encountered during ML training workflow: %s", exc, exc_info=True)
        raise RuntimeError(f"Training pipeline execution failure: {exc}") from exc


if __name__ == "__main__":
    train_pipeline()
