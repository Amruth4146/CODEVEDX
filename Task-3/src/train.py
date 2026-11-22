"""Model Training, Validation, and Model Comparison Module for Task 3.

Responsible for:
1. Training baseline classifiers (Logistic Regression and Multinomial Naive Bayes).
2. Fitting TF-IDF vectorization strictly on Training data.
3. Evaluating performance on the isolated Validation dataset.
4. Computing comprehensive classification metrics:
   - Accuracy, Precision, Recall, F1-score (weighted and macro).
   - Class-specific metrics for Fake News (Class 0) and Real News (Class 1).
   - Confusion Matrix.
5. Implementing a clear, reproducible model selection methodology.
6. Packaging and serializing the champion model artifact into Task-3/models/.
"""

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import MultinomialNB

from src.data_loader import get_default_data_dir
from src.feature_extraction import TFIDFExtractor
from src.model_utils import (
    FakeNewsPipeline,
    get_default_models_dir,
    save_model_pipeline,
)
from src.preprocessing import clean_text_by_mode

logger = logging.getLogger(__name__)


def calculate_evaluation_metrics(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    y_pred: Sequence[int] | np.ndarray | pd.Series,
) -> dict[str, Any]:
    """Compute comprehensive classification evaluation metrics.

    Calculates overall, macro, weighted, and class-specific metrics:
    - Fake News is designated as Target Class 0.
    - Real News is designated as Target Class 1.

    Args:
        y_true: True binary target labels (0 or 1).
        y_pred: Predicted binary labels (0 or 1).

    Returns:
        Dictionary of formatted metrics and confusion matrix counts.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)

    acc = float(accuracy_score(y_true_arr, y_pred_arr))

    # Overall / Weighted metrics
    f1_weighted = float(f1_score(y_true_arr, y_pred_arr, average="weighted"))
    prec_weighted = float(precision_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0))
    rec_weighted = float(recall_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0))

    # Macro metrics
    f1_macro = float(f1_score(y_true_arr, y_pred_arr, average="macro"))
    prec_macro = float(precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))

    # Fake News specific metrics (Class 0)
    fake_prec = float(precision_score(y_true_arr, y_pred_arr, pos_label=0, zero_division=0))
    fake_rec = float(recall_score(y_true_arr, y_pred_arr, pos_label=0, zero_division=0))
    fake_f1 = float(f1_score(y_true_arr, y_pred_arr, pos_label=0, zero_division=0))

    # Real News specific metrics (Class 1)
    real_prec = float(precision_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0))
    real_rec = float(recall_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0))
    real_f1 = float(f1_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0))

    # Confusion Matrix: [[TN, FP], [FN, TP]]
    # row 0 = actual Fake (0), row 1 = actual Real (1)
    # col 0 = pred Fake (0),   col 1 = pred Real (1)
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    return {
        "accuracy": round(acc, 4),
        "f1_weighted": round(f1_weighted, 4),
        "f1_macro": round(f1_macro, 4),
        "precision_weighted": round(prec_weighted, 4),
        "recall_weighted": round(rec_weighted, 4),
        "precision_macro": round(prec_macro, 4),
        "recall_macro": round(rec_macro, 4),
        "fake_precision": round(fake_prec, 4),
        "fake_recall": round(fake_rec, 4),
        "fake_f1": round(fake_f1, 4),
        "real_precision": round(real_prec, 4),
        "real_recall": round(real_rec, 4),
        "real_f1": round(real_f1, 4),
        "confusion_matrix": cm.tolist(),
        "true_fake": tn,
        "false_real": fp,
        "false_fake": fn,
        "true_real": tp,
    }


def train_logistic_regression(
    X_train: Any,
    y_train: np.ndarray,
    C: float = 1.0,
    max_iter: int = 1000,
    random_state: int = 42,
) -> LogisticRegression:
    """Train a Logistic Regression binary classifier."""
    clf = LogisticRegression(
        C=C,
        max_iter=max_iter,
        random_state=random_state,
        solver="lbfgs",
    )
    clf.fit(X_train, y_train)
    return clf


def train_multinomial_nb(
    X_train: Any,
    y_train: np.ndarray,
    alpha: float = 1.0,
) -> MultinomialNB:
    """Train a Multinomial Naive Bayes classifier."""
    clf = MultinomialNB(alpha=alpha)
    clf.fit(X_train, y_train)
    return clf


def select_best_model(metrics_by_model: dict[str, dict[str, Any]]) -> str:
    """Select the winning model using a principled multi-criteria methodology.

    Selection Criteria Priority:
    1. Highest Validation Weighted F1-score (balances overall precision and recall).
    2. Highest Fake News F1-score (measures effectiveness on the core misinformation task).
    3. Highest Accuracy.

    Args:
        metrics_by_model: Dictionary mapping model names to their metrics dictionary.

    Returns:
        Name of the winning model.
    """
    if not metrics_by_model:
        raise ValueError("Cannot select best model from empty metrics dictionary.")

    # Sort candidates by (weighted F1, fake F1, accuracy)
    sorted_candidates = sorted(
        metrics_by_model.keys(),
        key=lambda name: (
            metrics_by_model[name].get("f1_weighted", 0.0),
            metrics_by_model[name].get("fake_f1", 0.0),
            metrics_by_model[name].get("accuracy", 0.0),
        ),
        reverse=True,
    )

    return sorted_candidates[0]


def train_and_compare_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    preprocessing_mode: str = "standard",
    max_features: int = 5000,
    min_df: int | float = 2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train, evaluate, and compare Logistic Regression and Multinomial Naive Bayes.

    Pipeline Steps:
    1. Preprocesses train and validation content using the specified cleaning mode.
    2. Fits TFIDFExtractor strictly on train texts.
    3. Transforms validation texts using the fitted vectorizer (zero leakage).
    4. Trains Logistic Regression and Multinomial Naive Bayes on X_train.
    5. Evaluates both on X_val.
    6. Selects the champion model based on validation F1 metrics.
    7. Builds deployable FakeNewsPipeline artifacts for both.

    Args:
        train_df: DataFrame containing 'content' and 'label' for training.
        val_df: DataFrame containing 'content' and 'label' for validation.
        preprocessing_mode: Cleaning mode ('standard', 'minimal', or 'aggressive').
        max_features: Maximum TF-IDF vocabulary features.
        min_df: Minimum document frequency for TF-IDF.
        random_state: Random state for reproducibility.

    Returns:
        Dictionary containing metrics, trained pipelines, and the winning model name.
    """
    if len(train_df) == 0 or len(val_df) == 0:
        raise ValueError("Training and validation DataFrames must be non-empty.")

    # 1. Preprocess texts
    logger.info("Preprocessing training and validation texts (mode='%s')...", preprocessing_mode)
    train_texts = [clean_text_by_mode(t, mode=preprocessing_mode) for t in train_df["content"]]
    val_texts = [clean_text_by_mode(t, mode=preprocessing_mode) for t in val_df["content"]]

    y_train = train_df["label"].values.astype(int)
    y_val = val_df["label"].values.astype(int)

    # 2. Fit TF-IDF on Training text ONLY
    logger.info("Fitting TF-IDF extractor strictly on training text...")
    effective_min_df = 1 if len(train_df) < 20 else min_df
    effective_max_df = 1.0 if len(train_df) < 20 else 0.95
    extractor = TFIDFExtractor(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=effective_min_df,
        max_df=effective_max_df,
        sublinear_tf=True,
    )
    X_train = extractor.fit_transform(train_texts)
    X_val = extractor.transform(val_texts)

    # 3. Train Model 1: Logistic Regression
    logger.info("Training Logistic Regression...")
    t0 = time.time()
    lr_clf = train_logistic_regression(X_train, y_train, random_state=random_state)
    lr_train_time = round(time.time() - t0, 3)
    lr_preds = lr_clf.predict(X_val)
    lr_metrics = calculate_evaluation_metrics(y_val, lr_preds)
    lr_metrics["train_time_sec"] = lr_train_time

    lr_pipeline = FakeNewsPipeline(
        classifier=lr_clf,
        feature_extractor=extractor,
        model_name="Logistic Regression",
        preprocessing_mode=preprocessing_mode,
        metrics=lr_metrics,
    )

    # 4. Train Model 2: Multinomial Naive Bayes
    logger.info("Training Multinomial Naive Bayes...")
    t0 = time.time()
    nb_clf = train_multinomial_nb(X_train, y_train)
    nb_train_time = round(time.time() - t0, 3)
    nb_preds = nb_clf.predict(X_val)
    nb_metrics = calculate_evaluation_metrics(y_val, nb_preds)
    nb_metrics["train_time_sec"] = nb_train_time

    nb_pipeline = FakeNewsPipeline(
        classifier=nb_clf,
        feature_extractor=extractor,
        model_name="Multinomial Naive Bayes",
        preprocessing_mode=preprocessing_mode,
        metrics=nb_metrics,
    )

    # 5. Model Comparison and Selection
    metrics_by_model = {
        "Logistic Regression": lr_metrics,
        "Multinomial Naive Bayes": nb_metrics,
    }
    pipelines_by_model = {
        "Logistic Regression": lr_pipeline,
        "Multinomial Naive Bayes": nb_pipeline,
    }

    best_name = select_best_model(metrics_by_model)
    best_pipeline = pipelines_by_model[best_name]

    return {
        "metrics": metrics_by_model,
        "pipelines": pipelines_by_model,
        "best_model_name": best_name,
        "best_pipeline": best_pipeline,
        "feature_count": extractor.vocabulary_size,
    }


def run_training_pipeline(
    data_dir: Path | str | None = None,
    models_dir: Path | str | None = None,
    preprocessing_mode: str = "standard",
) -> dict[str, Any]:
    """Execute end-to-end model training from processed CSV files and persist artifacts.

    Args:
        data_dir: Path to Task-3/data directory.
        models_dir: Path to Task-3/models directory.
        preprocessing_mode: Text cleaning mode.

    Returns:
        Dictionary summarizing execution results and artifact paths.
    """
    base_data = Path(data_dir).resolve() if data_dir else get_default_data_dir()
    base_models = Path(models_dir).resolve() if models_dir else get_default_models_dir()

    processed_dir = base_data / "processed"
    train_file = processed_dir / "train.csv"
    val_file = processed_dir / "val.csv"

    if not train_file.exists() or not val_file.exists():
        raise FileNotFoundError(
            f"Required processed split files not found in '{processed_dir}'. "
            f"Please ensure Phase 5.2 dataset splitting has been run."
        )

    logger.info("Loading processed splits from '%s'...", processed_dir)
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)

    results = train_and_compare_models(
        train_df=train_df,
        val_df=val_df,
        preprocessing_mode=preprocessing_mode,
    )

    # Save artifacts
    base_models.mkdir(parents=True, exist_ok=True)
    saved_paths: dict[str, Path] = {}

    # Save champion model
    champion_path = base_models / "fake_news_model.joblib"
    save_model_pipeline(results["best_pipeline"], champion_path)
    saved_paths["champion"] = champion_path

    # Save comparison models
    lr_path = base_models / "logistic_regression.joblib"
    nb_path = base_models / "multinomial_nb.joblib"
    save_model_pipeline(results["pipelines"]["Logistic Regression"], lr_path)
    save_model_pipeline(results["pipelines"]["Multinomial Naive Bayes"], nb_path)
    saved_paths["logistic_regression"] = lr_path
    saved_paths["multinomial_nb"] = nb_path

    results["saved_paths"] = saved_paths
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("Running Model Training Pipeline...")
    res = run_training_pipeline()
    print("\n--- Training Results Summary ---")
    print("Winning Model:", res["best_model_name"])
    for m_name, m_stats in res["metrics"].items():
        print(f"\n{m_name}:")
        for k, v in m_stats.items():
            print(f"  {k}: {v}")
    print("\nSaved Artifacts:")
    for k, p in res["saved_paths"].items():
        print(f"  {k} -> {p}")
