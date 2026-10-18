"""
Data loader module for Student Performance Prediction System.
Handles loading, caching, benchmark data generation, and exporting of student records.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Base Paths (relative to project root)
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "student_data.csv"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_student_data.csv"


def generate_benchmark_dataset(
    num_samples: int = 750,
    random_state: int = 42,
    save_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Generates a realistic benchmark student academic dataset.
    Features include attendance, previous marks, study hours, assignments,
    extracurricular involvement, and real-world final exam performance.
    """
    logger.info("Generating benchmark student dataset with %d samples...", num_samples)
    np.random.seed(random_state)
    
    # Target file path
    target_path = save_path or RAW_DATA_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Student IDs
    student_ids = [f"STU-{1001 + i}" for i in range(num_samples)]

    # 2. Previous Exam Marks (30 to 98, mean ~68)
    previous_marks = np.clip(np.random.normal(loc=68.0, scale=13.5, size=num_samples), 30.0, 98.0)

    # 3. Attendance Percentage (45% to 100%, correlated with previous marks)
    attendance = np.clip(
        0.35 * previous_marks + np.random.normal(loc=52.0, scale=8.5, size=num_samples),
        42.0,
        100.0,
    )

    # 4. Daily Study Hours (0.5 to 11.0 hrs, mean ~4.2 hrs)
    study_hours = np.clip(np.random.gamma(shape=4.0, scale=1.1, size=num_samples), 0.5, 11.0)

    # 5. Assignments Completed (out of 10)
    assignments = np.clip(
        np.round(np.random.normal(loc=7.8, scale=1.8, size=num_samples)),
        1,
        10,
    ).astype(int)

    # 6. Extracurricular Activities
    extracurricular = np.random.choice(["Yes", "No"], size=num_samples, p=[0.48, 0.52])

    # 7. Sleep Hours (4.5 to 9.5)
    sleep_hours = np.clip(np.random.normal(loc=7.0, scale=1.0, size=num_samples), 4.5, 9.5)

    # 8. Final Exam Marks (Ground truth relationship with realistic noise)
    extra_boost = np.where(extracurricular == "Yes", 2.0, 0.0)
    noise = np.random.normal(loc=0.0, scale=3.2, size=num_samples)

    final_marks = (
        0.36 * previous_marks
        + 0.38 * attendance
        + 2.65 * study_hours
        + 1.15 * assignments
        + extra_boost
        - 21.5
        + noise
    )
    final_marks = np.clip(np.round(final_marks, 1), 18.0, 100.0)

    df = pd.DataFrame({
        "student_id": student_ids,
        "attendance_percentage": np.round(attendance, 1),
        "previous_exam_marks": np.round(previous_marks, 1),
        "daily_study_hours": np.round(study_hours, 1),
        "assignments_completed": assignments,
        "extracurricular_activities": extracurricular,
        "sleep_hours": np.round(sleep_hours, 1),
        "final_exam_marks": final_marks,
    })

    # Save to disk
    df.to_csv(target_path, index=False)
    logger.info("Saved benchmark dataset to: %s", target_path)
    return df


def load_raw_data(filepath: Optional[str | Path] = None) -> pd.DataFrame:
    """
    Loads the raw student academic dataset.
    If the file does not exist, automatically generates a high-quality benchmark dataset.
    """
    path = Path(filepath) if filepath else RAW_DATA_PATH
    try:
        if not path.exists():
            logger.warning("Raw data file not found at %s. Creating benchmark dataset.", path)
            return generate_benchmark_dataset(save_path=path)
        
        df = pd.read_csv(path)
        logger.info("Successfully loaded raw dataset (%d rows, %d cols) from %s", len(df), len(df.columns), path)
        return df
    except Exception as exc:
        logger.error("Error loading raw data from %s: %s", path, exc)
        raise RuntimeError(f"Failed to load raw data: {exc}") from exc


def save_processed_data(df: pd.DataFrame, filepath: Optional[str | Path] = None) -> Path:
    """
    Saves a cleaned or preprocessed DataFrame to CSV.
    """
    path = Path(filepath) if filepath else PROCESSED_DATA_PATH
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info("Successfully saved processed data to %s", path)
        return path
    except Exception as exc:
        logger.error("Failed to save processed data: %s", exc)
        raise


def load_processed_data(filepath: Optional[str | Path] = None) -> pd.DataFrame:
    """
    Loads preprocessed student dataset.
    """
    path = Path(filepath) if filepath else PROCESSED_DATA_PATH
    if not path.exists():
        logger.info("Processed data not found at %s. Cleaning raw data...", path)
        from src.preprocessing import clean_data
        raw_df = load_raw_data()
        cleaned_df = clean_data(raw_df)
        save_processed_data(cleaned_df, path)
        return cleaned_df
    return pd.read_csv(path)


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Returns high-level metadata and diagnostic statistics for a DataFrame.
    """
    return {
        "total_records": int(len(df)),
        "total_columns": int(len(df.columns)),
        "columns": list(df.columns),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_columns": list(df.select_dtypes(include=[np.number]).columns),
        "categorical_columns": list(df.select_dtypes(include=["object", "category"]).columns),
    }
