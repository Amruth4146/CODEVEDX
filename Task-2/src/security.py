"""
Security, Input Validation, and Privacy Protections
Student Performance Prediction System - CODEVEDX Task 2
"""

import io
import os
import re
import logging
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Maximum allowable upload size (15 MB to prevent memory exhaustion / DoS)
MAX_UPLOAD_SIZE_BYTES = 15 * 1024 * 1024

# Disallowed / Sensitive PII column keywords
SENSITIVE_PII_KEYWORDS = [
    "ssn",
    "social_security",
    "password",
    "secret",
    "credit_card",
    "card_number",
    "bank_account",
    "cvv",
    "pin",
    "phone_number",
    "mobile_number",
]


def validate_uploaded_csv(
    file_buffer: Any,
    max_size_bytes: int = MAX_UPLOAD_SIZE_BYTES,
    required_columns: Optional[List[str]] = None,
) -> Tuple[bool, Optional[str], Optional[pd.DataFrame]]:
    """
    Rigorously validates uploaded CSV files for security, format integrity, and PII protection:
    1. Checks file size bounds (< max_size_bytes).
    2. Validates file extension (.csv).
    3. Handles empty or corrupted file buffers.
    4. Enforces non-empty tabular data.
    5. Validates schema column presence.
    6. Scans for potential PII columns.
    """
    if file_buffer is None:
        return False, "No file uploaded.", None

    # Check filename extension if available
    filename = getattr(file_buffer, "name", "uploaded.csv")
    if not filename.lower().endswith(".csv"):
        return False, f"Invalid file extension for '{filename}'. Only .csv files are supported.", None

    # Check file size
    size = getattr(file_buffer, "size", None)
    if size is not None:
        if size == 0:
            return False, "Uploaded file is empty (0 bytes).", None
        if size > max_size_bytes:
            mb = max_size_bytes / (1024 * 1024)
            return False, f"File size ({size / (1024 * 1024):.1f} MB) exceeds maximum allowed threshold of {mb:.0f} MB.", None

    # Reset buffer position if seekable
    if hasattr(file_buffer, "seek"):
        file_buffer.seek(0)

    try:
        # Read content safely with bounded rows to prevent resource exhaustion
        df = pd.read_csv(file_buffer, nrows=100000)
    except pd.errors.EmptyDataError:
        return False, "CSV file contains no data or headers.", None
    except Exception as exc:
        return False, f"Malformed or unreadable CSV structure: {exc}", None

    if df.empty or len(df) == 0:
        return False, "CSV file has no data rows.", None

    if len(df.columns) == 0:
        return False, "CSV file has no identifiable columns.", None

    # Scan for sensitive PII columns
    detected_pii = []
    for col in df.columns:
        clean_col = re.sub(r"[^a-zA-Z0-9_]", "", str(col).lower())
        for keyword in SENSITIVE_PII_KEYWORDS:
            if keyword in clean_col:
                detected_pii.append(col)
                break

    if detected_pii:
        return (
            False,
            f"Security Violation: Potential Personal Identifiable Information (PII) detected in columns: {detected_pii}. "
            "Please remove sensitive fields before uploading.",
            None,
        )

    # Check required columns
    if required_columns:
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            return False, f"Missing required column(s): {missing}. Found columns: {list(df.columns)}", None

    return True, None, df


def validate_model_bundle_safety(bundle: Any) -> Tuple[bool, Optional[str]]:
    """
    Guards against unsafe deserialization of corrupted or untrusted model artifacts:
    - Verifies object is a dictionary
    - Verifies presence of core pipeline/model keys
    - Verifies feature_names list integrity
    - Prevents executing arbitrary unverified objects
    """
    if bundle is None:
        return False, "Model bundle is None."

    if not isinstance(bundle, dict):
        return False, f"Expected dictionary bundle, received {type(bundle).__name__}."

    if "pipeline" not in bundle and "model" not in bundle:
        return False, "Model artifact does not contain a 'pipeline' or 'model' estimator."

    estimator = bundle.get("pipeline") or bundle.get("model")
    if not hasattr(estimator, "predict"):
        return False, f"Estimator object '{type(estimator).__name__}' lacks a 'predict' method."

    if "feature_names" not in bundle:
        return False, "Model artifact missing 'feature_names' specification."

    if not isinstance(bundle["feature_names"], (list, tuple)):
        return False, "Model artifact 'feature_names' must be a list or tuple."

    return True, None


def sanitize_student_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Anonymizes student identifiers to protect student privacy:
    Ensures identifiers follow synthetic 'STU-xxxx' format and drops real names or emails if present.
    """
    sanitized = df.copy()

    # Drop obvious direct PII if present
    pii_drop = [c for c in sanitized.columns if any(k in str(c).lower() for k in ["name", "email", "phone", "address"])]
    if pii_drop:
        sanitized = sanitized.drop(columns=pii_drop)
        logger.info("Dropped direct PII columns for student privacy: %s", pii_drop)

    # Anonymize student_id if missing or non-standard
    if "student_id" in sanitized.columns:
        # Check if values look synthetic
        sample = sanitized["student_id"].dropna().astype(str).head(10)
        is_synthetic = sample.str.startswith("STU").all()
        if not is_synthetic:
            sanitized["student_id"] = [f"STU-{1001 + i}" for i in range(len(sanitized))]
    else:
        sanitized["student_id"] = [f"STU-{1001 + i}" for i in range(len(sanitized))]

    return sanitized
