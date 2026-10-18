"""
Unit tests for model prediction and inference engine.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.prediction import ModelPredictor


@pytest.fixture(scope="module")
def predictor():
    return ModelPredictor()


def test_predictor_loads_properly(predictor):
    assert predictor.model is not None
    assert predictor.model_name != ""
    assert isinstance(predictor.feature_names, list)


def test_predict_returns_valid_bounds(predictor):
    result = predictor.predict(
        attendance_percentage=85.0,
        previous_exam_marks=78.0,
        daily_study_hours=4.5,
    )
    score = result["predicted_score"]
    assert 0.0 <= score <= 100.0
    assert "classification" in result
    assert result["classification"]["category"] in [
        "Distinction",
        "First Class",
        "Pass",
        "At-Risk (Fail)",
    ]


def test_predict_batch(predictor):
    test_df = pd.DataFrame({
        "attendance_percentage": [95.0, 50.0, 80.0],
        "previous_exam_marks": [90.0, 40.0, 75.0],
        "daily_study_hours": [7.0, 1.5, 4.0],
        "assignments_completed": [10, 4, 8],
        "extracurricular_activities": ["Yes", "No", "No"],
        "sleep_hours": [7.0, 6.0, 7.5],
    })
    preds_df = predictor.predict_batch(test_df)
    assert "predicted_final_marks" in preds_df.columns
    assert "performance_category" in preds_df.columns
    assert len(preds_df) == 3
    # Higher student should score higher than lower student
    assert preds_df["predicted_final_marks"].iloc[0] > preds_df["predicted_final_marks"].iloc[1]


def test_what_if_simulation(predictor):
    sim_df = predictor.simulate_what_if(
        attendance=80.0,
        previous_marks=70.0,
        study_hours=3.0,
    )
    assert len(sim_df) == 5
    assert "Projected Score" in sim_df.columns


def test_explain_prediction_drivers(predictor):
    explanation = predictor.explain_prediction_drivers(
        attendance=85.0,
        previous_marks=75.0,
        study_hours=4.5,
        predicted_score=82.0,
    )
    assert "cohort_benchmarks" in explanation
    assert "drivers" in explanation
    assert len(explanation["drivers"]) == 3
    assert "summary" in explanation
    assert "82.0/100" in explanation["summary"]


def test_recommendations_no_guarantees(predictor):
    recs = predictor.generate_study_recommendations(
        attendance=65.0,
        previous_marks=50.0,
        study_hours=2.0,
        predicted_score=52.0,
    )
    assert len(recs) >= 1
    # Check that text does not contain deceptive guarantee claims
    for r in recs:
        assert "statistically proven to raise" not in r["text"].lower()
        assert "guarantee" not in r["text"].lower()

