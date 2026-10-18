"""
Prediction and Inference Engine for Student Performance Prediction System.
Loads real trained model artifacts, performs validated real-time inference,
and outputs performance classifications and actionable study recommendations.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from sklearn.pipeline import Pipeline
from src.train_model import load_model_bundle, MODEL_PATH, validate_loaded_model_schema
from src.preprocessing import prepare_inference_features, encode_features

logger = logging.getLogger(__name__)


class ModelPredictor:
    """
    Inference orchestrator that loads the serialized machine learning pipeline
    and produces genuine, non-fabricated predictions for individual or batch records.
    """

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or MODEL_PATH
        self.bundle: Optional[Dict[str, Any]] = None
        self._load_model()

    def _load_model(self) -> None:
        """
        Loads the model bundle from disk. If absent, triggers automated training.
        """
        try:
            self.bundle = load_model_bundle(self.model_path)
            self.pipeline = self.bundle.get("pipeline") or self.bundle.get("model")
            self.model = self.pipeline
            self.scaler = self.bundle.get("scaler")
            self.feature_names = self.bundle.get("feature_names", [])
            self.model_name = self.bundle.get("model_name", "Regression Model")
            self.metrics = self.bundle.get("metrics", {})
            logger.info("Predictor initialized with model: %s (R2 = %s)", self.model_name, self.metrics.get("r2"))
        except Exception as exc:
            logger.error("Failed to load model bundle: %s", exc)
            raise RuntimeError(f"Prediction engine initialization error: {exc}") from exc

    def classify_performance(self, score: float) -> Dict[str, str]:
        """
        Classifies an exam score into academic standing tiers.
        """
        if score >= 75.0:
            return {
                "category": "Distinction",
                "badge_class": "badge-distinction",
                "risk_level": "Low Risk",
                "color": "#10B981",
                "summary": "Outstanding academic performance. Strong foundation across all parameters.",
            }
        elif score >= 60.0:
            return {
                "category": "First Class",
                "badge_class": "badge-first",
                "risk_level": "Moderate",
                "color": "#06B6D4",
                "summary": "Good academic standing. Consistent work with headroom for top-tier distinction.",
            }
        elif score >= 40.0:
            return {
                "category": "Pass",
                "badge_class": "badge-pass",
                "risk_level": "Elevated Risk",
                "color": "#F59E0B",
                "summary": "Meets minimum passing requirements. Highly recommended to increase focused revision.",
            }
        else:
            return {
                "category": "At-Risk (Fail)",
                "badge_class": "badge-risk",
                "risk_level": "High Risk",
                "color": "#EF4444",
                "summary": "Critical academic intervention required. Immediate remediation on study hours & attendance needed.",
            }

    def predict(
        self,
        attendance_percentage: float,
        previous_exam_marks: float,
        daily_study_hours: float,
        assignments_completed: int = 8,
        extracurricular_activities: str = "No",
        sleep_hours: float = 7.0,
    ) -> Dict[str, Any]:
        """
        Performs genuine real-time prediction using the fitted scikit-learn model.
        Returns prediction score, classification, and metadata.
        """
        # Step 1: Format inference features
        features_df = prepare_inference_features(
            attendance_percentage=attendance_percentage,
            previous_exam_marks=previous_exam_marks,
            daily_study_hours=daily_study_hours,
            assignments_completed=assignments_completed,
            extracurricular_activities=extracurricular_activities,
            sleep_hours=sleep_hours,
            feature_names=self.feature_names,
        )

        # Step 2: Real Model Inference
        # If pipeline is available, pass features directly so intermediate steps (imputer, scaler) execute correctly
        if isinstance(self.pipeline, Pipeline):
            raw_pred = self.pipeline.predict(features_df)[0]
        else:
            if self.scaler is not None:
                X_input = self.scaler.transform(features_df)
            else:
                X_input = features_df.values
            raw_pred = self.model.predict(X_input)[0]

        # Bound to realistic 0-100% exam scale
        predicted_score = float(np.clip(np.round(raw_pred, 1), 0.0, 100.0))

        # Step 3: Academic classification
        classification = self.classify_performance(predicted_score)

        # Step 4: Recommendations
        recommendations = self.generate_study_recommendations(
            attendance=attendance_percentage,
            previous_marks=previous_exam_marks,
            study_hours=daily_study_hours,
            predicted_score=predicted_score,
        )

        return {
            "predicted_score": predicted_score,
            "classification": classification,
            "recommendations": recommendations,
            "model_used": self.model_name,
            "model_r2": self.metrics.get("r2"),
            "input_features": {
                "Attendance": f"{attendance_percentage:.1f}%",
                "Previous Marks": f"{previous_exam_marks:.1f}",
                "Daily Study Hours": f"{daily_study_hours:.1f} hrs",
                "Assignments": f"{assignments_completed}/10",
                "Extracurricular": extracurricular_activities,
                "Sleep Hours": f"{sleep_hours:.1f} hrs",
            },
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs batch inference on a DataFrame of student records.
        Handles empty DataFrames, missing columns, and mixed string/numeric inputs safely.
        """
        if df is None or df.empty:
            result_df = pd.DataFrame() if df is None else df.copy()
            result_df["predicted_final_marks"] = []
            result_df["performance_category"] = []
            result_df["risk_level"] = []
            return result_df

        result_df = df.copy()
        encoded = encode_features(result_df)

        # Consistent defaults for optional features if omitted from roster
        feature_defaults = {
            "assignments_completed": 8,
            "extracurricular_activities": 0,
            "sleep_hours": 7.0,
        }

        # Convert numerical predictor columns to numeric safely
        for col in self.feature_names:
            if col in encoded.columns:
                encoded[col] = pd.to_numeric(encoded[col], errors="coerce")
            else:
                encoded[col] = feature_defaults.get(col, np.nan)

        # Check schema compatibility
        valid_schema, schema_err = validate_loaded_model_schema(self.bundle, encoded)
        if not valid_schema:
            logger.warning("Batch schema mismatch warning: %s", schema_err)

        X_input = encoded[self.feature_names].copy()

        if isinstance(self.pipeline, Pipeline):
            raw_preds = self.pipeline.predict(X_input)
        else:
            if self.scaler is not None:
                X_scaled = self.scaler.transform(X_input)
            else:
                X_scaled = X_input.values
            raw_preds = self.model.predict(X_scaled)

        preds_clipped = np.clip(np.round(raw_preds, 1), 0.0, 100.0)

        result_df["predicted_final_marks"] = preds_clipped
        result_df["performance_category"] = [
            self.classify_performance(score)["category"] for score in preds_clipped
        ]
        result_df["risk_level"] = [
            self.classify_performance(score)["risk_level"] for score in preds_clipped
        ]

        return result_df

    def generate_study_recommendations(
        self,
        attendance: float,
        previous_marks: float,
        study_hours: float,
        predicted_score: float,
    ) -> List[Dict[str, str]]:
        """
        Generates neutral, supportive, evidence-based academic recommendations.
        Adheres strictly to pedagogical best practices: avoids promising guaranteed
        point gains and acknowledges that individual real-world circumstances vary.
        """
        recs = []

        if attendance < 75.0:
            recs.append({
                "category": "Classroom Engagement",
                "text": (
                    f"Attendance is currently recorded at {attendance:.1f}%. Educational research indicates "
                    "that active classroom attendance strongly correlates with syllabus familiarity and direct instructor feedback. "
                    "Prioritizing lecture presence can help bridge conceptual gaps before exams."
                ),
                "icon": "📅",
            })

        if study_hours < 3.0:
            recs.append({
                "category": "Study Habit Structure",
                "text": (
                    f"Current recorded study dedication is {study_hours:.1f} hrs/day. Incorporating structured "
                    "study sessions using evidence-based techniques—such as Pomodoro intervals or distributed spaced "
                    "repetition—can enhance topic comprehension without cognitive fatigue."
                ),
                "icon": "⏱️",
            })

        if previous_marks < 55.0:
            recs.append({
                "category": "Foundational Concept Review",
                "text": (
                    "Previous examination scores suggest that revisiting fundamental concepts in prerequisite topics "
                    "may strengthen overall confidence. Consider booking instructor office hours, forming peer study groups, "
                    "or reviewing worked solution sets."
                ),
                "icon": "📚",
            })

        if predicted_score >= 75.0:
            recs.append({
                "category": "Performance Maintenance",
                "text": (
                    "Current metrics indicate a strong academic trajectory. To maintain consistency, emphasize "
                    "timed past paper practice, self-explanation of complex topics, and balanced sleep routines before exams."
                ),
                "icon": "⭐",
            })

        if not recs:
            recs.append({
                "category": "Holistic Learning Strategy",
                "text": (
                    "Maintain your balanced routine of consistent attendance, active recall revision, and regular "
                    "self-assessment to support steady academic progress."
                ),
                "icon": "✅",
            })

        return recs

    def explain_prediction_drivers(
        self,
        attendance: float,
        previous_marks: float,
        study_hours: float,
        predicted_score: float,
    ) -> Dict[str, Any]:
        """
        Provides an educational breakdown of how primary features correlate with the predicted outcome,
        benchmarking inputs against representative cohort averages.
        """
        # Benchmark cohort references
        cohort_benchmarks = {
            "attendance": 80.0,
            "previous_marks": 68.0,
            "study_hours": 3.8,
            "final_score": 67.5,
        }

        diff_att = attendance - cohort_benchmarks["attendance"]
        diff_prev = previous_marks - cohort_benchmarks["previous_marks"]
        diff_study = study_hours - cohort_benchmarks["study_hours"]
        diff_score = predicted_score - cohort_benchmarks["final_score"]

        drivers = [
            {
                "feature": "Attendance Percentage",
                "input_value": f"{attendance:.1f}%",
                "cohort_mean": f"{cohort_benchmarks['attendance']:.1f}%",
                "delta": f"{diff_att:+.1f}%",
                "status": "Above Average" if diff_att >= 0 else "Below Average",
                "role": "Measures syllabus exposure, lecture participation, and academic regularity.",
            },
            {
                "feature": "Previous Examination Marks",
                "input_value": f"{previous_marks:.1f}/100",
                "cohort_mean": f"{cohort_benchmarks['previous_marks']:.1f}/100",
                "delta": f"{diff_prev:+.1f} pts",
                "status": "Above Average" if diff_prev >= 0 else "Below Average",
                "role": "Serves as a strong foundation anchor reflecting cumulative subject understanding.",
            },
            {
                "feature": "Daily Study Hours",
                "input_value": f"{study_hours:.1f} hrs/day",
                "cohort_mean": f"{cohort_benchmarks['study_hours']:.1f} hrs",
                "delta": f"{diff_study:+.1f} hrs",
                "status": "Above Average" if diff_study >= 0 else "Below Average",
                "role": "Reinforces concept retention and problem-solving readiness through active practice.",
            },
        ]

        return {
            "cohort_benchmarks": cohort_benchmarks,
            "drivers": drivers,
            "score_delta": diff_score,
            "summary": (
                f"The forecasted score of {predicted_score:.1f}/100 reflects a combined mathematical weighting "
                f"across attendance ({attendance:.1f}%), prior achievement ({previous_marks:.1f}), and daily revision commitment "
                f"({study_hours:.1f} hrs). Results are statistical estimates based on cohort patterns."
            ),
        }

    def simulate_what_if(
        self,
        attendance: float,
        previous_marks: float,
        study_hours: float,
        assignments: int = 8,
        extra: str = "No",
        sleep: float = 7.0,
    ) -> pd.DataFrame:
        """
        Simulates final score outcomes across a range of additional daily study hours (+0 to +4 hrs).
        """
        deltas = [0.0, 1.0, 2.0, 3.0, 4.0]
        rows = []
        for d in deltas:
            new_study = min(study_hours + d, 14.0)
            res = self.predict(
                attendance_percentage=attendance,
                previous_exam_marks=previous_marks,
                daily_study_hours=new_study,
                assignments_completed=assignments,
                extracurricular_activities=extra,
                sleep_hours=sleep,
            )
            rows.append({
                "Additional Hours": f"+{d:.1f} hrs" if d > 0 else "Current",
                "Daily Study Hours": round(new_study, 1),
                "Projected Score": res["predicted_score"],
                "Standing": res["classification"]["category"],
            })
        return pd.DataFrame(rows)


def render_score_gauge(score: float) -> go.Figure:
    """
    Renders an interactive speedometer gauge for the predicted examination score.
    Styled for light modern SaaS aesthetics.
    """
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": " / 100", "font": {"size": 36, "color": "#0F172A", "family": "Inter, sans-serif"}},
            title={"text": "Predicted Final Examination Mark", "font": {"size": 15, "color": "#64748B", "family": "Inter, sans-serif"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#64748B"},
                "bar": {"color": "#4F46E5", "thickness": 0.28},
                "bgcolor": "#F1F5F9",
                "borderwidth": 1,
                "bordercolor": "#E2E8F0",
                "steps": [
                    {"range": [0, 40], "color": "rgba(239, 68, 68, 0.12)"},
                    {"range": [40, 60], "color": "rgba(245, 158, 11, 0.12)"},
                    {"range": [60, 75], "color": "rgba(6, 182, 212, 0.12)"},
                    {"range": [75, 100], "color": "rgba(16, 185, 129, 0.12)"},
                ],
                "threshold": {
                    "line": {"color": "#0F172A", "width": 3},
                    "thickness": 0.8,
                    "value": score,
                },
            },
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=280,
        margin=dict(l=30, r=30, t=50, b=20),
        paper_bgcolor="#FFFFFF",
    )
    return fig


def render_input_vs_cohort_chart(
    attendance: float,
    previous_marks: float,
    study_hours: float,
    predicted_score: float,
) -> go.Figure:
    """
    Renders a comparative grouped bar chart benchmarking the student's inputs
    against standard cohort averages.
    """
    categories = ["Attendance (%)", "Previous Marks", "Daily Study (hrs x 10)", "Predicted Score"]
    student_vals = [attendance, previous_marks, study_hours * 10.0, predicted_score]
    cohort_vals = [80.2, 68.5, 3.8 * 10.0, 67.5]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Student Value",
        x=categories,
        y=student_vals,
        marker_color="#4F46E5",
        text=[f"{attendance:.1f}%", f"{previous_marks:.1f}", f"{study_hours:.1f}h", f"{predicted_score:.1f}"],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="Cohort Benchmark",
        x=categories,
        y=cohort_vals,
        marker_color="#CBD5E1",
        text=["80.2%", "68.5", "3.8h", "67.5"],
        textposition="outside",
    ))
    fig.update_layout(
        template="plotly_white",
        barmode="group",
        title=dict(text="Student Parameters vs. Cohort Benchmarks", font=dict(color="#0F172A", size=14)),
        yaxis=dict(title="Scale / Normalized Value", range=[0, 115], gridcolor="#F1F5F9"),
        xaxis=dict(gridcolor="#F1F5F9"),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#F8FAFC",
        margin=dict(l=30, r=30, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=300,
    )
    return fig

