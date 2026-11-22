"""Streamlit Web Application for AI-Based Fake News Detection.

CODEVEDX AI/ML Internship — Task 3.
Provides an interactive, accessible, and professional user interface to classify
news articles and headlines using a pre-trained TF-IDF + Logistic Regression pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys
from typing import Any

# Ensure Task-3 root is present in sys.path
TASK3_ROOT = Path(__file__).resolve().parent
if str(TASK3_ROOT) not in sys.path:
    sys.path.insert(0, str(TASK3_ROOT))

import streamlit as st

from src.predictor import (
    FakeNewsPredictor,
    PredictionResult,
    get_default_predictor,
    validate_news_input,
)

logger = logging.getLogger(__name__)

# Preset news examples for quick user testing
EXAMPLE_REAL = (
    "The United States Senate voted on Thursday to approve a bipartisan budget compromise "
    "following days of intensive floor negotiations between leadership. The legislation allocates "
    "essential federal funding for transportation infrastructure and domestic development projects "
    "over the upcoming fiscal calendar."
)

EXAMPLE_FAKE = (
    "SHOCKING BOMBSHELL: Anonymous whistleblower inside secret government committee reveals "
    "massive covert plot to alter nationwide voting tallies using hidden servers! Corrupt mainstream "
    "media executives are actively colluding to bury this unbelievable revelation from the public!"
)

# Mandatory educational disclaimer
EDUCATIONAL_DISCLAIMER_TEXT = (
    "This tool classifies news based on patterns learned from its training dataset. "
    "It does not independently verify facts or determine whether a news article is "
    "objectively true or false. Predictions may be inaccurate."
)


@st.cache_resource(show_spinner="Loading pre-trained Fake News Detection model...")
def load_cached_predictor() -> FakeNewsPredictor:
    """Load and cache the pre-trained champion inference model into memory.

    Utilizes Streamlit's cache_resource to ensure the .joblib artifact is loaded
    exactly once into RAM across all user sessions. Strictly avoids any retraining
    or repeated disk reads.
    """
    model_path = TASK3_ROOT / "models" / "fake_news_model.joblib"
    return get_default_predictor(model_path=model_path)


def initialize_session_state() -> None:
    """Initialize session state variables for input text and prediction results."""
    if "news_input" not in st.session_state:
        st.session_state.news_input = ""
    if "last_prediction" not in st.session_state:
        st.session_state.last_prediction = None
    if "input_key" not in st.session_state:
        st.session_state.input_key = 0


def set_example_text(text: str) -> None:
    """Set the text area to a preset example and clear previous prediction."""
    st.session_state.news_input = text
    st.session_state.last_prediction = None
    st.session_state.input_key += 1


def clear_inputs() -> None:
    """Clear user input and cached prediction result."""
    st.session_state.news_input = ""
    st.session_state.last_prediction = None
    st.session_state.input_key += 1


def render_sidebar(predictor: FakeNewsPredictor) -> None:
    """Render informative sidebar with model specs, pipeline info, and guidelines."""
    with st.sidebar:
        st.header("About This Project")
        st.caption("CODEVEDX AI/ML Internship — Task 3")
        st.write(
            "An AI-based natural language processing tool designed to evaluate news headlines "
            "and articles for stylistic markers and vocabulary patterns characteristic of fake news."
        )

        st.divider()

        st.subheader("Model Specifications")
        st.markdown(
            f"""
            - **Champion Classifier:** {predictor.model_name}
            - **Feature Extractor:** TF-IDF (Unigrams & Bigrams)
            - **Vocabulary Size:** {predictor.feature_count:,} features
            - **Sublinear TF:** Enabled (Logarithmic scaling)
            - **Validation Accuracy:** 98.95%
            - **Macro F1 Score:** 0.9895
            """
        )

        st.divider()

        st.subheader("Pipeline Details")
        st.markdown(
            """
            1. **Sanitization:** Wire datelines stripped, HTML unescaped, URLs removed.
            2. **Text Cleaning:** Contractions expanded (preserving negations like *not*, *no*).
            3. **Vectorization:** Fixed vocabulary projection onto 5,000 TF-IDF features.
            4. **Classification:** Sigmoid posterior model probabilities ($P(\\text{Real})$, $P(\\text{Fake})$).
            """
        )

        st.divider()

        st.subheader("Training Corpus")
        st.caption(
            "Trained on the **ISOT Fake News Dataset** (2016–2017 US political and world news). "
            "Articles outside this domain or era may reflect distribution shift."
        )


def render_prediction_card(result: PredictionResult) -> None:
    """Display visually distinct, accessible results for Real vs Fake classification."""
    is_real = result.label == 1
    pct_confidence = result.confidence * 100.0
    pct_real = result.real_probability * 100.0
    pct_fake = result.fake_probability * 100.0

    st.divider()
    st.subheader("Classification Results")

    # Visually distinct styling card
    if is_real:
        st.success(
            f"### Result: **REAL NEWS** ({pct_confidence:.1f}% Confidence)\n\n"
            f"The analyzed text strongly aligns with objective journalism patterns and vocabulary "
            f"found in legitimate reporting."
        )
    else:
        st.error(
            f"### Result: **FAKE NEWS** ({pct_confidence:.1f}% Confidence)\n\n"
            f"The analyzed text exhibits linguistic markers, sensationalist phrasing, or stylometric "
            f"patterns characteristic of misleading or fabricated news stories."
        )

    # Key metrics row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="Prediction",
            value="REAL" if is_real else "FAKE",
            delta=f"{pct_confidence:.1f}% confidence",
        )
    with col2:
        st.metric(
            label="Real Probability",
            value=f"{pct_real:.1f}%",
        )
    with col3:
        st.metric(
            label="Fake Probability",
            value=f"{pct_fake:.1f}%",
        )

    # Visual probability distribution
    st.write("**Probability Distribution:**")
    st.progress(float(result.real_probability), text=f"Real: {pct_real:.1f}% | Fake: {pct_fake:.1f}%")

    # Cleaned text preview
    with st.expander("View Cleaned Text Representation (Processed Input)"):
        st.write(
            "Below is the normalized text passed to the TF-IDF feature extractor "
            "(with datelines removed, lowercased, and contractions expanded):"
        )
        st.code(result.cleaned_preview or "(Cleaned text preview unavailable)", language="text")

    # Mandatory educational disclaimer
    st.info(f"**Educational Notice:** {result.disclaimer}")


def main() -> None:
    """Main application layout and execution flow."""
    st.set_page_config(
        page_title="AI-Based Fake News Detection",
        page_icon="📰",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    initialize_session_state()

    # Title and high-level description
    st.title("📰 AI-Based Fake News Detection")
    st.markdown(
        "Evaluate news headlines and articles for linguistic patterns and stylometric markers "
        "associated with credible vs. misleading reporting using machine learning."
    )

    # Load pre-trained model safely
    try:
        predictor = load_cached_predictor()
    except Exception as err:
        st.error(
            f"**Error initializing model:** Unable to load champion model artifact. Details: {err}\n\n"
            "Please ensure `Task-3/models/fake_news_model.joblib` exists."
        )
        return

    # Render informative sidebar
    render_sidebar(predictor)

    # Educational Disclaimer Banner
    st.warning(
        f"**Important Disclaimer:** {EDUCATIONAL_DISCLAIMER_TEXT}",
        icon="⚠️",
    )

    # Quick Example Buttons
    st.write("##### Quick Test Examples")
    col_ex1, col_ex2, col_clear = st.columns([1, 1, 1])

    with col_ex1:
        if st.button("📰 Load Real News Example", use_container_width=True):
            set_example_text(EXAMPLE_REAL)
            st.rerun()

    with col_ex2:
        if st.button("🚨 Load Fake News Example", use_container_width=True):
            set_example_text(EXAMPLE_FAKE)
            st.rerun()

    with col_clear:
        if st.button("🔄 Clear / Reset", use_container_width=True):
            clear_inputs()
            st.rerun()

    # User Input Area
    user_input = st.text_area(
        label="Enter News Headline and/or Article Text:",
        value=st.session_state.news_input,
        height=200,
        placeholder="Paste full news article body or headline here (minimum 3 words, 10 characters)...",
        key=f"text_area_{st.session_state.input_key}",
    )

    # Sync text area edits back to session state
    st.session_state.news_input = user_input

    # Prediction Action Button
    col_btn, _ = st.columns([1, 3])
    with col_btn:
        predict_clicked = st.button("🔍 Classify News", type="primary", use_container_width=True)

    if predict_clicked:
        # Step 1: Input Validation
        try:
            valid_text = validate_news_input(user_input)
        except (TypeError, ValueError) as val_err:
            st.error(f"**Input Validation Error:** {val_err}")
            st.session_state.last_prediction = None
            return

        # Step 2: Prediction with loading spinner
        with st.spinner("Analyzing text patterns and vocabulary..."):
            try:
                result = predictor.predict(valid_text)
                st.session_state.last_prediction = result
            except Exception as exc:
                st.error(f"**Prediction Error:** An error occurred during inference: {exc}")
                logger.exception("Inference error in Streamlit app")
                return

    # Render result if present
    if st.session_state.last_prediction is not None:
        render_prediction_card(st.session_state.last_prediction)


if __name__ == "__main__":
    main()
