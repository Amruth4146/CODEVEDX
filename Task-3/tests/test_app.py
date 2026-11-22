"""Unit and Integration Tests for Streamlit UI (Task 3: AI-Based Fake News Detection).

Tests cover:
1. Streamlit application startup and page elements.
2. Example loading (Real and Fake preset examples).
3. Clear / Reset input functionality.
4. Input validation handling (empty, whitespace-only, short text).
5. Prediction execution and result display (REAL and FAKE badges, metrics, probability bar).
6. Educational disclaimer presence.
7. Absence of model retraining during app execution.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest
from streamlit.testing.v1 import AppTest

# Ensure Task-3 root is on sys.path
TASK3_ROOT = Path(__file__).resolve().parent.parent
if str(TASK3_ROOT) not in sys.path:
    sys.path.insert(0, str(TASK3_ROOT))

from app import (
    EDUCATIONAL_DISCLAIMER_TEXT,
    EXAMPLE_FAKE,
    EXAMPLE_REAL,
    load_cached_predictor,
)
from src.predictor import FakeNewsPredictor


APP_PATH = str(TASK3_ROOT / "app.py")


class TestStreamlitAppStartup:
    """Verify application startup, default state, and required static elements."""

    def test_app_loads_without_unhandled_exceptions(self) -> None:
        """App initializes cleanly without raising uncaught exceptions."""
        at = AppTest.from_file(APP_PATH).run()
        assert not at.exception, f"App raised unexpected exception: {at.exception}"

    def test_app_title_and_disclaimer_rendered(self) -> None:
        """App renders the main title and mandatory educational disclaimer."""
        at = AppTest.from_file(APP_PATH).run()

        # Check title
        assert any("AI-Based Fake News Detection" in str(t.value) for t in at.title)

        # Check educational disclaimer in warnings or text
        warning_texts = [w.value for w in at.warning]
        assert any(EDUCATIONAL_DISCLAIMER_TEXT in wt for wt in warning_texts)

    def test_sidebar_contains_model_specifications(self) -> None:
        """Sidebar displays model architecture details and validation metrics."""
        at = AppTest.from_file(APP_PATH).run()
        sidebar_text = " ".join(str(m.value) for m in at.sidebar.markdown)

        assert "Logistic Regression" in sidebar_text
        assert "TF-IDF" in sidebar_text
        assert "98.95%" in sidebar_text
        assert "5,000" in sidebar_text

    def test_cached_predictor_returns_instance(self) -> None:
        """load_cached_predictor returns a valid FakeNewsPredictor with frozen model."""
        predictor = load_cached_predictor()
        assert isinstance(predictor, FakeNewsPredictor)
        assert predictor.model_name == "Logistic Regression"
        assert predictor.feature_count == 5000


class TestStreamlitExamplesAndControls:
    """Verify example buttons, text area syncing, and reset button behavior."""

    def test_load_real_news_example_button(self) -> None:
        """Clicking 'Load Real News Example' populates the text area with EXAMPLE_REAL."""
        at = AppTest.from_file(APP_PATH).run()

        # Locate the button for Real News example
        btn_real = [b for b in at.button if "Real News Example" in str(b.label)]
        assert len(btn_real) == 1
        btn_real[0].click().run()

        assert not at.exception
        # Confirm text area contains the example text
        assert at.text_area[0].value == EXAMPLE_REAL

    def test_load_fake_news_example_button(self) -> None:
        """Clicking 'Load Fake News Example' populates the text area with EXAMPLE_FAKE."""
        at = AppTest.from_file(APP_PATH).run()

        btn_fake = [b for b in at.button if "Fake News Example" in str(b.label)]
        assert len(btn_fake) == 1
        btn_fake[0].click().run()

        assert not at.exception
        assert at.text_area[0].value == EXAMPLE_FAKE

    def test_clear_reset_button_clears_text(self) -> None:
        """Clicking Clear / Reset empties the text area."""
        at = AppTest.from_file(APP_PATH).run()

        # First load an example
        btn_real = [b for b in at.button if "Real News Example" in str(b.label)][0]
        btn_real.click().run()
        assert len(at.text_area[0].value) > 0

        # Now click Clear
        btn_clear = [b for b in at.button if "Clear / Reset" in str(b.label)][0]
        btn_clear.click().run()
        assert not at.exception
        assert at.text_area[0].value == ""


class TestStreamlitValidationAndPrediction:
    """Verify input validation errors and prediction rendering."""

    def test_empty_input_shows_validation_error(self) -> None:
        """Clicking classify with empty text produces a validation error message."""
        at = AppTest.from_file(APP_PATH).run()

        # Click classify without text
        btn_classify = [b for b in at.button if "Classify News" in str(b.label)][0]
        btn_classify.click().run()

        assert not at.exception
        errors = [e.value for e in at.error]
        assert any("Input Validation Error" in err for err in errors)
        assert any("cannot be empty" in err for err in errors)

    def test_short_input_shows_validation_error(self) -> None:
        """Clicking classify with too few words produces an informative validation error."""
        at = AppTest.from_file(APP_PATH).run()

        at.text_area[0].input("Breaking news").run()
        btn_classify = [b for b in at.button if "Classify News" in str(b.label)][0]
        btn_classify.click().run()

        assert not at.exception
        errors = [e.value for e in at.error]
        assert any("Input Validation Error" in err for err in errors)
        assert any("too short" in err for err in errors)

    def test_classify_real_news_displays_real_badge_and_metrics(self) -> None:
        """Evaluating real news displays REAL badge, metrics, and probability distribution."""
        at = AppTest.from_file(APP_PATH).run()

        # Load real news
        btn_real = [b for b in at.button if "Real News Example" in str(b.label)][0]
        btn_real.click().run()

        # Click Classify
        btn_classify = [b for b in at.button if "Classify News" in str(b.label)][0]
        btn_classify.click().run()

        assert not at.exception
        # Check success banner for REAL NEWS
        success_texts = [s.value for s in at.success]
        assert any("REAL NEWS" in stext for stext in success_texts)

        # Check metrics
        metric_labels = [m.label for m in at.metric]
        assert "Prediction" in metric_labels
        assert "Real Probability" in metric_labels
        assert "Fake Probability" in metric_labels

        pred_metric = [m for m in at.metric if m.label == "Prediction"][0]
        assert pred_metric.value == "REAL"

    def test_classify_fake_news_displays_fake_badge_and_metrics(self) -> None:
        """Evaluating fake news displays FAKE badge, metrics, and probability distribution."""
        at = AppTest.from_file(APP_PATH).run()

        # Load fake news
        btn_fake = [b for b in at.button if "Fake News Example" in str(b.label)][0]
        btn_fake.click().run()

        # Click Classify
        btn_classify = [b for b in at.button if "Classify News" in str(b.label)][0]
        btn_classify.click().run()

        assert not at.exception
        # Check error/alert banner for FAKE NEWS
        error_texts = [e.value for e in at.error]
        assert any("FAKE NEWS" in etext for etext in error_texts)

        pred_metric = [m for m in at.metric if m.label == "Prediction"][0]
        assert pred_metric.value == "FAKE"


class TestStreamlitSafetyAndIsolation:
    """Verify that UI execution does not retrain models or touch test data."""

    def test_app_source_does_not_call_fit_or_touch_test_csv(self) -> None:
        """Verify static source code of app.py contains no .fit calls or references to test.csv."""
        app_source = Path(APP_PATH).read_text(encoding="utf-8")
        assert ".fit(" not in app_source
        assert ".fit_transform(" not in app_source
        assert "test.csv" not in app_source
        assert "train_and_compare_models" not in app_source
