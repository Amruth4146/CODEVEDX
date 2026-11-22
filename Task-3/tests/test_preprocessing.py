"""Unit tests for preprocessing.py in Task-3."""

import pytest
from src.preprocessing import (
    clean_html,
    clean_text,
    expand_contractions,
    normalize_whitespace,
    remove_punctuation,
    remove_reuters_dateline,
    remove_urls,
    to_lowercase,
)


class TestReutersDatelineRemoval:
    """Tests for safe Reuters dateline removal."""

    def test_standard_city_dateline_removed(self):
        text = "WASHINGTON (Reuters) - The Senate approved the federal budget resolution on Thursday."
        expected = "The Senate approved the federal budget resolution on Thursday."
        assert remove_reuters_dateline(text) == expected

    def test_dual_city_dateline_removed(self):
        text = "LONDON/BRUSSELS (Reuters) — European diplomats convened for an emergency session."
        expected = "European diplomats convened for an emergency session."
        assert remove_reuters_dateline(text) == expected

    def test_dateline_with_colon_removed(self):
        text = "BEIJING (Reuters): China reported new export statistics today."
        expected = "China reported new export statistics today."
        assert remove_reuters_dateline(text) == expected

    def test_standalone_reuters_dateline_removed(self):
        text = "(Reuters) - Breaking news from the international wire service."
        expected = "Breaking news from the international wire service."
        assert remove_reuters_dateline(text) == expected

    def test_dateline_with_leading_whitespace(self):
        text = "   \nWASHINGTON (Reuters) - Lawmakers debated the bill."
        expected = "Lawmakers debated the bill."
        assert remove_reuters_dateline(text) == expected

    def test_reuters_in_middle_of_sentence_preserved(self):
        """Crucial test: Reuters mentioned inside an article must NOT be removed."""
        text = "President Trump gave an exclusive interview to Reuters on Wednesday afternoon."
        assert remove_reuters_dateline(text) == text

    def test_reuters_as_subject_preserved(self):
        text = "A senior Reuters editor published an analysis of global monetary policy."
        assert remove_reuters_dateline(text) == text

    def test_article_without_dateline_unchanged(self):
        text = "Breaking News: A significant weather event was detected off the Pacific coast."
        assert remove_reuters_dateline(text) == text

    def test_non_string_dateline_input(self):
        assert remove_reuters_dateline(None) == ""
        assert remove_reuters_dateline(12345) == ""


class TestHTMLCleaning:
    """Tests for HTML stripping and entity decoding."""

    def test_html_tags_removed(self):
        html_text = "<p>This is a <b>bold</b> article with <a href='https://example.com'>links</a>.</p>"
        result = clean_html(html_text)
        assert "<p>" not in result
        assert "<b>" not in result
        assert "bold" in result

    def test_html_entities_unescaped(self):
        text = "Smith &amp; Wesson reported &quot;record profits&quot; &lt;today&gt;."
        result = clean_html(text)
        assert "&amp;" not in result
        assert "&quot;" not in result
        assert "&" in result
        assert '"record profits"' in result

    def test_non_string_html_input(self):
        assert clean_html(None) == ""


class TestURLRemoval:
    """Tests for web URL removal."""

    def test_http_and_https_removed(self):
        text = "Read the report at https://example.com/news or http://test.org/data today."
        result = remove_urls(text)
        assert "https://" not in result
        assert "http://" not in result
        assert "Read the report at" in result

    def test_www_urls_removed(self):
        text = "Check www.breakingnews.com for live commentary."
        result = remove_urls(text)
        assert "www.breakingnews.com" not in result

    def test_non_string_url_input(self):
        assert remove_urls(None) == ""


class TestContractionExpansionAndNegations:
    """Tests for contraction expansion and preserving negations."""

    def test_standard_negation_contractions(self):
        text = "They didn't believe it wasn't true and couldn't say it wouldn't happen."
        result = expand_contractions(text)
        assert "did not" in result
        assert "was not" in result
        assert "could not" in result
        assert "would not" in result

    def test_special_negations(self):
        text = "I can't accept this and won't agree, but they cannot force us."
        result = expand_contractions(text)
        assert "can not" in result
        assert "will not" in result

    def test_negations_are_never_removed(self):
        """Negation words must stay intact."""
        text = "There is no evidence, not a single trace, and never any proof."
        result = clean_text(text)
        assert "no" in result.split()
        assert "not" in result.split()
        assert "never" in result.split()

    def test_non_negation_contractions(self):
        text = "It's obvious that they're ready and we've arrived."
        result = expand_contractions(text)
        assert "it is" in result.lower()
        assert "they are" in result.lower()
        assert "we have" in result.lower()


class TestWhitespaceAndCasing:
    """Tests for whitespace normalization and lowercasing."""

    def test_whitespace_normalization(self):
        text = "  This   has \t\t many \n\n newlines  and   spaces.  "
        assert normalize_whitespace(text) == "This has many newlines and spaces."

    def test_to_lowercase(self):
        assert to_lowercase("BREAKING NEWS: US Politics") == "breaking news: us politics"


class TestPunctuationRemoval:
    """Tests for optional punctuation removal."""

    def test_punctuation_stripped(self):
        text = "Hello, world! Donald Trump's policy: 100% success?"
        result = remove_punctuation(text)
        assert "," not in result
        assert "!" not in result
        assert "?" not in result
        assert "%" not in result
        assert "Hello" in result
        assert "world" in result

    def test_punctuation_preserves_words(self):
        text = "did not, will not, never!"
        result = remove_punctuation(text)
        words = result.split()
        assert "did" in words
        assert "not" in words
        assert "will" in words
        assert "never" in words


class TestEndToEndCleanText:
    """Tests for unified clean_text pipeline."""

    def test_complete_cleaning_pipeline(self):
        raw = (
            "WASHINGTON (Reuters) - Officials didn't verify the claim! "
            "Visit <a href='https://news.org'>https://news.org</a> for updates &amp; alerts."
        )
        cleaned = clean_text(raw, strip_punctuation=False)
        assert "washington (reuters) -" not in cleaned
        assert "did not" in cleaned
        assert "https://" not in cleaned
        assert "<a" not in cleaned
        assert "&" in cleaned
        assert "officials" in cleaned

    def test_empty_and_whitespace_input(self):
        assert clean_text("") == ""
        assert clean_text("   \n\t  ") == ""
        assert clean_text(None) == ""

    def test_unexpected_input_types(self):
        assert clean_text(12345) == "12345"
        assert clean_text(99.5) == "99.5"
        assert clean_text(True) == "true"
