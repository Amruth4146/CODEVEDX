"""Text Preprocessing Module for Task 3: AI-Based Fake News Detection.

Provides modular, reusable, and safe text preprocessing functions tailored for
misinformation detection on the ISOT Fake News Dataset and general text inputs.

Key design principles:
1. Reusability: Each cleaning step is encapsulated in an independent, testable function.
2. Preservation of semantic signals: Negation words ('not', 'no', 'never') are never removed.
3. Safe dateline handling: Leading wire datelines (e.g. 'WASHINGTON (Reuters) -') are removed
   only from article headers without corrupting internal text mentioning Reuters.
4. Stylistic flexibility: Supports configurable cleaning modes (minimal, standard, aggressive)
   to evaluate the predictive value of punctuation and capitalization.
5. Robustness: Gracefully handles unexpected input types (None, NaN, numbers, booleans).
"""

from __future__ import annotations

import html
import re
from typing import Any

# Specific exceptions where standard contraction suffix patterns do not apply directly
CONTRACTION_OVERRIDES: dict[str, str] = {
    r"\bcan['’]t\b": "can not",
    r"\bcannot\b": "can not",
    r"\bwon['’]t\b": "will not",
    r"\bshan['’]t\b": "shall not",
    r"\blet['’]s\b": "let us",
}

# General English contraction suffixes
CONTRACTION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"n['’]t\b", flags=re.IGNORECASE), " not"),
    (re.compile(r"['’]re\b", flags=re.IGNORECASE), " are"),
    (re.compile(r"['’]s\b", flags=re.IGNORECASE), " is"),
    (re.compile(r"['’]d\b", flags=re.IGNORECASE), " would"),
    (re.compile(r"['’]ll\b", flags=re.IGNORECASE), " will"),
    (re.compile(r"['’]ve\b", flags=re.IGNORECASE), " have"),
    (re.compile(r"['’]m\b", flags=re.IGNORECASE), " am"),
]

# Regex for Reuters initial wire datelines
# Matches patterns such as:
# "WASHINGTON (Reuters) - "
# "PARIS/BRUSSELS (Reuters) — "
# "(Reuters) - "
# Anchored at the start of text (\A or ^) to strictly prevent matching inside sentences.
REUTERS_DATELINE_REGEX = re.compile(
    r"^\s*(?:[A-Z][A-Za-z0-9\s,\/\.\-]{0,60}?\s*)?\((?:Reuters|REUTERS)\)\s*[-—–:]\s*",
    flags=re.MULTILINE,
)

# General URL pattern
URL_REGEX = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)

# HTML tags pattern
HTML_TAG_REGEX = re.compile(r"<[^>]+>")

# Non-alphanumeric punctuation pattern (preserves word characters and spaces)
PUNCTUATION_REGEX = re.compile(r"[^\w\s]")


def remove_reuters_dateline(text: str) -> str:
    """Remove wire dateline headers from the beginning of an article.

    Safely removes lead prefixes such as 'WASHINGTON (Reuters) - ' or '(Reuters) - '.
    Anchored strictly at the start of the article so that mentions of 'Reuters' in the
    body or middle of sentences are completely preserved.

    Args:
        text: Input text string.

    Returns:
        Text with leading Reuters dateline removed, or original text if no dateline exists.
    """
    if not isinstance(text, str):
        return ""
    return REUTERS_DATELINE_REGEX.sub("", text, count=1)


def clean_html(text: str) -> str:
    """Unescape HTML entities and remove HTML tags.

    Converts entities like '&amp;' to '&' and '&quot;' to '\"', then strips
    all HTML tags like '<p>', '<div>', etc.

    Args:
        text: Input text string.

    Returns:
        Text stripped of HTML markup and with decoded entities.
    """
    if not isinstance(text, str):
        return ""
    unescaped = html.unescape(text)
    return HTML_TAG_REGEX.sub(" ", unescaped)


def remove_urls(text: str) -> str:
    """Remove web URLs (http, https, www) from text.

    Args:
        text: Input text string.

    Returns:
        Text with URLs replaced by spaces.
    """
    if not isinstance(text, str):
        return ""
    return URL_REGEX.sub(" ", text)


def expand_contractions(text: str) -> str:
    """Expand English contractions to preserve and highlight negation words.

    For example, 'didn't' -> 'did not', 'won't' -> 'will not', and 'can't' -> 'can not'.
    This guarantees that critical semantic negation words ('not') remain distinct
    and are not corrupted or dropped during subsequent tokenization.

    Args:
        text: Input text string.

    Returns:
        Text with contractions expanded.
    """
    if not isinstance(text, str):
        return ""

    result = text
    # Apply specific overrides first
    for pattern, replacement in CONTRACTION_OVERRIDES.items():
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    # Apply general suffix patterns
    for pattern, replacement in CONTRACTION_PATTERNS:
        result = pattern.sub(replacement, result)

    return result


def to_lowercase(text: str) -> str:
    """Convert text to lowercase.

    Args:
        text: Input text string.

    Returns:
        Lowercased text.
    """
    if not isinstance(text, str):
        return ""
    return text.lower()


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace by collapsing multiple spaces, tabs, and newlines.

    Args:
        text: Input text string.

    Returns:
        Trimmed text with single spaces between words.
    """
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()


def remove_punctuation(text: str) -> str:
    """Remove punctuation characters while preserving alphanumeric characters.

    Replaces non-alphanumeric, non-whitespace characters with spaces.
    Negation tokens (e.g., 'not', 'no', 'never') are composed of letters and are
    fully preserved.

    Args:
        text: Input text string.

    Returns:
        Punctuation-free text with whitespace preserved.
    """
    if not isinstance(text, str):
        return ""
    return PUNCTUATION_REGEX.sub(" ", text)


def clean_text_by_mode(text: Any, mode: str = "standard") -> str:
    """Clean text according to one of three configurable cleaning modes.

    Modes:
    - 'minimal': Strips dateline, HTML, and URLs; normalizes whitespace.
      Preserves original casing and punctuation (ideal for stylistic clickbait analysis).
    - 'standard' (Default Baseline): Minimal + contraction expansion + lowercase.
      Preserves negations while standardizing vocabulary; leaves punctuation intact.
    - 'aggressive': Standard + complete punctuation removal (replaces punctuation with spaces).

    Args:
        text: Input text or object.
        mode: Cleaning mode ('minimal', 'standard', or 'aggressive').

    Returns:
        Cleaned text string.

    Raises:
        ValueError: If an unknown cleaning mode is specified.
    """
    mode_lower = mode.lower().strip()
    if mode_lower == "minimal":
        return clean_text(
            text,
            remove_dateline=True,
            remove_html_tags=True,
            remove_urls_flag=True,
            expand_contr=False,
            lowercase=False,
            strip_punctuation=False,
        )
    elif mode_lower == "standard":
        return clean_text(
            text,
            remove_dateline=True,
            remove_html_tags=True,
            remove_urls_flag=True,
            expand_contr=True,
            lowercase=True,
            strip_punctuation=False,
        )
    elif mode_lower == "aggressive":
        return clean_text(
            text,
            remove_dateline=True,
            remove_html_tags=True,
            remove_urls_flag=True,
            expand_contr=True,
            lowercase=True,
            strip_punctuation=True,
        )
    else:
        raise ValueError(
            f"Unknown cleaning mode: '{mode}'. Supported modes are: 'minimal', 'standard', 'aggressive'."
        )


def clean_text(
    text: Any,
    remove_dateline: bool = True,
    remove_html_tags: bool = True,
    remove_urls_flag: bool = True,
    expand_contr: bool = True,
    lowercase: bool = True,
    strip_punctuation: bool = False,
) -> str:
    """Execute end-to-end modular text cleaning on an article or sentence.

    Applies the individual cleaning steps in a controlled sequence:
    1. Type validation (gracefully converts or ignores non-strings).
    2. Dateline header stripping (if remove_dateline=True).
    3. HTML tag removal and entity decoding (if remove_html_tags=True).
    4. URL removal (if remove_urls_flag=True).
    5. Contraction expansion to protect negations (if expand_contr=True).
    6. Lowercase conversion (if lowercase=True).
    7. Punctuation removal (if strip_punctuation=True; default False to preserve punctuation cues).
    8. Whitespace normalization (collapses multiple spaces, strips margins).

    Args:
        text: Input text, string, or arbitrary object.
        remove_dateline: Whether to strip leading Reuters datelines.
        remove_html_tags: Whether to remove HTML tags and unescape entities.
        remove_urls_flag: Whether to strip web URLs.
        expand_contr: Whether to expand contractions ('didn't' -> 'did not').
        lowercase: Whether to convert text to lowercase.
        strip_punctuation: Whether to replace punctuation with spaces.

    Returns:
        Cleaned and normalized text string.
    """
    if text is None:
        return ""

    if not isinstance(text, str):
        text = str(text)

    if not text.strip():
        return ""

    if remove_dateline:
        text = remove_reuters_dateline(text)

    if remove_html_tags:
        text = clean_html(text)

    if remove_urls_flag:
        text = remove_urls(text)

    if expand_contr:
        text = expand_contractions(text)

    if lowercase:
        text = to_lowercase(text)

    if strip_punctuation:
        text = remove_punctuation(text)

    text = normalize_whitespace(text)
    return text
