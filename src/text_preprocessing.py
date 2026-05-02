"""Text cleaning helpers for the journal finder project."""

from __future__ import annotations

import html
import re


_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
_NON_TEXT_RE = re.compile(r"[^a-zA-Z0-9\s+\-#/]")


def clean_text(value: object) -> str:
    """Normalize database text fields into plain searchable text."""
    if value is None:
        return ""

    text = html.unescape(str(value))
    text = _TAG_RE.sub(" ", text)
    text = text.replace("&nbsp;", " ")
    text = _NON_TEXT_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip().lower()


def join_terms(values: object) -> str:
    """Clean comma-separated or database-aggregated term strings."""
    return clean_text(values).replace(",", " ")
