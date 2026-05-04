"""Text normalization and feature assembly for the multi-channel model."""

from __future__ import annotations

import re
from html import unescape

import pandas as pd


def _text_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series([""] * len(frame), index=frame.index)
    return frame[column].fillna("")


def clean_text(text: object) -> str:
    """Normalize HTML-heavy publication text into model input text."""
    if pd.isna(text) or text is None:
        return ""
    cleaned = unescape(str(text))
    cleaned = re.sub(r"<.*?>", " ", cleaned)
    cleaned = cleaned.lower()
    cleaned = re.sub(r"[\r\n\t]+", " ", cleaned)
    cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def build_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the title, abstract, keyword, and subject channels."""
    output = pd.DataFrame(index=frame.index)
    output["title_channel"] = _text_series(frame, "title_clean").map(clean_text)
    output["abstract_channel"] = _text_series(frame, "abstract_clean").map(clean_text)
    keywords = (
        _text_series(frame, "keywords_clean").map(clean_text)
        + " "
        + _text_series(frame, "keyword_plus_clean").map(clean_text)
    ).str.strip()
    output["keywords_channel"] = keywords
    output["subjects_channel"] = _text_series(frame, "subjects_clean").map(clean_text)
    return output


def build_single_prediction_frame(
    abstract: str,
    title: str = "",
    keywords: str = "",
    subjects: str = "",
) -> pd.DataFrame:
    """Build one prediction row matching the trained ColumnTransformer schema."""
    return pd.DataFrame(
        [
            {
                "title_channel": clean_text(title),
                "abstract_channel": clean_text(abstract),
                "keywords_channel": clean_text(keywords),
                "subjects_channel": clean_text(subjects),
            }
        ]
    )
