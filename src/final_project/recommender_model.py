"""Saved classifier loading and Top-K journal recommendation."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .paths import PIPELINE_PATH, RECOMMENDER_META_PATH, require_file
from .text import build_single_prediction_frame


def _patch_sklearn_private_validation_helpers() -> None:
    """Keep old/mixed local sklearn installs from failing while unpickling."""
    try:
        import sklearn.utils.validation as validation
    except ImportError:
        return

    if not hasattr(validation, "_is_pandas_df"):
        validation._is_pandas_df = lambda value: isinstance(value, pd.DataFrame)

    if not hasattr(validation, "_is_polars_df"):
        validation._is_polars_df = lambda value: False


def load_pipeline(path: Path = PIPELINE_PATH):
    """Load the persisted multi-channel journal recommender pipeline."""
    _patch_sklearn_private_validation_helpers()
    return joblib.load(require_file(path, "Journal recommender pipeline"))


def load_metadata(path: Path = RECOMMENDER_META_PATH) -> dict | None:
    """Load saved training metrics when available."""
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8-sig"))


def recommend_journals(
    pipeline,
    abstract: str,
    title: str = "",
    keywords: str = "",
    subjects: str = "",
    top_k: int = 5,
) -> pd.DataFrame:
    """Return the highest-probability journals for one article description."""
    if not abstract.strip():
        raise ValueError("Please enter an article abstract.")

    row = build_single_prediction_frame(
        abstract=abstract,
        title=title,
        keywords=keywords,
        subjects=subjects,
    )
    probabilities = pipeline.predict_proba(row)[0]
    classes = pipeline.classes_
    top_indexes = np.argsort(probabilities)[::-1][:top_k]

    return pd.DataFrame(
        {
            "rank": range(1, len(top_indexes) + 1),
            "journal": [classes[index] for index in top_indexes],
            "score": [round(float(probabilities[index]), 4) for index in top_indexes],
        }
    )
