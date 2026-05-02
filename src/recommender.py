"""Journal recommendation logic."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

try:
    from .text_preprocessing import clean_text
except ImportError:
    from text_preprocessing import clean_text


def recommend_journals(
    input_abstract: str,
    frame: pd.DataFrame,
    vectorizer,
    matrix,
    top_n: int = 5,
    neighbor_count: int = 150,
) -> pd.DataFrame:
    """Rank journals by similarity between input abstract and known articles."""
    cleaned = clean_text(input_abstract)
    if len(cleaned.split()) < 20:
        raise ValueError("Please enter a longer abstract with at least 20 words.")

    query_vector = vectorizer.transform([cleaned])
    similarities = cosine_similarity(query_vector, matrix).ravel()

    if similarities.size == 0 or float(similarities.max()) <= 0:
        return pd.DataFrame(columns=["rank", "journal", "score", "matched_articles", "example_title"])

    top_article_ids = np.argsort(similarities)[-neighbor_count:][::-1]
    candidates = frame.iloc[top_article_ids].copy()
    candidates["similarity"] = similarities[top_article_ids]

    journal_scores = (
        candidates.groupby("journal")
        .agg(
            score=("similarity", "mean"),
            best_score=("similarity", "max"),
            matched_articles=("record_id", "count"),
            example_title=("title", "first"),
            subjects=("subjects", "first"),
        )
        .reset_index()
    )
    journal_scores["score"] = (0.75 * journal_scores["best_score"]) + (0.25 * journal_scores["score"])

    result = journal_scores.sort_values(
        ["score", "matched_articles"], ascending=[False, False]
    ).head(top_n)
    result = result.reset_index(drop=True)
    result.insert(0, "rank", result.index + 1)

    return result[["rank", "journal", "score", "matched_articles", "example_title", "subjects"]]


def format_recommendations(result: pd.DataFrame) -> pd.DataFrame:
    """Prepare recommendation output for display."""
    if result.empty:
        return result

    formatted = result.copy()
    formatted["score"] = formatted["score"].map(lambda value: round(float(value), 4))
    return formatted
