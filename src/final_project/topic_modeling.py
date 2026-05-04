"""Topic-clustering pipeline for the enriched final-project dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from .paths import CLUSTERED_DATASET_PATH, STEP5_DATASET_PATH, require_file


def build_topic_clusters(
    input_path: Path = STEP5_DATASET_PATH,
    output_path: Path = CLUSTERED_DATASET_PATH,
    n_clusters: int = 10,
) -> pd.DataFrame:
    """Fit TF-IDF + KMeans over rich article text and persist cluster labels."""
    frame = pd.read_csv(require_file(input_path, "Step 5 enriched dataset"))
    frame = frame[["AcademicRecordID", "JournalName", "text_rich"]].dropna().copy()

    journal_counts = frame["JournalName"].value_counts()
    valid_journals = journal_counts[journal_counts >= 5].index
    frame = frame[frame["JournalName"].isin(valid_journals)].copy()

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2),
        min_df=3,
    )
    matrix = vectorizer.fit_transform(frame["text_rich"])
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    frame["cluster"] = model.fit_predict(matrix)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, encoding="utf-8")
    return frame


if __name__ == "__main__":
    clustered = build_topic_clusters()
    print(clustered["cluster"].value_counts().sort_index())

