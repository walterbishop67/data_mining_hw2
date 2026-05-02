"""Model training utilities for recommendation and topic clustering."""

from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


def train_vectorizer(texts: pd.Series):
    """Fit a TF-IDF vectorizer and return the vectorizer plus sparse matrix."""
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.85,
        max_features=40000,
        sublinear_tf=True,
        norm="l2",
    )
    matrix = vectorizer.fit_transform(texts.fillna(""))
    return vectorizer, matrix


def cluster_topics(frame: pd.DataFrame, n_clusters: int = 8, random_state: int = 42) -> pd.DataFrame:
    """Cluster articles and produce concise topic descriptors."""
    vectorizer, matrix = train_vectorizer(frame["training_text"])
    n_clusters = max(2, min(int(n_clusters), len(frame)))

    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(matrix)
    terms = vectorizer.get_feature_names_out()

    rows = []
    for cluster_id in range(n_clusters):
        cluster_frame = frame.loc[labels == cluster_id].copy()
        centroid = model.cluster_centers_[cluster_id]
        top_term_ids = centroid.argsort()[-12:][::-1]
        top_terms = [terms[index] for index in top_term_ids]
        subjects = (
            cluster_frame["subjects"]
            .fillna("")
            .str.split(",")
            .explode()
            .str.strip()
            .replace("", pd.NA)
            .dropna()
            .value_counts()
            .head(5)
            .index.tolist()
        )

        rows.append(
            {
                "cluster": cluster_id,
                "article_count": int(len(cluster_frame)),
                "top_terms": ", ".join(top_terms),
                "dominant_subjects": ", ".join(subjects),
                "sample_journals": ", ".join(cluster_frame["journal"].value_counts().head(3).index),
            }
        )

    return pd.DataFrame(rows).sort_values("article_count", ascending=False).reset_index(drop=True)
