"""Training entry points for the saved final journal recommender."""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .paths import PIPELINE_PATH, RECOMMENDER_META_PATH, STEP5_DATASET_PATH, require_file
from .text import build_feature_frame

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")


def build_pipeline() -> Pipeline:
    """Create the multi-channel TF-IDF + SGD classifier pipeline."""
    features = ColumnTransformer(
        transformers=[
            (
                "title",
                TfidfVectorizer(
                    stop_words="english",
                    max_features=6000,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                ),
                "title_channel",
            ),
            (
                "abstract",
                TfidfVectorizer(
                    stop_words="english",
                    max_features=12000,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                ),
                "abstract_channel",
            ),
            (
                "keywords",
                TfidfVectorizer(
                    stop_words="english",
                    max_features=6000,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                ),
                "keywords_channel",
            ),
            (
                "subjects",
                TfidfVectorizer(
                    stop_words="english",
                    max_features=3000,
                    ngram_range=(1, 1),
                    min_df=2,
                    sublinear_tf=True,
                ),
                "subjects_channel",
            ),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    classifier = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=3e-5,
        max_iter=3000,
        tol=1e-4,
        random_state=42,
        n_jobs=1,
        early_stopping=False,
    )
    return Pipeline([("features", features), ("clf", classifier)])


def top1_top5_accuracy(model: Pipeline, features: pd.DataFrame, labels: np.ndarray) -> tuple[float, float]:
    """Measure top-1 and top-5 journal prediction accuracy."""
    top1 = accuracy_score(labels, model.predict(features))
    probabilities = model.predict_proba(features)
    classes = model.classes_
    top5_hits = 0
    for row_index, label in enumerate(labels):
        top_indexes = np.argsort(probabilities[row_index])[::-1][:5]
        if label in set(classes[top_indexes]):
            top5_hits += 1
    return float(top1), float(top5_hits / len(labels)) if len(labels) else 0.0


def load_training_data(
    input_path: Path = STEP5_DATASET_PATH,
    min_samples_per_journal: int = 5,
    min_words: int = 20,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Load and filter the enriched dataset for classifier training."""
    frame = pd.read_csv(require_file(input_path, "Step 5 enriched dataset"))
    needed = [
        "JournalName",
        "title_clean",
        "abstract_clean",
        "keywords_clean",
        "keyword_plus_clean",
        "subjects_clean",
    ]
    frame = frame[needed].dropna(subset=["JournalName", "abstract_clean"]).copy()

    journal_counts = frame["JournalName"].value_counts()
    valid_journals = journal_counts[journal_counts >= min_samples_per_journal].index
    frame = frame[frame["JournalName"].isin(valid_journals)].reset_index(drop=True)

    features = build_feature_frame(frame)
    labels = frame["JournalName"].reset_index(drop=True).to_numpy(dtype=str)
    word_counts = features["abstract_channel"].str.split().str.len()
    mask = word_counts >= min_words
    return features.loc[mask].reset_index(drop=True), labels[mask.to_numpy()]


def train_final_recommender(
    input_path: Path = STEP5_DATASET_PATH,
    pipeline_path: Path = PIPELINE_PATH,
    metadata_path: Path = RECOMMENDER_META_PATH,
) -> dict[str, float | int]:
    """Train, evaluate, and persist the final journal recommender."""
    min_samples_per_journal = 5
    min_words = 20
    features, labels = load_training_data(
        input_path=input_path,
        min_samples_per_journal=min_samples_per_journal,
        min_words=min_words,
    )
    model = build_pipeline()
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    model.fit(x_train, y_train)
    train_top1, train_top5 = top1_top5_accuracy(model, x_train, y_train)
    test_top1, test_top5 = top1_top5_accuracy(model, x_test, y_test)

    metadata = {
        "n_articles": int(features.shape[0]),
        "n_journals": int(len(np.unique(labels))),
        "min_words_abstract": min_words,
        "min_samples_per_journal": min_samples_per_journal,
        "holdout_test_size": 0.2,
        "holdout_random_state": 42,
        "split_strategy": "stratified_train_test_split",
        "saved_model_fit_scope": "train_split_only",
        "train_n_samples": int(len(y_train)),
        "train_top1_accuracy": train_top1,
        "train_top5_accuracy": train_top5,
        "test_n_samples": int(len(y_test)),
        "holdout_top1_accuracy": test_top1,
        "holdout_top5_accuracy": test_top5,
        "classifier": "SGDClassifier(loss='log_loss', penalty='l2')",
        "classifier_alpha": 3e-5,
        "classifier_early_stopping": False,
    }

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    pipeline_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, pipeline_path)
    return metadata


if __name__ == "__main__":
    metrics = train_final_recommender()
    print(json.dumps(metrics, indent=2))
