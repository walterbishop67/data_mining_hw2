"""Topic-clustering artifact helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .paths import CLUSTERED_DATASET_PATH, require_file


CLUSTER_NAMES = {
    0: "Cloud Computing and Web Services",
    1: "General CS, Formal Methods and Systems",
    2: "Computer Vision and Neural Learning",
    3: "Data Mining, Databases and Knowledge Discovery",
    4: "Optimization and Evolutionary Computation",
    5: "Information Systems and Human-Centered Computing",
    6: "Bioinformatics and Computational Biology",
    7: "Communication Networks and Telecommunications",
    8: "Fuzzy Logic and Decision Making",
    9: "Wireless Sensor Networks and Mobile Systems",
}

CLUSTER_TERMS = {
    0: ["cloud", "service", "computing", "web", "iot"],
    1: ["software", "systems", "theory", "methods", "model"],
    2: ["image", "recognition", "neural", "learning", "detection"],
    3: ["data", "mining", "clustering", "query", "big data"],
    4: ["optimization", "algorithm", "genetic", "evolutionary", "swarm"],
    5: ["information", "social", "user", "business", "knowledge"],
    6: ["biology", "gene", "protein", "molecular", "bioinformatics"],
    7: ["networks", "wireless", "telecommunications", "routing", "traffic"],
    8: ["fuzzy", "decision", "rough sets", "uncertainty", "intuitionistic"],
    9: ["sensor", "wireless sensor", "nodes", "network", "wsn"],
}


def infer_cluster_name(cluster_id: int) -> str:
    return CLUSTER_NAMES.get(int(cluster_id), f"Cluster {cluster_id}")


def get_cluster_keywords(cluster_id: int) -> list[str]:
    return CLUSTER_TERMS.get(int(cluster_id), [])


def load_clustered_data(path: Path = CLUSTERED_DATASET_PATH) -> pd.DataFrame:
    return pd.read_csv(require_file(path, "Topic clustering output"))


def summarize_cluster(frame: pd.DataFrame, cluster_id: int) -> dict[str, object]:
    subset = frame[frame["cluster"] == cluster_id].copy()
    top_journals = subset["JournalName"].value_counts().head(10).reset_index()
    top_journals.columns = ["JournalName", "Count"]
    sample_articles = subset[["AcademicRecordID", "JournalName"]].head(10)
    return {
        "name": infer_cluster_name(cluster_id),
        "keywords": get_cluster_keywords(cluster_id),
        "article_count": int(len(subset)),
        "top_journals": top_journals,
        "sample_articles": sample_articles,
    }

