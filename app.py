from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.data_loader import DEFAULT_DB_PATH, load_dataset, summarize_dataset
from src.modeling import cluster_topics, train_vectorizer
from src.recommender import format_recommendations, recommend_journals


st.set_page_config(page_title="Computer Science Journal Finder", layout="wide")


@st.cache_data(show_spinner=False)
def get_dataset(db_path: str):
    return load_dataset(db_path)


@st.cache_resource(show_spinner=False)
def get_vector_model(texts):
    return train_vectorizer(texts)


@st.cache_data(show_spinner=False)
def get_topic_clusters(_frame, n_clusters: int):
    return cluster_topics(_frame, n_clusters=n_clusters)


db_path = Path(DEFAULT_DB_PATH)

st.title("Computer Science Journal Finder")
st.caption("TF-IDF, cosine similarity, and topic clustering on the provided Web of Science journal dataset.")

if not db_path.exists():
    st.error(f"Database not found: {db_path}")
    st.stop()

with st.spinner("Loading database and model..."):
    df = get_dataset(str(db_path))
    vectorizer, matrix = get_vector_model(df["training_text"])

summary = summarize_dataset(df)
metric_a, metric_b, metric_c = st.columns(3)
metric_a.metric("Articles", f"{summary['articles']:,}")
metric_b.metric("Journals", f"{summary['journals']:,}")
metric_c.metric("Subjects", f"{summary['subjects']:,}")

abstract = st.text_area(
    "Article abstract",
    height=220,
    placeholder="Paste the abstract of a computer science article here...",
)

left, right = st.columns([1, 1])
with left:
    top_n = st.slider("Number of journals", min_value=5, max_value=10, value=5)
with right:
    n_clusters = st.slider("Topic clusters", min_value=4, max_value=14, value=8)

if st.button("Find journals", type="primary"):
    try:
        recommendations = recommend_journals(abstract, df, vectorizer, matrix, top_n=top_n)
        st.subheader("Recommended journals")
        st.dataframe(format_recommendations(recommendations), use_container_width=True, hide_index=True)
    except ValueError as exc:
        st.warning(str(exc))

st.subheader("Generated topic clusters")
with st.spinner("Generating topic clusters..."):
    clusters = get_topic_clusters(df, n_clusters)
st.dataframe(clusters, use_container_width=True, hide_index=True)
