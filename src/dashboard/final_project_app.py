"""Streamlit UI for the saved final-project artifacts."""

from __future__ import annotations

import streamlit as st

from src.final_project.clustering import (
    infer_cluster_name,
    load_clustered_data,
    summarize_cluster,
)
from src.final_project.recommender_model import (
    load_metadata,
    load_pipeline,
    recommend_journals,
)


@st.cache_resource(show_spinner=False)
def get_saved_pipeline():
    return load_pipeline()


@st.cache_data(show_spinner=False)
def get_saved_metadata():
    return load_metadata()


@st.cache_data(show_spinner=False)
def get_saved_clusters():
    return load_clustered_data()


def render_sidebar(pipeline, metadata: dict | None) -> None:
    st.sidebar.title("Project Info")
    st.sidebar.markdown("**Model:** Multi-channel TF-IDF + SGD logistic classifier")
    st.sidebar.markdown("**Channels:** title, abstract, keywords, subjects")
    st.sidebar.markdown(f"**Journal classes:** {len(pipeline.classes_)}")

    if metadata:
        st.sidebar.markdown(
            f"**Training articles:** {metadata['n_articles']:,} "
            f"(abstract >= {metadata['min_words_abstract']} words)"
        )
        st.sidebar.markdown(
            "**Holdout test:** "
            f"Top-1 = {metadata['holdout_top1_accuracy']:.4f}, "
            f"Top-5 = {metadata['holdout_top5_accuracy']:.4f}"
        )


def render_recommender_tab(pipeline) -> None:
    abstract = st.text_area(
        "Article abstract",
        height=250,
        placeholder="Paste the article abstract here...",
    )

    with st.expander("Optional fields"):
        title = st.text_input("Title", placeholder="Can be left empty")
        keywords = st.text_area(
            "Keywords",
            height=80,
            placeholder="Example: deep learning, graph neural networks",
        )
        subjects = st.text_area(
            "Subject terms",
            height=80,
            placeholder="Example: Artificial Intelligence, Data Mining",
        )

    if st.button("Recommend Journals", type="primary"):
        try:
            result = recommend_journals(
                pipeline,
                abstract=abstract,
                title=title,
                keywords=keywords,
                subjects=subjects,
                top_k=5,
            )
            st.dataframe(result, width="stretch", hide_index=True)
        except ValueError as exc:
            st.warning(str(exc))


def render_clusters_tab(cluster_frame) -> None:
    cluster_ids = sorted(cluster_frame["cluster"].unique().tolist())
    selected_cluster = st.selectbox(
        "Cluster",
        cluster_ids,
        format_func=lambda value: f"{value} - {infer_cluster_name(value)}",
    )
    summary = summarize_cluster(cluster_frame, selected_cluster)

    st.metric("Articles in cluster", f"{summary['article_count']:,}")
    if summary["keywords"]:
        st.caption("Top terms: " + ", ".join(summary["keywords"]))

    st.subheader("Top Journals")
    st.dataframe(summary["top_journals"], width="stretch", hide_index=True)

    st.subheader("Sample Articles")
    st.dataframe(summary["sample_articles"], width="stretch", hide_index=True)


def render_final_project_app() -> None:
    pipeline = get_saved_pipeline()
    metadata = get_saved_metadata()
    cluster_frame = get_saved_clusters()

    render_sidebar(pipeline, metadata)

    st.title("Computer Science Journal Finder")
    st.write(
        "Enter an article abstract to get the top 5 recommended journals. "
        "Optional title, keyword, and subject fields align the input with the "
        "multi-channel model."
    )

    recommender_tab, clusters_tab = st.tabs(["Journal Recommender", "Topic Clusters"])
    with recommender_tab:
        render_recommender_tab(pipeline)
    with clusters_tab:
        render_clusters_tab(cluster_frame)
