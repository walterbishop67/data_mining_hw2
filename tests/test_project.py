import unittest
from pathlib import Path

from src.data_loader import load_dataset, summarize_dataset
from src.final_project.clustering import load_clustered_data
from src.final_project.paths import (
    CLUSTERED_DATASET_PATH,
    DB_PATH as FINAL_PROJECT_DB_PATH,
    PIPELINE_PATH,
)
from src.final_project.recommender_model import (
    load_metadata,
    load_pipeline,
    _patch_sklearn_private_validation_helpers,
    recommend_journals as recommend_saved_journals,
)
from src.modeling import cluster_topics, train_vectorizer
from src.recommender import format_recommendations, recommend_journals
from src.text_preprocessing import clean_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "CompSciencePub.sqlite"


class JournalFinderProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = load_dataset(DB_PATH, limit=1200)
        cls.vectorizer, cls.matrix = train_vectorizer(cls.frame["training_text"])

    def test_clean_text_removes_html_and_normalizes_text(self):
        text = clean_text("<p>Deep&nbsp;Learning for Software-Defect Prediction!</p>")

        self.assertNotIn("<p>", text)
        self.assertIn("deep", text)
        self.assertIn("learning", text)
        self.assertIn("software-defect", text)

    def test_dataset_loads_required_fields(self):
        required_columns = {
            "record_id",
            "title",
            "abstract",
            "journal",
            "subjects",
            "training_text",
        }

        self.assertFalse(self.frame.empty)
        self.assertTrue(required_columns.issubset(self.frame.columns))
        self.assertGreaterEqual(len(self.frame), 1000)
        self.assertTrue((self.frame["training_text"].str.len() > 40).all())

    def test_dataset_summary_counts_are_positive(self):
        summary = summarize_dataset(self.frame)

        self.assertGreater(summary["articles"], 0)
        self.assertGreater(summary["journals"], 0)
        self.assertGreater(summary["subjects"], 0)

    def test_tfidf_matrix_matches_dataset_rows(self):
        self.assertEqual(self.matrix.shape[0], len(self.frame))
        self.assertGreater(self.matrix.shape[1], 100)

    def test_recommend_journals_returns_top_five_sorted_results(self):
        sample_abstract = """
        This paper proposes a machine learning approach for software defect prediction
        using source code metrics, repository mining, neural classification, and
        cross project validation. The experiments evaluate precision, recall,
        accuracy, and generalization on open source software systems.
        """

        result = format_recommendations(
            recommend_journals(sample_abstract, self.frame, self.vectorizer, self.matrix, top_n=5)
        )

        self.assertEqual(len(result), 5)
        self.assertEqual(result["rank"].tolist(), [1, 2, 3, 4, 5])
        self.assertTrue(result["score"].is_monotonic_decreasing)
        self.assertTrue(result["journal"].notna().all())

    def test_recommend_journals_rejects_empty_abstract(self):
        with self.assertRaises(ValueError):
            recommend_journals("", self.frame, self.vectorizer, self.matrix)

    def test_topic_clustering_returns_requested_clusters(self):
        clusters = cluster_topics(self.frame, n_clusters=4)

        self.assertEqual(len(clusters), 4)
        self.assertTrue((clusters["article_count"] > 0).all())
        self.assertTrue(clusters["top_terms"].str.len().gt(0).all())

    def test_final_project_paths_match_project_layout(self):
        self.assertEqual(FINAL_PROJECT_DB_PATH, DB_PATH)
        self.assertTrue(FINAL_PROJECT_DB_PATH.is_file())
        self.assertTrue(PIPELINE_PATH.is_file())
        self.assertTrue(CLUSTERED_DATASET_PATH.is_file())

    def test_saved_final_recommender_returns_top_five(self):
        pipeline = load_pipeline()
        result = recommend_saved_journals(
            pipeline,
            abstract=(
                "This study proposes a graph neural network model for mining "
                "software repository data and predicting source code defects."
            ),
            title="Graph neural networks for software defect prediction",
            keywords="software mining, defect prediction, graph neural networks",
            subjects="Artificial Intelligence, Software Engineering",
            top_k=5,
        )

        self.assertEqual(len(result), 5)
        self.assertEqual(result["rank"].tolist(), [1, 2, 3, 4, 5])
        self.assertTrue(result["score"].is_monotonic_decreasing)
        self.assertTrue(result["journal"].notna().all())

    def test_saved_final_recommender_uses_clean_holdout_split(self):
        metadata = load_metadata()

        self.assertEqual(metadata["split_strategy"], "stratified_train_test_split")
        self.assertEqual(metadata["saved_model_fit_scope"], "train_split_only")
        self.assertGreaterEqual(metadata["holdout_top1_accuracy"], 0.70)
        self.assertGreaterEqual(metadata["holdout_top5_accuracy"], 0.93)

    def test_sklearn_validation_compatibility_patch(self):
        import sklearn.utils.validation as validation

        original = getattr(validation, "_is_pandas_df", None)
        if original is not None:
            delattr(validation, "_is_pandas_df")
        try:
            _patch_sklearn_private_validation_helpers()

            self.assertTrue(hasattr(validation, "_is_pandas_df"))
            self.assertTrue(validation._is_pandas_df(self.frame.head(1)))
        finally:
            if original is not None:
                validation._is_pandas_df = original

    def test_saved_topic_cluster_artifact_loads(self):
        cluster_frame = load_clustered_data()

        self.assertFalse(cluster_frame.empty)
        required_columns = {"AcademicRecordID", "JournalName", "cluster"}
        self.assertTrue(required_columns.issubset(cluster_frame.columns))
        self.assertGreaterEqual(cluster_frame["cluster"].nunique(), 2)


if __name__ == "__main__":
    unittest.main()
