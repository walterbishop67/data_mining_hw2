"""Central paths for the final-project pipeline and generated artifacts."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPORT_DIR = PROJECT_ROOT / "exports" / "20210808053"

DB_PATH = PROJECT_ROOT / "CompSciencePub.sqlite"
STEP5_DATASET_PATH = EXPORT_DIR / "step5_enriched_dataset.csv"
CLUSTERED_DATASET_PATH = EXPORT_DIR / "step9_clustered_dataset.csv"
PIPELINE_PATH = EXPORT_DIR / "journal_recommender_pipeline.pkl"
RECOMMENDER_META_PATH = EXPORT_DIR / "journal_recommender_meta.json"


def require_file(path: Path, purpose: str) -> Path:
    """Return an existing artifact path or raise a clear setup error."""
    if not path.is_file():
        raise FileNotFoundError(
            f"{purpose} not found: {path}. "
            "Run the matching final-project pipeline step or place the generated "
            "artifact in the documented project location."
        )
    return path
