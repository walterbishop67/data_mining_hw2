"""Build the enriched final-project dataset from the SQLite publication data."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from .paths import DB_PATH, STEP5_DATASET_PATH, require_file
from .text import clean_text


ENRICHED_DATASET_QUERY = """
SELECT
    ar.AcademicRecordID,
    ar.Title,
    ara.AbstractText,
    ar.PublicationId,
    p.Name AS JournalName,
    ar.PubYear,
    kw.keywords_text,
    kp.keyword_plus_text,
    subj.subjects_text
FROM AcademicRecord ar
LEFT JOIN AcademicRecordAbstract ara
    ON ar.AcademicRecordID = ara.AcademicRecordId
LEFT JOIN Publication p
    ON ar.PublicationId = p.PublicationID
LEFT JOIN (
    SELECT
        ark.AcademicRecordId,
        GROUP_CONCAT(ak.Name, ' ') AS keywords_text
    FROM AcademicRecordKeyword ark
    LEFT JOIN AcademicKeyword ak
        ON ark.AcademicKeywordId = ak.AcademicKeywordID
    GROUP BY ark.AcademicRecordId
) kw
    ON ar.AcademicRecordID = kw.AcademicRecordId
LEFT JOIN (
    SELECT
        arkp.AcademicRecordId,
        GROUP_CONCAT(akp.Name, ' ') AS keyword_plus_text
    FROM AcademicRecordKeywordPlus arkp
    LEFT JOIN AcademicKeywordPlus akp
        ON arkp.AcademicKeywordPlusId = akp.AcademicKeywordPlusID
    GROUP BY arkp.AcademicRecordId
) kp
    ON ar.AcademicRecordID = kp.AcademicRecordId
LEFT JOIN (
    SELECT
        ars.AcademicRecordId,
        GROUP_CONCAT(asu.NameEn, ' ') AS subjects_text
    FROM AcademicRecordSubject ars
    LEFT JOIN AcademicSubject asu
        ON ars.AcademicSubjectId = asu.AcademicSubjectID
    GROUP BY ars.AcademicRecordId
) subj
    ON ar.AcademicRecordID = subj.AcademicRecordId
"""


def build_enriched_dataset(
    db_path: Path = DB_PATH,
    output_path: Path = STEP5_DATASET_PATH,
) -> pd.DataFrame:
    """Create the rich article text used by the final recommender and clustering."""
    with sqlite3.connect(require_file(db_path, "SQLite database")) as connection:
        frame = pd.read_sql_query(ENRICHED_DATASET_QUERY, connection)

    frame = frame.dropna(subset=["AbstractText", "JournalName"]).copy()
    frame = frame.drop_duplicates(subset=["AcademicRecordID"]).reset_index(drop=True)

    frame["title_clean"] = frame["Title"].map(clean_text)
    frame["abstract_clean"] = frame["AbstractText"].map(clean_text)
    frame["keywords_clean"] = frame["keywords_text"].map(clean_text)
    frame["keyword_plus_clean"] = frame["keyword_plus_text"].map(clean_text)
    frame["subjects_clean"] = frame["subjects_text"].map(clean_text)

    frame["text_title_abstract"] = (
        frame["title_clean"] + " " + frame["abstract_clean"]
    ).str.strip()
    frame["text_rich"] = (
        frame["title_clean"]
        + " "
        + frame["abstract_clean"]
        + " "
        + frame["keywords_clean"]
        + " "
        + frame["keyword_plus_clean"]
        + " "
        + frame["subjects_clean"]
    ).str.replace(r"\s+", " ", regex=True).str.strip()
    frame["rich_word_count"] = frame["text_rich"].str.split().str.len()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, encoding="utf-8")
    return frame


if __name__ == "__main__":
    dataset = build_enriched_dataset()
    print(f"Saved {len(dataset):,} enriched rows to {STEP5_DATASET_PATH}")
