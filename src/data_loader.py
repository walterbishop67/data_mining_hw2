"""SQLite loading and feature assembly for the journal finder project."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

try:
    from .text_preprocessing import clean_text, join_terms
except ImportError:  # Allows running modules directly from the project root.
    from text_preprocessing import clean_text, join_terms


DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "CompSciencePub.sqlite"


def load_dataset(db_path: str | Path = DEFAULT_DB_PATH, limit: int | None = None) -> pd.DataFrame:
    """Load article, abstract, journal, keyword, and subject fields."""
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite database not found: {db_path}")

    limit_clause = f"LIMIT {int(limit)}" if limit else ""
    query = f"""
        WITH author_keywords AS (
            SELECT
                ark.AcademicRecordId,
                group_concat(ak.Name, ', ') AS author_keywords
            FROM AcademicRecordKeyword ark
            JOIN AcademicKeyword ak
                ON ak.AcademicKeywordID = ark.AcademicKeywordId
            GROUP BY ark.AcademicRecordId
        ),
        keyword_plus AS (
            SELECT
                arkp.AcademicRecordId,
                group_concat(akp.Name, ', ') AS keyword_plus
            FROM AcademicRecordKeywordPlus arkp
            JOIN AcademicKeywordPlus akp
                ON akp.AcademicKeywordPlusID = arkp.AcademicKeywordPlusId
            GROUP BY arkp.AcademicRecordId
        ),
        subjects AS (
            SELECT
                ars.AcademicRecordId,
                group_concat(s.NameEn, ', ') AS subjects
            FROM AcademicRecordSubject ars
            JOIN AcademicSubject s
                ON s.AcademicSubjectID = ars.AcademicSubjectId
            GROUP BY ars.AcademicRecordId
        )
        SELECT
            ar.AcademicRecordID AS record_id,
            ar.Title AS title,
            ara.AbstractText AS abstract,
            ar.PubYear AS pub_year,
            ar.CiteCount AS cite_count,
            ar.ImpactFactor AS impact_factor,
            ar.QValue AS q_value,
            p.PublicationID AS publication_id,
            p.Name AS journal,
            COALESCE(author_keywords.author_keywords, '') AS author_keywords,
            COALESCE(keyword_plus.keyword_plus, '') AS keyword_plus,
            COALESCE(subjects.subjects, '') AS subjects
        FROM AcademicRecord ar
        JOIN AcademicRecordAbstract ara
            ON ara.AcademicRecordId = ar.AcademicRecordID
        JOIN Publication p
            ON p.PublicationID = ar.PublicationId
        LEFT JOIN author_keywords
            ON author_keywords.AcademicRecordId = ar.AcademicRecordID
        LEFT JOIN keyword_plus
            ON keyword_plus.AcademicRecordId = ar.AcademicRecordID
        LEFT JOIN subjects
            ON subjects.AcademicRecordId = ar.AcademicRecordID
        WHERE ara.AbstractText IS NOT NULL
          AND LENGTH(TRIM(ara.AbstractText)) > 0
          AND p.Name IS NOT NULL
        {limit_clause}
    """

    with sqlite3.connect(db_path) as connection:
        frame = pd.read_sql_query(query, connection)

    return build_training_text(frame)


def build_training_text(frame: pd.DataFrame) -> pd.DataFrame:
    """Add normalized text fields used by recommendation and clustering."""
    df = frame.copy()
    for column in ["title", "abstract", "author_keywords", "keyword_plus", "subjects"]:
        if column not in df:
            df[column] = ""

    df["clean_title"] = df["title"].map(clean_text)
    df["clean_abstract"] = df["abstract"].map(clean_text)
    df["clean_author_keywords"] = df["author_keywords"].map(join_terms)
    df["clean_keyword_plus"] = df["keyword_plus"].map(join_terms)
    df["clean_subjects"] = df["subjects"].map(join_terms)

    df["training_text"] = (
        df["clean_title"]
        + " "
        + df["clean_abstract"]
        + " "
        + df["clean_author_keywords"]
        + " "
        + df["clean_keyword_plus"]
        + " "
        + df["clean_subjects"]
    ).str.replace(r"\s+", " ", regex=True).str.strip()

    df = df[df["training_text"].str.len() > 40].reset_index(drop=True)
    return df


def summarize_dataset(frame: pd.DataFrame) -> dict[str, int]:
    """Return compact dataset counts for README/notebook display."""
    return {
        "articles": int(len(frame)),
        "journals": int(frame["journal"].nunique()),
        "subjects": int(
            frame["subjects"]
            .fillna("")
            .str.split(",")
            .explode()
            .str.strip()
            .replace("", pd.NA)
            .dropna()
            .nunique()
        ),
    }
