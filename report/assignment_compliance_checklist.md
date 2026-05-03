# Assignment Compliance Checklist

This checklist maps the project to `Data_mining_journal_homework 2.pdf`.

## PDF Requirements

| PDF requirement | Project implementation | Status |
|---|---|---|
| Submit source codes with GitHub link | Source code is in `src/` and `app.py`; GitHub link is listed in `README.md`. | Done |
| Submit Jupyter Notebook format | `notebooks/journal_finder_project.ipynb` contains the full executed workflow. | Done |
| Submit project report in IEEE Conference format with literature review | `report/ieee_report.tex`, `report/ieee_report.md`, and `report/ieee_report.docx` are included; the report has a Related Work section and references. | Done |
| Implement a journal finder software tailored to computer science subject areas | Streamlit app in `app.py`; recommendation logic in `src/recommender.py`. | Done |
| Generate clusters of topics for subject areas | `cluster_topics` in `src/modeling.py`; notebook exports `notebook_topic_clusters.csv`. | Done |
| Use provided database entities/attributes freely | Uses `AcademicRecord`, `AcademicRecordAbstract`, `Publication`, `AcademicRecordKeyword`, `AcademicRecordKeywordPlus`, and `AcademicRecordSubject`. | Done |
| When author enters article abstract, list top 5 most relevant journals | `recommend_journals(..., top_n=5)` and Streamlit button return top-5 journals. | Done |

## Important Interpretation

The PDF does not specify a minimum abstract length. The application therefore rejects only empty input and returns recommendations for any non-empty abstract text.

## Verification

- Notebook executed from start to finish with saved outputs.
- Unit tests pass with `python -m unittest discover -s tests`.
- Streamlit app runs the top-5 recommendation workflow.
