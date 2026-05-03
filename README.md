# Computer Science Journal Finder

This project implements the data mining final project described in `Data_mining_journal_homework 2.pdf`.
It recommends the top 5 journals for a new computer science article abstract and generates topic clusters from the provided publication database.

## Project Outputs

- Source code for data loading, preprocessing, modeling, and recommendation.
- Jupyter Notebook workflow in `notebooks/journal_finder_project.ipynb`.
- Streamlit software interface in `app.py`.
- IEEE conference report in `report/ieee_report.docx`.
- IEEE conference report source in `report/ieee_report.tex`.
- Readable report copy in `report/ieee_report.md`.
- Full-dataset validation results: 23,061 usable article abstracts, 455 journals, and 80 subject areas.

## Data Files

- `CS_JournalAbstracts/CompSciencePub.bak` is the original SQL Server backup provided with the assignment. It is kept as raw source material and ignored by Git because it is large.
- `CompSciencePub.sqlite` is the converted SQLite database used by the Python code, notebook, tests, and Streamlit app.
- The project does not read the `.bak` file directly; restoring/exporting it to SQLite is required if `CompSciencePub.sqlite` is missing.

## Submission Checklist

- Source code with GitHub link: included in `src/`, `app.py`, and the GitHub section below.
- Jupyter Notebook format: included as `notebooks/journal_finder_project.ipynb` with saved outputs.
- IEEE Conference report with literature review: included as `report/ieee_report.docx` and `report/ieee_report.tex`.
- Journal finder software tailored to computer science subject areas: implemented in the Streamlit app and reusable modules.
- Top-5 journal list from an entered article abstract: implemented by `recommend_journals(..., top_n=5)`.
- Topic clusters for subject areas: implemented by `cluster_topics`.

## Setup

```powershell
python -m pip install -r requirements.txt
```

## Run the App

```powershell
streamlit run app.py
```

Paste an article abstract into the text area and click `Find journals`. The app returns the top-5 journal recommendations and displays generated topic clusters.

## Run the Notebook

```powershell
jupyter notebook notebooks/journal_finder_project.ipynb
```

The notebook has been executed once and includes saved outputs for dataset summary, TF-IDF training, journal recommendation, and topic clustering.

## Run Tests

```powershell
python -m unittest discover -s tests -v
```

The tests validate text cleaning, SQLite loading, dataset summary counts, TF-IDF training, top-5 journal recommendation, empty abstract validation, and topic clustering.

## Method

The project uses the provided `CompSciencePub.sqlite` database. Article title, abstract, author keywords, Web of Science Keyword Plus, and subject fields are cleaned and combined into one training text. Title, author keyword, and subject fields are intentionally weighted more strongly because they identify the article scope and improve journal matching. A TF-IDF vectorizer with up to 80,000 unigram/bigram features converts articles and user abstracts into vectors. Cosine similarity finds the closest known articles, and scores are aggregated by journal to produce the final top-5 ranking.

Topic clusters are generated with KMeans over TF-IDF vectors. Each cluster is summarized with high-weight terms, dominant Web of Science subjects, and sample journals.

## GitHub Link

https://github.com/walterbishop67/data_mining_hw2.git
