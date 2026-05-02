# Computer Science Journal Finder

This project implements the data mining final project described in `Data_mining_journal_homework 2.pdf`.
It recommends the top journals for a new computer science article abstract and generates topic clusters from the provided publication database.

## Project Outputs

- Source code for data loading, preprocessing, modeling, and recommendation.
- Jupyter Notebook workflow in `notebooks/journal_finder_project.ipynb`.
- Streamlit software interface in `app.py`.
- IEEE-style report draft in `report/ieee_report.md`.
- Full-dataset validation results: 23,061 articles, 455 journals, and 80 subject areas.

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

The tests validate text cleaning, SQLite loading, dataset summary counts, TF-IDF training, top-5 journal recommendation, short abstract validation, and topic clustering.

## Method

The project uses the provided `CompSciencePub.sqlite` database. Article title, abstract, author keywords, Web of Science Keyword Plus, and subject fields are cleaned and combined into one training text. A TF-IDF vectorizer converts articles and user abstracts into vectors. Cosine similarity finds the closest known articles, and scores are aggregated by journal to produce the final journal ranking.

Topic clusters are generated with KMeans over TF-IDF vectors. Each cluster is summarized with high-weight terms, dominant Web of Science subjects, and sample journals.

## GitHub Link

https://github.com/walterbishop67/data_mining_hw2.git
