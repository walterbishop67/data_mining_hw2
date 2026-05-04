# Computer Science Journal Finder

Final data mining project for recommending computer science journals from an article abstract and showing topic clusters from the publication dataset.

GitHub repository: https://github.com/walterbishop67/data_mining_hw2.git

## Submission Contents

- `app.py` - Streamlit interface.
- `main.py` - launcher for Streamlit and Jupyter Notebook.
- `src/` - data loading, preprocessing, model, recommender, final-project, and dashboard modules.
- `notebooks/20210808053_Final_Project.ipynb` - final notebook deliverable.
- `report/20210808053_IEEE_PROJECT_REPORT.tex` - IEEE report source.
- `report/20210808053_PROJE_DOKUMANTASYONU.md` - project documentation.
- `CompSciencePub.sqlite` - converted SQLite database used by the code.
- `exports/20210808053/journal_recommender_pipeline.pkl` - saved final journal recommender.
- `exports/20210808053/journal_recommender_meta.json` - saved model metrics.
- `exports/20210808053/step9_clustered_dataset.csv` - saved topic-clustering output.
- `tests/` - verification tests.

## Setup

```powershell
python -m pip install -r requirements.txt
```

## Run the App

```powershell
python main.py
```

The app has two tabs:

- Journal Recommender: enter an abstract and get the top 5 journal recommendations.
- Topic Clusters: inspect the KMeans topic clusters and their dominant journals.

To run the final notebook:

```powershell
python main.py --jupyter
```

To start both Streamlit and Jupyter:

```powershell
python main.py --both
```

## Rebuild Final Artifacts

The large intermediate `step5_enriched_dataset.csv` is intentionally not kept in the clean submission folder. Regenerate it only when retraining is needed:

```powershell
python -m src.final_project.enrichment
python -m src.final_project.training
python -m src.final_project.topic_modeling
```

## Test

```powershell
python -m unittest discover -s tests -v
```

The tests verify SQLite loading, text preprocessing, baseline recommendation, saved final model loading, top-5 prediction, and saved topic-cluster loading.

## Method

The final recommender uses separate TF-IDF channels for title, abstract, keywords, and subjects, then trains an `SGDClassifier(loss="log_loss")` for multi-class journal prediction. The Streamlit app loads the saved pipeline and returns the five highest-probability journals.

The current saved model is trained only on the stratified training split. The holdout test split is kept separate for evaluation and reaches `0.7061` Top-1 accuracy and `0.9323` Top-5 accuracy across 406 journal classes.

Topic clusters are generated with TF-IDF and KMeans over enriched article text. The dashboard displays cluster names, representative terms, article counts, top journals, and sample records.
