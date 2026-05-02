# Journal Finder for Computer Science Articles

This file is a readable copy of the project report. The IEEE conference source version is available in `report/ieee_report.tex`.

## Abstract

Selecting a suitable journal is a common challenge for researchers because journal scope, subject area, and article terminology must align. This project builds a journal finder for computer science articles using a publication database containing abstracts, journals, keywords, and subject categories. The system recommends the five most relevant journals for a new article abstract and generates topic clusters for the corpus.

## Keywords

Journal recommendation, text mining, TF-IDF, cosine similarity, clustering, computer science publications.

## I. Introduction

Journal articles are one of the most important research outputs. For authors, choosing a journal that fits the content of an article can improve the chance of reaching the right audience. The goal of this project is to use data mining methods to recommend relevant computer science journals based on an article abstract.

## II. Related Work

Text mining is widely used in document retrieval, topic discovery, and recommendation systems. Vector space models such as TF-IDF represent documents by term importance and support efficient similarity matching. Salton and Buckley showed that term weighting is a practical foundation for information retrieval because it balances term frequency with collection-level rarity. In this project, the same idea is used to represent each article as a weighted vector.

Cosine similarity is a standard method for ranking documents by semantic and lexical overlap in sparse vector spaces. It is especially suitable for TF-IDF because the angle between vectors compares term distribution while reducing the effect of document length. For journal recommendation, this makes it possible to compare a new abstract against thousands of existing article records.

Clustering methods are also commonly used to discover groups of similar documents and summarize large text collections by topic. KMeans is a simple and interpretable baseline for partitioning vectorized text into topical groups. Sebastiani's survey of text categorization also supports the use of machine learning methods for organizing and classifying textual documents. These studies motivate the use of TF-IDF, cosine similarity, and KMeans as transparent data mining methods for the assignment.

## III. Dataset

The provided SQLite database contains publication records from computer science journals. The main entities used in this project are:

- `AcademicRecord`: article metadata such as title, year, citation count, and publication id.
- `AcademicRecordAbstract`: abstract text.
- `Publication`: journal name and identifiers.
- `AcademicRecordKeyword`: author keywords.
- `AcademicRecordKeywordPlus`: Web of Science keyword terms.
- `AcademicRecordSubject`: subject categories assigned to the article.

## IV. Methodology

The preprocessing step removes HTML tags from abstracts, normalizes whitespace, lowercases text, and combines title, abstract, keywords, keyword plus terms, and subjects into one training document for each article.

For journal recommendation, the system fits a TF-IDF vectorizer on all article training documents. When a user enters a new abstract, the abstract is cleaned and transformed with the same vectorizer. Cosine similarity is calculated between the input abstract and known articles. The most similar articles are grouped by journal, and journal scores are calculated from the best and average similarity values. The final output is the top five journals.

For topic discovery, the same TF-IDF representation is clustered using KMeans. Each cluster is summarized by top centroid terms, dominant subject categories, and sample journals.

## V. Implementation

The implementation is organized as reusable Python modules under `src/`, a Jupyter Notebook for reproducible analysis, and a Streamlit application for interactive use. The user interface accepts an abstract, displays recommended journals, and shows generated topic clusters.

## VI. Results and Evaluation

The final dataset used by the system contains 23,061 articles with abstracts, 455 journals, and 80 distinct subject areas. The TF-IDF representation contains 40,000 features after applying document frequency thresholds and English stop-word filtering.

For a sample abstract about machine learning methods for software defect prediction, the full-dataset system returned five relevant journals:

1. JOURNAL OF WEB ENGINEERING
2. AUTOMATED SOFTWARE ENGINEERING
3. IEEE TRANSACTIONS ON SOFTWARE ENGINEERING
4. EMPIRICAL SOFTWARE ENGINEERING
5. SOFTWARE QUALITY JOURNAL

The topic clustering stage generated interpretable clusters with representative terms such as software, artificial intelligence, theory and methods, architecture, hardware, cloud computing, telecommunications, networks, and wireless systems.

The project is evaluated with functional checks:

- Database loading returns article records with journals and abstracts.
- Text preprocessing removes HTML tags and produces searchable plain text.
- Recommendation returns five journals for a sufficiently long abstract.
- Scores are sorted in descending order.
- Clustering produces topic groups with representative terms and subjects.
- The Jupyter Notebook executes from start to finish and stores output cells.
- The Streamlit interface is available as the project software component.

## VII. Conclusion

The project demonstrates a practical journal finder tailored to computer science subject areas. TF-IDF and cosine similarity provide a transparent baseline for journal recommendation, while KMeans clustering gives an overview of major topics in the dataset. Future work can compare this baseline with transformer embeddings, supervised journal classification, and citation-aware ranking.

## References

[1] G. Salton and C. Buckley, "Term-weighting approaches in automatic text retrieval," Information Processing & Management, 1988.

[2] J. MacQueen, "Some methods for classification and analysis of multivariate observations," Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability, 1967.

[3] F. Sebastiani, "Machine learning in automated text categorization," ACM Computing Surveys, 2002.
