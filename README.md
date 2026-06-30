# SpamurAI — SMS Spam Detector

**SAIA2163 Natural Language Processing — Semester II, Session 2025/2026**
Universiti Teknologi Malaysia

An NLP pipeline and Streamlit web application that classifies SMS messages as spam or legitimate (ham), built on the UCI SMS Spam Collection dataset.

## Team

| No. | Name | Matric No. |
|---|---|---|
| 1 | Nur Asyura binti Mohd Anwar | A24AI0075 |
| 2 | Md Wasiful Hoque | A24AI4006 |
| 3 | Wafa Wan | A24AI4003 |
| 4 | Siti Noor Balqis binti Yuslimi | A24AI0090 |

Lecturer: Dr Azizul bin Azizan

## Overview

The pipeline covers text preprocessing, two feature extraction methods (TF-IDF and Word2Vec), three trained classifiers, full evaluation, and a Streamlit interface for live predictions. Full methodology, results, and visualizations are documented in the project report.

| Model | Features | Accuracy | F1-Score |
|---|---|---|---|
| **Logistic Regression (deployed)** | TF-IDF | 98.2% | 0.981 |
| Naive Bayes | TF-IDF | 96.1% | 0.958 |
| Logistic Regression | Word2Vec | 92.2% | 0.914 |

Logistic Regression on TF-IDF features was selected as the final model based on F1-score and is the model loaded by the web app.

## Project Structure

```
.
├── app.py                       # Streamlit application (entry point)
├── visualizations.py            # Chart generation scripts
├── utils.py                     # Shared text preprocessing module
├── data/
│   ├── spam.csv                 # Raw dataset
│   └── cleaned_data.csv         # Preprocessed dataset
├── models/
│   ├── best_model.pkl           # Deployed model (Logistic Regression, TF-IDF)
│   ├── tfidf_vectorizer.pkl
│   ├── word2vec_vectors.bin
│   ├── model_comparison_metrics.csv
│   ├── confusion_matrices.npz
│   └── feature_weights.csv
├── notebooks/
│   └── spam_detection.ipynb     # Full training & evaluation notebook

```

## Setup Instructions

**1. Clone the repository and install dependencies**

```bash
git clone <repository-url>
cd <repository-folder>
pip install streamlit scikit-learn pandas numpy joblib nltk gensim
```

**2. Place model files in the working directory**

`app.py` currently loads its model and data files (`tfidf_vectorizer.pkl`, `best_model.pkl`, `model_comparison_metrics.csv`, `confusion_matrices.npz`) directly from the folder it's run from — **not** from `models/`. Until the app's file paths are updated to point into `models/`, run it from inside that folder, or copy those files up to the repository root before launching.

**3. Run the app**

```bash
streamlit run app.py
```

## Known Limitations of the Current Build

- **Naive Bayes predictions are not yet live in the app.** The notebook only exports the single best model (`best_model.pkl`, Logistic Regression). No `naive_bayes_model.pkl` is currently produced, so selecting "Naive Bayes (TF-IDF)" in the app falls back to a keyword-based estimate rather than a real model prediction.
- **Word2Vec predictions are not yet live in the app.** No Word2Vec-based classifier is currently loaded by `app.py`; selecting that option also falls back to a keyword-based estimate.
- Only the **Logistic Regression (TF-IDF)** option produces a genuine model-based prediction, and only when `tfidf_vectorizer.pkl` and `best_model.pkl` are both present and successfully loaded.
- If model files are missing entirely, the app still runs using heuristic keyword matching and randomized scoring, so it remains demonstrable without trained model files present — but predictions in that state are not real.

## Application Features

- **Home:** Paste a message, select a model, and get a spam/ham prediction with a confidence score.
- **Intel page:** Per-model accuracy, precision, recall, F1-score, and confusion matrix breakdown.
- **Result page:** Predicted class, confidence gauge, flagged trigger words, and message metadata.

## Visualizations

Six visualizations are generated from the dataset and model outputs: word clouds by class, class distribution, confusion matrices for all three models, a grouped model comparison chart, top-20 frequent words per class, and text length distribution by class. See `report/` for full figures and discussion.

## References

1. A. Almeida and T. Hidalgo, "UCI SMS Spam Collection Dataset," UCI Machine Learning Repository, 2012.
2. T. Mikolov et al., "Efficient Estimation of Word Representations in Vector Space," arXiv:1301.3781, 2013.
3. A. Mueller, "WordCloud for Python Documentation," 2020.
4. Streamlit Inc., "A faster way to build and share data apps," 2023.
5. SMS Spam Collection Dataset, Kaggle.
