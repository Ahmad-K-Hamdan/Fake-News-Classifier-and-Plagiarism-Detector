# Fake News Classifier & Plagiarism Detector

A user-friendly Streamlit web app for detecting fake news and possible plagiarism in news articles. The application lets you interactively compare basic and advanced machine learning pipelines for fake-news classification, explore NLP preprocessing steps, and find content in your dataset that is similar to a given article.

---

## 📋 Project Overview

**Fake News Classifier & Plagiarism Detector** demonstrates two different approaches to fake news detection:

1. **Baseline Pipeline**: Uses TF-IDF on lemmatized text, with Decision Tree and Logistic Regression classifiers.
2. **Enriched Pipeline**: Adds character-level n-grams, readability metrics, POS/NER counts, sentiment, clickbait heuristics, and topic modeling features to the baseline, and applies the same classifiers.

Additionally, the app provides a **Plagiarism Detector** that finds news articles similar to a given input using TF-IDF and nearest neighbor search.

### Key Features

- **Train or load** all four models at once, side-by-side.
- **Inspect** dataset size, feature-matrix shapes, performance metrics, and confusion matrices.
- **Explore** NLP preprocessing steps on a sample article.
- **Experiment** by entering custom text to see how each model processes and classifies it.
- **Detect plagiarism** by finding the most similar articles in the dataset for a given input.

---

## 🔧 Installation

1. **Clone** this repository:
    ```bash
    git clone https://github.com/Ahmad-K-Hamdan/Fake-News-Classifier-and-Plagiarism-Detector.git
    cd Fake-News-Classifier-and-Plagiarism-Detector
    ```

2. **Create & activate** a Python virtual environment (recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate        # macOS/Linux
    venv\Scripts\activate           # Windows
    ```

3. **Install** dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. **Download NLP models & data (one-time):**
    ```bash
    python -m nltk.downloader punkt wordnet stopwords vader_lexicon averaged_perceptron_tagger
    python -m spacy download en_core_web_sm
    ```

---

## 🚀 Usage

Start the Streamlit app:

```bash
streamlit run app.py
```

Navigate through the sidebar to access:

1. **Train & Evaluate Models**: Train or load all classifiers, view metrics and confusion matrices.
2. **NLP Demo**: Step-by-step preprocessing on a sample article with feature explanations.
3. **Predict**: Enter custom text to see classification and NLP steps in real time.
4. **Plagiarism Similarity Search**: Paste an article to check for close matches in the dataset.

---

## ⚙️ How It Works

- **Baseline Pipeline**: TF-IDF on lemmatized text (word n-grams).
- **Enriched Pipeline**: Adds character n-grams, readability, POS/NER, sentiment, clickbait, and topic modeling features.
- **Plagiarism Detector**: Uses TF-IDF vectorization and nearest neighbors search to find similar articles.

All models are trained and evaluated interactively, and you can inspect and compare their performance.

---

*Happy fake-news hunting!*
