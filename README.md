<div align="center">

# 🎬 IMDB Sentiment Analysis
### End-to-End NLP Pipeline for Movie Review Classification

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-NLP-4A90D9?style=for-the-badge)
![Gradio](https://img.shields.io/badge/Gradio-Deploy-FF7C00?style=for-the-badge&logo=gradio&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

<br/>

> **Classify movie reviews as Positive or Negative using a complete NLP preprocessing pipeline, TF-IDF vectorization, and Logistic Regression — deployed with a live Gradio web interface.**

<br/>

![Accuracy](https://img.shields.io/badge/Accuracy-~89%25-brightgreen?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-50K%20Reviews-blue?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

</div>

---

## 📖 Table of Contents

- [About the Project](#-about-the-project)
- [Dataset](#-dataset)
- [Pipeline Overview](#-pipeline-overview)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Installation & Setup](#-installation--setup)
- [Running the Gradio App](#-running-the-gradio-app)
- [Model Details](#-model-details)
- [Results & Evaluation](#-results--evaluation)
- [Example Predictions](#-example-predictions)
- [EDA Highlights](#-eda-highlights)
- [Key Design Decisions](#-key-design-decisions)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🧩 About the Project

This project implements a **full NLP classification pipeline** from raw text to a deployed web application. It takes 50,000 IMDB movie reviews and determines whether each one expresses a **positive** or **negative** sentiment.

What makes this project stand out:
- 🔧 Carefully engineered text cleaning that **preserves negation words** (`not`, `never`, `no`) to avoid flipping the meaning of reviews
- 📊 Rich **Exploratory Data Analysis** with visualizations before and after preprocessing
- 🧠 A clean, step-by-step notebook designed to be easy to read and learn from
- 🚀 A fully deployed **Gradio web app** so anyone can try it without writing code

---

## 📂 Dataset

| Property | Value |
|----------|-------|
| Source | IMDB Movie Reviews Dataset |
| Total Reviews | 50,000 |
| Positive Reviews | 25,000 |
| Negative Reviews | 25,000 |
| Balance | Perfectly balanced (50/50) |
| Language | English |

The dataset is loaded from Google Drive in the notebook and contains one review per row with a `sentiment` column (`positive` / `negative`).

---

## 🔄 Pipeline Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      RAW REVIEW TEXT                        │
│  "This movie was <br/>absolutely amazing! 10/10 loved it"   │
└────────────────────────┬────────────────────────────────────┘
                         │
              ┌──────────▼──────────┐
              │  Step 4: Clean Text │  Remove HTML, lowercase,
              │                     │  strip URLs, numbers,
              └──────────┬──────────┘  punctuation
                         │
              ┌──────────▼──────────┐
              │  Step 5: Tokenize   │  Split into individual
              │                     │  word tokens
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │ Step 6: Stopwords   │  Remove common words,
              │                     │  KEEP negation words
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │ Step 7: Lemmatize   │  Reduce to base form
              │                     │  (running → run)
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │  Step 12: TF-IDF    │  Convert text to numeric
              │  Vectorization      │  feature matrix
              │  5000 features,     │
              │  1-2 ngrams         │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │ Step 14: Logistic   │  Train binary classifier
              │ Regression          │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │     PREDICTION      │  Positive ✅ / Negative ❌
              └─────────────────────┘
```

---

## 🗂️ Project Structure

```
📦 IMDB-Sentiment-Analysis
│
├── 📓 NLP_Project.ipynb           # Main notebook — all 17 steps
│
├── 📊 Data
│   └── IMDB_Preprocessed.csv      # Cleaned & preprocessed dataset (output)
│
├── 🤖 Models
│   ├── sentiment_model.pkl         # Trained Logistic Regression model
│   └── tfidf_vectorizer.pkl        # Fitted TF-IDF vectorizer
│
├── 📈 Visualizations
│   ├── token_distribution.png      # EDA: token count distribution
│   └── top_words.png               # EDA: top 20 words per sentiment class
│
├── 📄 docs/                        # Project documentation
│
├── requirements.txt                # Python dependencies
├── LICENSE
└── README.md
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| **Python 3.8+** | Core language |
| **NLTK** | Tokenization, stopwords, lemmatization |
| **scikit-learn** | TF-IDF vectorization, Logistic Regression, evaluation |
| **pandas / numpy** | Data manipulation |
| **matplotlib / seaborn** | EDA visualizations |
| **Gradio** | Web interface deployment |
| **joblib** | Model serialization |
| **Google Colab** | Development environment |

---

## ⚙️ Installation & Setup

### Option A — Google Colab (Recommended)

Simply open the notebook in Google Colab and run all cells. Mount your Google Drive and place the IMDB CSV file there.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

### Option B — Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/IMDB-Sentiment-Analysis.git
cd IMDB-Sentiment-Analysis
```

**2. Create a virtual environment (optional but recommended)**
```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Download NLTK data**
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

**5. Update the dataset path** in Step 2 of the notebook to your local IMDB CSV file, then:
```bash
jupyter notebook NLP_Project.ipynb
```

---

## 📦 Requirements

```txt
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
nltk>=3.7
matplotlib>=3.4.0
seaborn>=0.11.0
gradio>=3.0.0
joblib>=1.1.0
```

---

## 🚀 Running the Gradio App

After training is complete, the last cell of the notebook launches a live web interface:

```python
interface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=5, placeholder="Enter movie review here..."),
    outputs="text",
    title="🎬 IMDB Sentiment Analysis",
    description="Write your review and AI will predict the sentiment."
)

interface.launch(share=True)  # Generates a public shareable link
```

A **public URL** will be printed in the output — open it in any browser, type a review, and get an instant prediction. No code required!

---

## 🧠 Model Details

### Feature Extraction
```
TF-IDF Vectorizer
  ├── max_features : 5,000
  ├── ngram_range  : (1, 2)   ← captures unigrams + bigrams
  └── Input        : processed_text column
```

### Classifier
```
Logistic Regression
  ├── max_iter     : 1,000
  ├── solver       : lbfgs (default)
  └── Output       : Binary (0 = Negative, 1 = Positive)
```

### Why Logistic Regression?
- Fast to train on large text datasets
- Highly interpretable — you can inspect which words drive predictions
- Strong baseline that is hard to beat on balanced classification tasks
- Works excellently with TF-IDF sparse matrices

---

## 📊 Results & Evaluation

```
══════════════════════════════════════════════
              MODEL PERFORMANCE
══════════════════════════════════════════════
  Accuracy            :  ~89%
  Training Samples    :  40,000
  Testing Samples     :  10,000
══════════════════════════════════════════════

  Classification Report:
  ┌────────────┬───────────┬────────┬──────────┐
  │  Class     │ Precision │ Recall │ F1-Score │
  ├────────────┼───────────┼────────┼──────────┤
  │  Negative  │   0.89    │  0.89  │   0.89   │
  │  Positive  │   0.89    │  0.89  │   0.89   │
  └────────────┴───────────┴────────┴──────────┘
══════════════════════════════════════════════
```

A **Confusion Matrix Heatmap** is also generated in Step 17 of the notebook for a visual breakdown of True Positives, True Negatives, False Positives, and False Negatives.

---

## 🔍 Example Predictions

```python
# Positive examples
predict_sentiment("I absolutely loved this movie! The acting was superb.")
# → Positive Review ✅

predict_sentiment("One of the best films I've ever seen. Truly breathtaking.")
# → Positive Review ✅

# Negative examples
predict_sentiment("This was the worst film I've ever seen. Total waste of time.")
# → Negative Review ❌

predict_sentiment("The plot made no sense and the acting was terrible.")
# → Negative Review ❌

# Negation handling — why we keep 'not', 'never'
predict_sentiment("This movie was not good at all.")
# → Negative Review ❌  ✓ Correctly understands negation
```

---

## 📈 EDA Highlights

After preprocessing, the notebook generates two key visualizations:

**1. Token Count Distribution** (`token_distribution.png`)
- Histogram of token counts across all reviews after cleaning
- Red dashed line showing the median token count
- Side-by-side view split by sentiment class

**2. Top 20 Words per Sentiment Class** (`top_words.png`)
- Horizontal bar charts for positive vs. negative reviews
- Reveals which words are the strongest signals for each sentiment
- Helps understand what the model actually learns

---

## 🔑 Key Design Decisions

### Preserving Negation Words
Standard NLP pipelines remove all stopwords. This project **keeps negation words** (`not`, `no`, `never`, `neither`, `nor`, `nobody`, `nothing`, `nowhere`) because removing them can completely flip the meaning:

```
Without this fix:
  "This movie was not good"  →  remove 'not'  →  "movie good"  ❌ Wrong!

With this fix:
  "This movie was not good"  →  keep 'not'    →  "movie not good" ✅ Correct
```

### Lemmatization over Stemming
The project uses **Lemmatization** (WordNetLemmatizer) instead of Stemming (PorterStemmer):

| Method | Example | Result |
|--------|---------|--------|
| Stemming | `running` | `runn` (not a real word) |
| Lemmatization | `running` | `run` (real dictionary word) |

Lemmatization produces cleaner, more meaningful features for the classifier.

---

## 🤝 Contributing

Contributions are welcome! Here are some ideas to extend this project:

- [ ] Try other classifiers (SVM, Random Forest, Naive Bayes)
- [ ] Add Word2Vec or GloVe embeddings
- [ ] Build a deep learning version (LSTM, Transformer)
- [ ] Deploy to Hugging Face Spaces permanently
- [ ] Add support for multilingual reviews

To contribute:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---


<div align="center">

Made with ❤️ using Python & NLP

⭐ **If you found this project helpful, please give it a star!** ⭐

</div>
