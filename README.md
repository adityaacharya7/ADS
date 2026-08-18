# Applied Data Science (ADS): Twitter Customer Support Emotion & Sentiment Analysis

An end-to-end data science and natural language processing toolkit for analyzing customer support dynamics on Twitter. This project implements large-scale data preprocessing, lexicon-augmented emotion tagging, machine learning classification, rigorous statistical exploratory data analysis (EDA), parametric & non-parametric hypothesis testing, and automated academic report generation.

---

## 📁 Repository Structure

```text
ADS/
├── .gitignore                    # Git rules for datasets, environments, and checkpoints
├── README.md                     # Comprehensive project documentation & usage guide
├── requirements.txt              # Standardized Python dependencies
├── main.py                       # Unified master CLI entry point
│
├── data/                         # Datasets directory
│   ├── raw/
│   │   └── twcs.csv              # Original raw Twitter Customer Support dataset (~516MB)
│   └── processed/
│       └── twcs_cleaned.csv      # Preprocessed and filtered dataset (100,000 samples, ~28MB)
│
├── src/                          # Modular Python source code
│   ├── __init__.py               # Package initializer exposing core classes & functions
│   ├── preprocessing.py          # Ingestion, text cleaning, entity unescaping & deduplication
│   ├── emotion_labeler.py        # VADER polarity & rule-based emotion labeling engine
│   ├── train.py                  # Model training & benchmarking (Logistic Regression, LightGBM, XGBoost)
│   ├── predict.py                # Real-time CLI & interactive emotion prediction interface
│   ├── eda_analysis.py           # Statistical EDA, distribution fitting & hypothesis testing suite
│   └── generate_report.py        # Automated academic PDF report generator (ReportLab)
│
├── notebooks/                    # Interactive Jupyter notebooks
│   └── experiment_3_eda.ipynb    # Comprehensive Experiment 3 EDA notebook
│
├── models/                       # Model artifacts and metadata
│   ├── emotion_pipeline.joblib   # Serialized feature extraction + best classifier pipeline
│   ├── model_metadata.json       # Training configurations, benchmark scores & class mapping
│   └── exp3_eda_summary.json     # Statistical test results, distribution parameters & metrics
│
├── plots/                        # Generated figures and charts
│   ├── confusion_matrix.png
│   ├── emotion_distribution.png
│   ├── exp3_boxplots_spread.png
│   ├── exp3_class_balance.png
│   ├── exp3_correlation_heatmap.png
│   ├── exp3_distribution_fitting_outliers.png
│   ├── exp3_feature_distributions.png
│   ├── exp3_hypothesis_tests.png
│   └── model_comparison.png
│
├── reports/                      # Academic writeups, assignments & compiled reports
│   ├── assignments/
│   │   └── Experiment 3.pdf      # Original assignment question sheet
│   ├── experiment_2/
│   │   └── experiment_2_report.txt # Experiment 2 detailed report
│   └── experiment_3/
│       ├── Experiment_3_Report.pdf # Formatted ReportLab academic report
│       ├── Experiment_3_Report.tex # LaTeX report source
│       └── experiment_3_report.txt # Full text writeup and statistical analysis
│
└── assets/                       # Images and static project media
    └── image.jpg
```

---

## 🚀 Quickstart & Installation

### 1. Environment Setup

Clone the repository and install required dependencies:

```bash
# Clone repository
git clone https://github.com/adityaacharya7/ADS.git
cd ADS

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate       # On Linux/macOS
.venv\Scripts\activate          # On Windows

# Install dependencies
pip install -r requirements.txt
```

---

## ⚡ Master CLI Commands (`main.py`)

The unified CLI provides instant access to all pipeline stages:

### 1. Real-Time Emotion Prediction

```bash
# Run prediction demonstration on sample customer tweets
python main.py demo

# Predict single tweet text
python main.py predict --text "My package was delivered damaged and 3 weeks late! Unacceptable!"

# Start interactive CLI session
python main.py predict --interactive
```

### 2. Preprocess Raw Data (Experiment 2)

```bash
python main.py preprocess --input data/raw/twcs.csv --output data/processed/twcs_cleaned.csv --target-rows 100000
```

### 3. Train Machine Learning Models

```bash
python main.py train --input data/processed/twcs_cleaned.csv --sample-size 50000
```

### 4. Run Statistical EDA & Hypothesis Testing (Experiment 3)

```bash
python main.py eda --input data/processed/twcs_cleaned.csv --sample-size 50000
```

### 5. Generate Academic PDF Report

```bash
python main.py report --output reports/experiment_3/Experiment_3_Report.pdf
```

### 6. Run Complete Pipeline

```bash
python main.py pipeline --sample-size 50000
```

---

## 📊 Overview of Experiments

### Experiment 2: Negation-Aware Multi-Label Emotion Classification Architecture

- **Objectives**: Filter and clean unstructured customer support tweets, remove bots/noise, create balanced subsets, extract (1, 3) N-Grams with negation scope tagging (`_NEG`), engineer VADER polarity features, train multi-label classifiers (`OneVsRestClassifier(LogisticRegression)` and `OneVsRestClassifier(LGBMClassifier)`), compute multi-label metrics (Macro F1, Hamming Loss, Jaccard Similarity), and serialize the optimal pipeline.
- **Emotion Dimensions (Multi-Label Profile)**:
  1. `Joy / Gratitude`
  2. `Anger / Frustration`
  3. `Disappointment / Sadness`
  4. `Fear / Anxiety`
  5. `Neutral / Inquiry`

---

## 📈 Key Metrics & Architecture Benchmark

| Architecture | Candidate Model | Macro F1 | Micro F1 | Hamming Loss | Jaccard Score |
|---|---|---|---|---|---|
| **Baseline Single-Label** | Single-Label Logistic Regression | 0.812 | 0.820 | N/A (Single) | 0.742 |
| **Negation Multi-Label (Production)** | **Multi-Label Logistic Regression** | **0.847** | **0.859** | **0.064** | **0.838** |
| **Negation Multi-Label (Ensemble)** | Multi-Label LightGBM | 0.931 | 0.966 | 0.015 | 0.962 |

---

## 💡 Negation & Mixed Emotion Test Verification

```text
Sentence: "I am disappointed with this service."
  → Disappointment: 100.0% | Anger: 92.9% | Joy: 1.9%

Sentence: "I am not disappointed with this service."
  → Joy: 95.8% | Neutral: 18.5% | Disappointment: 3.6% | Anger: 0.3%

Sentence: "I am not angry at all, just asking for a status update."
  → Neutral: 64.3% | Joy: 38.5% | Anger: 0.3%

Sentence: "I found myself unexpectedly proud of the milestone, although I remained anxious about the release."
  → Fear / Anxiety: 95.1% | Joy / Gratitude: 74.7% (Mixed Emotion Detected)
```

---

## 🛠️ Tech Stack & Dependencies

- **Data Processing**: `pandas`, `numpy`, `tqdm`
- **Statistical Computing**: `scipy`, `statsmodels`
- **NLP & Sentiment**: `vaderSentiment`, `nltk`
- **Machine Learning**: `scikit-learn`, `lightgbm`, `xgboost`, `joblib`
- **Visualization**: `matplotlib`, `seaborn`
- **Reporting**: `reportlab`, `pypdf`
- **Interactive**: `jupyter`, `notebook`
