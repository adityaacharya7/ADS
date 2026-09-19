# Applied Data Science (ADS): Twitter Customer Support Emotion & Sentiment Analysis

An end-to-end data science, machine learning, and natural language processing toolkit for analyzing customer support dynamics on Twitter. The repository is organized modularly by **Experiment Numbers**, covering large-scale data preprocessing, statistical hypothesis testing, multi-model ML benchmarking, hyperparameter optimization, MLflow tracking, and academic report compilation.

---

## 📁 Repository Structure (Organized by Experiment Numbers)

```text
ADS/
├── .gitignore                         # Git rules for datasets, checkpoints & runs
├── README.md                          # Comprehensive project documentation & usage guide
├── requirements.txt                   # Standardized Python dependencies
├── main.py                            # Unified master CLI supporting experiment subcommands
│
├── data/                              # Shared dataset repository
│   ├── raw/
│   │   └── twcs.csv                   # Original Twitter Customer Support dataset (~516MB)
│   └── processed/
│       ├── twcs_cleaned.csv           # Preprocessed & filtered dataset (100,000 samples, ~28MB)
│       ├── twcs_sample_5k.csv         # Rapid prototyping sample (5,000 samples)
│       └── twcs_sample_20k.csv        # Balanced validation sample (20,000 samples)
│
├── experiments/
│   ├── experiment_2/                  # Negation-Aware Multi-Label Emotion Classification
│   │   ├── src/                       # Modular source code
│   │   │   ├── __init__.py
│   │   │   ├── preprocessing.py       # Cleaning, entity unescaping & negation scope tagging
│   │   │   ├── emotion_labeler.py     # VADER polarity & rule-based emotion labeling engine
│   │   │   ├── train.py               # Baseline & multi-label classifier training
│   │   │   └── predict.py             # CLI & interactive emotion prediction interface
│   │   ├── models/                    # Model artifacts and metadata
│   │   │   ├── emotion_pipeline.joblib# Serialized TF-IDF + classifier pipeline
│   │   │   └── model_metadata.json    # Benchmark metrics and training configurations
│   │   ├── plots/                     # Evaluation charts & confusion matrices
│   │   │   ├── confusion_matrix.png
│   │   │   ├── emotion_distribution.png
│   │   │   └── model_comparison.png
│   │   └── reports/                   # Detailed experiment writeup & metrics
│   │       └── experiment_2_report.txt
│   │
│   ├── experiment_3/                  # Statistical EDA, Distribution Fitting & Hypothesis Testing
│   │   ├── assignment/                # Curriculum assignment question sheet
│   │   │   └── Experiment 3.pdf
│   │   ├── src/                       # Statistical analysis and PDF generation code
│   │   │   ├── __init__.py
│   │   │   ├── eda_analysis.py        # Normality tests, ANOVA, Kruskal-Wallis & Chi-square
│   │   │   └── generate_report.py     # Automated ReportLab academic PDF generator
│   │   ├── notebooks/                 # Interactive Jupyter & Colab notebooks
│   │   │   ├── experiment_3_eda.ipynb # Comprehensive local EDA notebook
│   │   │   └── experiment_3_colab.ipynb# Self-contained Google Colab notebook
│   │   ├── models/                    # Statistical summaries
│   │   │   └── exp3_eda_summary.json  # Distribution parameters, KS test statistics & p-values
│   │   ├── plots/                     # 6 publication-ready analytical charts
│   │   │   ├── exp3_boxplots_spread.png
│   │   │   ├── exp3_class_balance.png
│   │   │   ├── exp3_correlation_heatmap.png
│   │   │   ├── exp3_distribution_fitting_outliers.png
│   │   │   ├── exp3_feature_distributions.png
│   │   │   └── exp3_hypothesis_tests.png
│   │   └── reports/                   # Academic writeups, LaTeX source & compiled documents
│   │       ├── Experiment_3_Report.pdf# Academic PDF report
│   │       ├── Experiment_3_Report.docx# Academic Word document
│   │       ├── Experiment_3_Report.tex# LaTeX source code
│   │       ├── Experiment_3_Overleaf_Package.zip# Complete Overleaf package
│   │       ├── EXPERIMENT_3_WALKTHROUGH_AND_VIVA_GUIDE.txt # Oral viva & interview guide
│   │       └── experiment_3_report.txt
│   │
│   └── experiment_4/                  # ML Modeling, Hyperparameter Tuning & MLflow Tracking
│       ├── assignment/                # Curriculum assignment question sheet
│       │   └── Experiment 4.pdf
│       ├── src/                       # Benchmark modeling and report compilation code
│       │   ├── __init__.py
│       │   ├── experiment_4_modeling.py# 5 baseline models, GridSearchCV & MLflow tracking
│       │   ├── generate_experiment_4_report.py # LaTeX, PDF & Overleaf packager
│       │   └── generate_experiment_4_docx.py   # Academic Word report generator
│       ├── notebooks/                 # Interactive notebooks
│       │   ├── experiment_4_colab.ipynb# Self-contained Google Colab notebook
│       │   └── experiment_4_mlflow.ipynb# Local MLflow inspection notebook
│       ├── models/                    # Champion model artifacts
│       │   ├── best_emotion_model_exp4.joblib # Serialized LightGBM champion pipeline
│       │   └── exp4_model_summary.json# Complete benchmark scores and hyperparameters
│       ├── mlruns/                    # Local MLflow experiment tracking database & runs
│       ├── plots/                     # Visual diagnostic plots & dashboard captures
│       │   ├── exp4_confusion_matrices.png
│       │   ├── exp4_mlflow_dashboard.png
│       │   ├── exp4_model_comparison.png
│       │   └── exp4_tuning_comparison.png
│       └── reports/                   # Benchmark CSVs, classification reports & formal writeups
│           ├── Experiment_4_Report.pdf# Academic PDF report
│           ├── Experiment_4_Report.docx# Academic Word document
│           ├── Experiment_4_Report.tex# LaTeX source code
│           ├── Experiment_4_Overleaf_Package.zip# Complete Overleaf bundle
│           ├── EXPERIMENT_4_WALKTHROUGH_AND_VIVA_GUIDE.txt # Oral viva & interview guide
│           ├── model_benchmark_results.csv # Quantitative model comparison table
│           └── clf_report_baseline_*.txt   # Per-model classification reports
│
│   ├── experiment_5/                  # Explainable AI (SHAP & LIME) & Fairness Auditing (Fairlearn)
│   │   ├── assignment/                # Curriculum assignment question sheet
│   │   │   └── Experiment 5.pdf
│   │   ├── src/                       # XAI, audit and report compilation code
│   │   │   ├── __init__.py
│   │   │   ├── experiment_5_xai_fairness.py # SHAP, LIME, Fairlearn audit & tri-modal mitigation
│   │   │   ├── generate_experiment_5_report.py # LaTeX, PDF & Overleaf packager
│   │   │   └── generate_experiment_5_docx.py   # Academic Word report generator
│   │   ├── notebooks/                 # Interactive notebooks
│   │   │   ├── experiment_5_xai_fairness.ipynb # Local Jupyter notebook
│   │   │   └── experiment_5_colab.ipynb        # Self-contained Google Colab notebook
│   │   ├── models/                    # Serialized models & summaries
│   │   │   ├── unmitigated_champion_model.joblib
│   │   │   ├── mitigated_reweighted_model.joblib
│   │   │   ├── mitigated_exponentiated_model.joblib
│   │   │   ├── mitigated_threshold_model.joblib
│   │   │   └── exp5_xai_fairness_summary.json
│   │   ├── plots/                     # High-resolution publication plots
│   │   │   ├── exp5_shap_summary_bar.png
│   │   │   ├── exp5_shap_beeswarm.png
│   │   │   ├── exp5_shap_dependence.png
│   │   │   ├── exp5_shap_waterfall.png
│   │   │   ├── exp5_lime_local_explanations.png
│   │   │   ├── exp5_xai_shap_vs_lime_comparison.png
│   │   │   ├── exp5_fairness_audit_disparity.png
│   │   │   └── exp5_fairness_mitigation_tradeoff.png
│   │   └── reports/                   # Academic reports & benchmark tables
│   │       ├── Experiment_5_Report.pdf
│   │       ├── Experiment_5_Report.docx
│   │       ├── Experiment_5_Report.tex
│   │       ├── Experiment_5_Overleaf_Package.zip
│   │       ├── EXPERIMENT_5_WALKTHROUGH_AND_VIVA_GUIDE.txt
│   │       ├── bias_mitigation_benchmark.csv
│   │       └── fairness_audit_metrics.csv
│   │
│   └── experiment_6/                  # Containerization & API Deployment (FastAPI & Docker)
│       ├── assignment/                # Curriculum assignment question sheet
│       │   └── Experiment 6.pdf
│       ├── Dockerfile                 # Hardened multi-stage non-root container image
│       ├── docker-compose.yml         # Container orchestration with healthchecks & resource limits
│       ├── .dockerignore              # Docker build context optimization rules
│       ├── src/                       # Microservice application & automated test code
│       │   ├── __init__.py
│       │   ├── app.py                 # FastAPI REST API with lifespan caching, CORS & Pydantic
│       │   ├── test_api.py            # Automated in-process endpoint verification & latency benchmarking
│       │   ├── generate_experiment_6_report.py # LaTeX, PDF & Overleaf packager
│       │   └── generate_experiment_6_docx.py   # Academic Word report generator
│       ├── notebooks/                 # Interactive deployment notebooks
│       │   ├── experiment_6_api_deployment.ipynb # Local API testing & Docker walkthrough notebook
│       │   └── experiment_6_colab.ipynb        # Self-contained Google Colab notebook
│       ├── plots/                     # Architectural & performance visualization figures
│       │   ├── exp6_architecture_diagram.png
│       │   └── exp6_api_latency_distribution.png
│       └── reports/                   # Academic writeups, test evidence logs & packages
│           ├── Experiment_6_Report.pdf# Academic PDF report
│           ├── Experiment_6_Report.docx# Academic Word document
│           ├── Experiment_6_Report.tex# LaTeX source code
│           ├── Experiment_6_Overleaf_Package.zip# Complete Overleaf bundle
│           ├── EXPERIMENT_6_WALKTHROUGH_AND_VIVA_GUIDE.txt # Oral viva & interview guide
│           ├── api_test_evidence.json # Automated JSON test execution logs
│           └── api_test_evidence.txt  # Human-readable test evidence log
│
├── src/                               # Backward-compatibility root shim layer
│   ├── __init__.py                    # Universal re-exports across all experiments
│   ├── preprocessing.py               # Re-exports experiments.experiment_2.src.preprocessing
│   ├── emotion_labeler.py             # Re-exports experiments.experiment_2.src.emotion_labeler
│   ├── train.py                       # Re-exports experiments.experiment_2.src.train
│   ├── predict.py                     # Re-exports experiments.experiment_2.src.predict
│   ├── eda_analysis.py                # Re-exports experiments.experiment_3.src.eda_analysis
│   ├── generate_report.py             # Re-exports experiments.experiment_3.src.generate_report
│   ├── experiment_4_modeling.py       # Re-exports experiments.experiment_4.src.experiment_4_modeling
│   ├── generate_experiment_4_report.py# Re-exports experiments.experiment_4.src.generate_experiment_4_report
│   ├── generate_experiment_4_docx.py  # Re-exports experiments.experiment_4.src.generate_experiment_4_docx
│   ├── experiment_5_xai_fairness.py   # Re-exports experiments.experiment_5.src.experiment_5_xai_fairness
│   ├── generate_experiment_5_report.py# Re-exports experiments.experiment_5.src.generate_experiment_5_report
│   ├── generate_experiment_5_docx.py  # Re-exports experiments.experiment_5.src.generate_experiment_5_docx
│   ├── app.py                         # Re-exports experiments.experiment_6.src.app
│   ├── generate_experiment_6_report.py# Re-exports experiments.experiment_6.src.generate_experiment_6_report
│   └── generate_experiment_6_docx.py  # Re-exports experiments.experiment_6.src.generate_experiment_6_docx
│
└── assets/                            # Static project visual assets
```

---

## 🚀 Quickstart & Installation

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

The unified CLI allows running pipelines either by **experiment number** or via **legacy shorthand commands**:

### 1. Experiment 2: Preprocessing, Feature Engineering & Classification

```bash
# Run prediction demonstration
python main.py exp2 demo

# Predict single tweet text
python main.py exp2 predict --text "My package was delivered damaged and 3 weeks late! Unacceptable!"

# Start interactive CLI session
python main.py exp2 predict --interactive

# Clean and downsample raw dataset
python main.py exp2 preprocess --input data/raw/twcs.csv --output data/processed/twcs_cleaned.csv --target-rows 100000

# Train emotion classification models
python main.py exp2 train --sample-size 50000
```

### 2. Experiment 3: Statistical EDA & Hypothesis Testing

```bash
# Run statistical exploratory analysis, distribution fitting & hypothesis tests
python main.py exp3 eda --sample-size 50000

# Compile publication-quality Experiment 3 academic PDF report
python main.py exp3 report
```

### 3. Experiment 4: ML Modeling, Tuning & MLflow Tracking

```bash
# Train 5 baseline models, run GridSearchCV tuning, and log runs in MLflow
python main.py exp4 train --sample-size 25000

# Generate academic PDF & LaTeX reports for Experiment 4
python main.py exp4 report

# Generate academic Word (.docx) report for Experiment 4
python main.py exp4 docx

# Launch the interactive local MLflow tracking dashboard
python main.py exp4 dashboard --port 5000
```

### 4. Experiment 5: Explainable AI (SHAP & LIME) & Fairness Auditing (Fairlearn)

```bash
# Run complete XAI interpretability, fairness audit & tri-modal bias mitigation
python main.py exp5 run

# Generate publication-quality academic reports (PDF, Word .docx, LaTeX, Overleaf ZIP)
python main.py exp5 report

# Display interactive XAI and fairness summary demonstration
python main.py exp5 demo
```

### 5. Experiment 6: Containerization & API Deployment (FastAPI & Docker)

```bash
# Launch FastAPI microservice locally with hot reload
python main.py exp6 serve --host 127.0.0.1 --port 8000 --reload

# Execute full automated test suite with latency benchmark & evidence capture
python main.py exp6 test

# Build production Docker container image
python main.py exp6 docker-build

# Run Docker container with port mapping
python main.py exp6 docker-run --port 8000

# Compile publication-quality academic deliverables (PDF, Word .docx, LaTeX, Overleaf ZIP)
python main.py exp6 report
```

### 6. Direct Legacy Shortcuts (Fully Supported)

```bash
python main.py demo
python main.py predict --text "Flight delayed 8 hours!"
python main.py preprocess
python main.py train
python main.py eda
python main.py report
python main.py pipeline
```

---

## 📊 Overview of Experiments

### Experiment 2: Negation-Aware Multi-Label Emotion Classification
- **Objectives**: Clean customer support tweets, handle negation scope tagging (`_NEG`), engineer VADER polarity scores, train multi-label classifiers, and evaluate Macro/Micro F1, Hamming Loss, and Jaccard similarity.
- **Emotion Dimensions**: `Joy / Gratitude`, `Anger / Frustration`, `Disappointment / Sadness`, `Fear / Anxiety`, `Neutral / Inquiry`.

### Experiment 3: Exploratory Data Analysis & Statistical Hypothesis Testing
- **Objectives**: Quantify class distributions, fit parametric distributions (Log-Normal for word counts, Poisson for punctuation), examine correlations via heatmaps, and conduct parametric & non-parametric hypothesis tests (Two-sample *t*-test, Mann-Whitney U, One-Way ANOVA, Kruskal-Wallis, Chi-square test of independence).

### Experiment 4: ML Modeling, Hyperparameter Tuning & MLflow Tracking
- **Objectives**: Train 5 diverse model families (Logistic Regression, LightGBM, Linear SVM, Multinomial Naive Bayes, Random Forest), perform cross-validated hyperparameter tuning via `GridSearchCV`, log all runs, metrics, parameters, and artifacts in **MLflow**, and serialize the champion pipeline for deployment.

### Experiment 5: Explainable AI (SHAP & LIME) & Fairness Auditing (Fairlearn)
- **Objectives**: Apply game-theoretic **SHAP** (global importance, beeswarm, dependence interaction, waterfall) and local surrogate **LIME** (`lime.lime_tabular`) to explain complex models. Audit demographic bias across sensitive protected features (`Sex` and `Race`) using **Fairlearn**. Implement tri-modal mitigation: Pre-processing (sample reweighting), In-processing (`ExponentiatedGradient`), and Post-processing (`ThresholdOptimizer`), establishing the Fairness-Performance Pareto frontier.

### Experiment 6: Containerization & API Deployment (FastAPI & Docker)
- **Objectives**: Productionize the champion LightGBM NLP model as a high-throughput, low-latency microservice. Build asynchronous REST endpoints (`/predict`, `/predict/batch`, `/health`) with **FastAPI** and **Pydantic V2** schema validation. Package into a hardened multi-stage, non-root **Docker** container governed by `docker-compose.yml` with healthchecks and resource limits. Benchmark single-request and batch latency ($p_{50}=19.35\text{ ms}$), generate automated test evidence logs, and compile complete academic reports.

---

## 📈 Key Benchmark Results

### Experiment 4: Multi-Model Emotion Classification
| Model Family | Candidate Classifier | Accuracy | Macro F1 | Weighted F1 | Training Time |
|---|---|---|---|---|---|
| **Gradient Boosting** | **LightGBM (Champion)** | **98.53%** | **0.9538** | **0.9852** | 2.82s |
| **Linear SVM** | LinearSVC (Calibrated) | 97.45% | 0.8939 | 0.9739 | 0.58s |
| **Linear Models** | Tuned Logistic Regression | 92.73% | 0.8931 | 0.9312 | 1.95s |
| **Linear Models** | Baseline Logistic Regression | 90.57% | 0.8715 | 0.9126 | 0.76s |
| **Ensemble Trees** | Random Forest | 88.08% | 0.7344 | 0.8767 | 20.37s |
| **Probabilistic** | Multinomial Naive Bayes | 76.81% | 0.5225 | 0.7483 | 0.05s |

### Experiment 5: Algorithmic Fairness & Bias Mitigation Benchmark
| Model Pipeline | Accuracy | Balanced Accuracy | F1-Score | DP Diff (Sex) | EO Diff (Sex) | Disparate Impact Ratio |
|---|---|---|---|---|---|---|
| **1. Baseline (Unmitigated LightGBM)** | **87.69%** | **80.65%** | **0.7240** | 0.1714 | 0.0611 | 0.3464 (Violates 80% rule) |
| **2. Pre-Processing (Reweighted)** | 86.92% | 78.71% | 0.6983 | **0.0867 (-49.4%)** | 0.2029 | **0.6086 (+75.7%)** |
| **3. In-Processing (ExponentiatedGrad)** | 85.69% | 75.72% | 0.6553 | 0.0973 | 0.0656 | 0.5292 |
| **4. Post-Processing (ThresholdOptimizer)** | 85.78% | 77.14% | 0.6719 | 0.1085 | **0.0393 (-35.6%)** | 0.5253 |

### Experiment 6: Microservice Latency & Throughput Benchmark
| Benchmark Metric | Measured Result | Production Target | Assessment |
|---|---|---|---|
| **Endpoint Test Pass Rate** | **100.0% (7/7 suites)** | 100% | Flawless validation |
| **Sample Assertion Pass Rate** | **100.0% (30/30 requests)** | 100% | Flawless execution |
| **Median Latency ($p_{50}$)** | **19.35 ms** | $< 50$ ms | Exceeds SLA by 2.6x |
| **Mean Latency** | **28.86 ms** | $< 75$ ms | Highly responsive |
| **95th Percentile ($p_{95}$)** | **86.81 ms** | $< 150$ ms | Controlled tail latency |
| **99th Percentile ($p_{99}$)** | **99.82 ms** | $< 250$ ms | Predictable worst-case |
| **Single-Worker Throughput** | **~34.6 requests/sec** | $> 20$ req/s | Scalable via Uvicorn workers |
| **Container Memory Footprint**| **~142 MB** | $< 512$ MB | Lightweight container runtime |

---

## 🛠️ Tech Stack & Dependencies

- **Web Microservice & API**: `fastapi`, `uvicorn`, `pydantic`, `httpx`
- **Containerization & Orchestration**: `Docker`, `docker-compose`
- **Data Processing**: `pandas`, `numpy`, `tqdm`
- **Statistical Computing**: `scipy`, `statsmodels`
- **NLP & Sentiment**: `vaderSentiment`, `nltk`
- **Machine Learning**: `scikit-learn`, `lightgbm`, `xgboost`, `joblib`
- **Explainable AI (XAI)**: `shap`, `lime`
- **Fairness & Bias Auditing**: `fairlearn`
- **Experiment Tracking**: `mlflow`
- **Reporting & Documents**: `reportlab`, `pypdf`, `python-docx`
- **Interactive**: `jupyter`, `notebook`


