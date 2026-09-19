"""
Experiment 4: Machine Learning Modeling & Experiment Tracking with MLflow
Course: Applied Data Science (ADS)
Dataset: Customer Support on Twitter (TWCS) Cleaned Corpus

Implements the complete ML lifecycle:
1. Dataset Preparation & 80/20 Stratified Split
2. Baseline Model Training (5 Diverse Architectures)
3. Systematic Hyperparameter Tuning with GridSearchCV
4. Experiment Tracking with MLflow (Runs, Params, Metrics, Artifacts)
5. Model Selection & Production Pipeline Serialization
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, classification_report, confusion_matrix
)
import joblib

from lightgbm import LGBMClassifier
import mlflow
import mlflow.sklearn

try:
    from experiments.experiment_2.src.preprocessing import clean_tweet_text, apply_negation_tagging
    from experiments.experiment_2.src.emotion_labeler import EmotionLabeler, EMOTION_CLASSES
except ImportError:
    from src.preprocessing import clean_tweet_text, apply_negation_tagging
    from src.emotion_labeler import EmotionLabeler, EMOTION_CLASSES

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = EXPERIMENT_DIR.parent.parent
PROJECT_ROOT = EXPERIMENT_DIR


class EndToEndEmotionPipeline:
    """
    Production-ready end-to-end inference pipeline bundling
    text preprocessing, TF-IDF vectorization, feature scaling,
    and the champion machine learning classifier.
    """
    def __init__(self, vectorizer, scaler, classifier, emotion_classes):
        self.vectorizer = vectorizer
        self.scaler = scaler
        self.classifier = classifier
        self.emotion_classes = emotion_classes

    def predict(self, texts: List[str], vader_features: np.ndarray) -> np.ndarray:
        negated_texts = [apply_negation_tagging(clean_tweet_text(t)) for t in texts]
        X_tfidf = self.vectorizer.transform(negated_texts)
        X_vader_scaled = self.scaler.transform(vader_features)
        X_combined = sp.hstack([X_tfidf, X_vader_scaled], format='csr')
        return self.classifier.predict(X_combined)

    def predict_proba(self, texts: List[str], vader_features: np.ndarray) -> np.ndarray:
        negated_texts = [apply_negation_tagging(clean_tweet_text(t)) for t in texts]
        X_tfidf = self.vectorizer.transform(negated_texts)
        X_vader_scaled = self.scaler.transform(vader_features)
        X_combined = sp.hstack([X_tfidf, X_vader_scaled], format='csr')
        
        if hasattr(self.classifier, "predict_proba"):
            return self.classifier.predict_proba(X_combined)
        elif hasattr(self.classifier, "decision_function"):
            df = self.classifier.decision_function(X_combined)
            exp_df = np.exp(df - np.max(df, axis=1, keepdims=True))
            return exp_df / np.sum(exp_df, axis=1, keepdims=True)
        else:
            preds = self.classifier.predict(X_combined)
            n_classes = len(self.emotion_classes)
            probas = np.zeros((len(preds), n_classes))
            for i, p in enumerate(preds):
                probas[i, p] = 1.0
            return probas


def load_and_prepare_data(sample_size: int = 25000, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Loads raw cleaned corpus, applies negation and emotion annotation,
    and returns an 80/20 stratified train/test split.
    """
    csv_path = WORKSPACE_ROOT / "data" / "processed" / "twcs_cleaned.csv"
    if not csv_path.exists():
        csv_path = EXPERIMENT_DIR / "data" / "processed" / "twcs_cleaned.csv"
    print(f"\n[Step 1/5] Loading TWCS dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"  Total records in file: {len(df):,}")

    if 0 < sample_size < len(df):
        print(f"  Stratified/representative sampling to {sample_size:,} records for fast execution...")
        df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)

    df['clean_text'] = df['clean_text'].fillna("")
    
    print("  Applying EmotionLabeler to extract ground-truth emotion classes & VADER features...")
    labeler = EmotionLabeler()
    df_labeled = labeler.label_dataframe(df, text_column='clean_text')
    
    # Pre-apply negation scope tagging for text representations
    df_labeled['negated_text'] = df_labeled['clean_text'].apply(apply_negation_tagging)

    # Class breakdown
    class_counts = df_labeled['emotion'].value_counts()
    print("\n  Extracted Class Distribution:")
    for cls, count in class_counts.items():
        pct = (count / len(df_labeled)) * 100
        print(f"    - {cls:25s}: {count:6,} ({pct:.2f}%)")

    # 80/20 Stratified Split
    train_df, test_df = train_test_split(
        df_labeled,
        test_size=0.20,
        random_state=random_state,
        stratify=df_labeled['emotion']
    )
    print(f"\n  Data Split Complete: Train={len(train_df):,} (80%), Test={len(test_df):,} (20%)")
    return train_df, test_df


def extract_features(train_df: pd.DataFrame, test_df: pd.DataFrame, max_features: int = 10000):
    """
    Extracts hybrid feature representations:
    1. Sparse TF-IDF (1, 2) n-grams on negation-tagged text
    2. Dense standardized VADER polarity scores (compound, pos, neg, neu)
    """
    print(f"\n[Feature Engineering] Vectorizing text with TF-IDF (max_features={max_features})...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=max_features,
        sublinear_tf=True
    )
    X_train_tfidf = vectorizer.fit_transform(train_df['negated_text'])
    X_test_tfidf = vectorizer.transform(test_df['negated_text'])

    vader_cols = ['vader_compound', 'vader_pos', 'vader_neg', 'vader_neu']
    scaler = StandardScaler()
    X_train_vader = scaler.fit_transform(train_df[vader_cols].values)
    X_test_vader = scaler.transform(test_df[vader_cols].values)

    # Combined hybrid feature matrix
    X_train_combined = sp.hstack([X_train_tfidf, X_train_vader], format='csr')
    X_test_combined = sp.hstack([X_test_tfidf, X_test_vader], format='csr')

    # MinMax scaled VADER for Naive Bayes (which requires strictly non-negative features)
    minmax = MinMaxScaler()
    X_train_vader_mm = minmax.fit_transform(train_df[vader_cols].values)
    X_test_vader_mm = minmax.transform(test_df[vader_cols].values)
    X_train_nb = sp.hstack([X_train_tfidf, X_train_vader_mm], format='csr')
    X_test_nb = sp.hstack([X_test_tfidf, X_test_vader_mm], format='csr')

    y_train = np.array(train_df['emotion'].tolist(), dtype=str)
    y_test = np.array(test_df['emotion'].tolist(), dtype=str)

    print(f"  Feature Matrix Dimensions: Train={X_train_combined.shape}, Test={X_test_combined.shape}")
    print(f"  Vocabulary Size: {len(vectorizer.vocabulary_):,}")

    return {
        "vectorizer": vectorizer,
        "scaler": scaler,
        "X_train": X_train_combined,
        "X_test": X_test_combined,
        "X_train_nb": X_train_nb,
        "X_test_nb": X_test_nb,
        "y_train": y_train,
        "y_test": y_test
    }


def evaluate_model(y_true, y_pred, model_name: str) -> Dict[str, float]:
    """Computes comprehensive classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    prec_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    rec_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_mac = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_wt = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "precision_macro": float(prec_macro),
        "precision_weighted": float(prec_weighted),
        "recall_macro": float(rec_macro),
        "recall_weighted": float(rec_weighted),
        "f1_macro": float(f1_mac),
        "f1_weighted": float(f1_wt)
    }


def train_and_track_experiments(sample_size: int = 0):
    """Main execution orchestrating all steps of Experiment 4."""
    start_time = time.time()
    plots_dir = EXPERIMENT_DIR / "plots"
    models_dir = EXPERIMENT_DIR / "models"
    reports_dir = EXPERIMENT_DIR / "reports"
    
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    print("=" * 85)
    print(" EXPERIMENT 4: ML MODELING, HYPERPARAMETER TUNING & MLFLOW TRACKING")
    print("=" * 85)

    # 1. Dataset Preparation (using entire 100,000 rows: 80,000 train / 20,000 test)
    train_df, test_df = load_and_prepare_data(sample_size=sample_size, random_state=42)
    feats = extract_features(train_df, test_df, max_features=10000)

    # Set up MLflow
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow_tracking_dir = PROJECT_ROOT / "mlruns"
    os.makedirs(mlflow_tracking_dir, exist_ok=True)
    sqlite_db = (mlflow_tracking_dir / "mlflow.db").as_posix()
    mlflow.set_tracking_uri(f"sqlite:///{sqlite_db}")
    experiment_name = "Customer_Support_Emotion_Classification_Exp4"
    mlflow.set_experiment(experiment_name)
    print(f"\n[MLflow Setup] Active Experiment: '{experiment_name}'")
    print(f"  Tracking URI: {mlflow.get_tracking_uri()}")

    # 2. Train 5 Baseline Models
    print("\n[Step 2/5] Training 5 Diverse Baseline Models...")
    
    baseline_models = {
        "Multinomial Naive Bayes": {
            "model": MultinomialNB(alpha=1.0),
            "params": {"alpha": 1.0, "fit_prior": True},
            "use_nb_feats": True,
            "family": "Probabilistic"
        },
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced', random_state=42),
            "params": {"C": 1.0, "max_iter": 1000, "class_weight": "balanced", "solver": "lbfgs"},
            "use_nb_feats": False,
            "family": "Linear"
        },
        "Linear Support Vector Machine": {
            "model": LinearSVC(C=1.0, class_weight='balanced', random_state=42, max_iter=2000),
            "params": {"C": 1.0, "max_iter": 2000, "class_weight": "balanced", "loss": "squared_hinge"},
            "use_nb_feats": False,
            "family": "Max-Margin"
        },
        "Random Forest": {
            "model": RandomForestClassifier(n_estimators=100, max_depth=25, class_weight='balanced', random_state=42, n_jobs=-1),
            "params": {"n_estimators": 100, "max_depth": 25, "class_weight": "balanced"},
            "use_nb_feats": False,
            "family": "Ensemble (Bagging)"
        },
        "LightGBM": {
            "model": LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1),
            "params": {"n_estimators": 100, "learning_rate": 0.1, "random_state": 42},
            "use_nb_feats": False,
            "family": "Gradient Boosting"
        }
    }

    baseline_results = []
    trained_clfs = {}
    test_preds_dict = {}

    for name, config in baseline_models.items():
        t0 = time.time()
        clf = config["model"]
        is_nb = config["use_nb_feats"]
        X_tr = feats["X_train_nb"] if is_nb else feats["X_train"]
        X_te = feats["X_test_nb"] if is_nb else feats["X_test"]

        print(f"  Training '{name}' ({config['family']})...", end="", flush=True)
        
        with mlflow.start_run(run_name=f"Baseline_{name.replace(' ', '_')}") as run:
            # Log params & tags
            mlflow.set_tag("stage", "baseline")
            mlflow.set_tag("model_family", config["family"])
            mlflow.log_params(config["params"])
            mlflow.log_param("train_samples", len(train_df))
            mlflow.log_param("test_samples", len(test_df))
            mlflow.log_param("features", X_tr.shape[1])

            # Train
            clf.fit(X_tr, feats["y_train"])
            train_time = time.time() - t0

            # Predict
            t_pred0 = time.time()
            y_pred = clf.predict(X_te)
            infer_time = time.time() - t_pred0

            # Evaluate
            metrics = evaluate_model(feats["y_test"], y_pred, name)
            metrics["train_time_sec"] = round(train_time, 3)
            metrics["infer_time_sec"] = round(infer_time, 3)

            # Log metrics to MLflow
            for m_key, m_val in metrics.items():
                mlflow.log_metric(m_key, m_val)

            # Log classification report text artifact
            report_str = classification_report(feats["y_test"], y_pred, digits=4, zero_division=0)
            report_file = reports_dir / f"clf_report_baseline_{name.lower().replace(' ', '_')}.txt"
            with open(report_file, "w") as f:
                f.write(report_str)
            mlflow.log_artifact(str(report_file), artifact_path="evaluation_reports")

            # Log model artifact in MLflow
            try:
                mlflow.sklearn.log_model(clf, artifact_path="model")
            except Exception as e:
                print(f" [MLflow log_model warning: {e}]", end="")

            trained_clfs[name] = clf
            test_preds_dict[name] = y_pred

            res_record = {
                "Model": name,
                "Stage": "Baseline",
                "Family": config["family"],
                "Accuracy": metrics["accuracy"],
                "Macro Precision": metrics["precision_macro"],
                "Macro Recall": metrics["recall_macro"],
                "Macro F1": metrics["f1_macro"],
                "Weighted F1": metrics["f1_weighted"],
                "Training Time (s)": metrics["train_time_sec"],
                "MLflow Run ID": run.info.run_id
            }
            baseline_results.append(res_record)
            print(f" Done ({train_time:.2f}s) | Acc: {metrics['accuracy']:.4f} | Macro F1: {metrics['f1_macro']:.4f}")

    # 3. Hyperparameter Tuning (GridSearchCV on Logistic Regression & LightGBM)
    print("\n[Step 3/5] Hyperparameter Tuning with GridSearchCV...")
    tuned_results = []

    # --- Tuning Model 1: Logistic Regression ---
    print("  [Tuning 1/2] Tuning Logistic Regression hyperparameters...")
    param_grid_lr = {
        'C': [0.5, 1.0, 5.0],
        'class_weight': ['balanced', None]
    }
    lr_base = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)
    grid_lr = GridSearchCV(
        lr_base,
        param_grid=param_grid_lr,
        cv=3,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=0
    )
    
    with mlflow.start_run(run_name="Tuned_Logistic_Regression_GridSearch") as run_lr:
        mlflow.set_tag("stage", "hyperparameter_tuning")
        mlflow.set_tag("tuning_method", "GridSearchCV")
        mlflow.log_param("cv_folds", 3)
        mlflow.log_param("candidate_combinations", len(param_grid_lr['C']) * len(param_grid_lr['class_weight']))

        t_tune0 = time.time()
        grid_lr.fit(feats["X_train"], feats["y_train"])
        tune_lr_time = time.time() - t_tune0
        
        best_lr = grid_lr.best_estimator_
        y_pred_tuned_lr = best_lr.predict(feats["X_test"])
        metrics_tuned_lr = evaluate_model(feats["y_test"], y_pred_tuned_lr, "Tuned Logistic Regression")
        metrics_tuned_lr["cv_best_score"] = float(grid_lr.best_score_)
        metrics_tuned_lr["tuning_time_sec"] = round(tune_lr_time, 3)

        # Log best params & metrics
        for p_k, p_v in grid_lr.best_params_.items():
            mlflow.log_param(f"best_{p_k}", p_v)
        for m_k, m_v in metrics_tuned_lr.items():
            mlflow.log_metric(m_k, m_v)

        # Log model
        try:
            mlflow.sklearn.log_model(best_lr, artifact_path="best_tuned_model")
        except Exception:
            pass

        trained_clfs["Tuned Logistic Regression"] = best_lr
        test_preds_dict["Tuned Logistic Regression"] = y_pred_tuned_lr

        tuned_lr_record = {
            "Model": "Tuned Logistic Regression",
            "Stage": "Tuned (GridSearchCV)",
            "Family": "Linear",
            "Accuracy": metrics_tuned_lr["accuracy"],
            "Macro Precision": metrics_tuned_lr["precision_macro"],
            "Macro Recall": metrics_tuned_lr["recall_macro"],
            "Macro F1": metrics_tuned_lr["f1_macro"],
            "Weighted F1": metrics_tuned_lr["f1_weighted"],
            "Training Time (s)": tune_lr_time,
            "MLflow Run ID": run_lr.info.run_id,
            "Best Params": grid_lr.best_params_
        }
        tuned_results.append(tuned_lr_record)
        print(f"    Best Params: {grid_lr.best_params_}")
        print(f"    CV Macro F1: {grid_lr.best_score_:.4f} -> Test Acc: {metrics_tuned_lr['accuracy']:.4f} | Test Macro F1: {metrics_tuned_lr['f1_macro']:.4f}")

    # --- Tuning Model 2: LightGBM ---
    print("  [Tuning 2/2] Tuning LightGBM hyperparameters...")
    param_grid_lgb = {
        'n_estimators': [100],
        'learning_rate': [0.1],
        'num_leaves': [31, 63]
    }
    lgb_base = LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)
    grid_lgb = GridSearchCV(
        lgb_base,
        param_grid=param_grid_lgb,
        cv=3,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=0
    )

    with mlflow.start_run(run_name="Tuned_LightGBM_GridSearch") as run_lgb:
        mlflow.set_tag("stage", "hyperparameter_tuning")
        mlflow.set_tag("tuning_method", "GridSearchCV")
        mlflow.log_param("cv_folds", 3)

        t_tune1 = time.time()
        grid_lgb.fit(feats["X_train"], feats["y_train"])
        tune_lgb_time = time.time() - t_tune1

        best_lgb = grid_lgb.best_estimator_
        y_pred_tuned_lgb = best_lgb.predict(feats["X_test"])
        metrics_tuned_lgb = evaluate_model(feats["y_test"], y_pred_tuned_lgb, "Tuned LightGBM")
        metrics_tuned_lgb["cv_best_score"] = float(grid_lgb.best_score_)
        metrics_tuned_lgb["tuning_time_sec"] = round(tune_lgb_time, 3)

        for p_k, p_v in grid_lgb.best_params_.items():
            mlflow.log_param(f"best_{p_k}", p_v)
        for m_k, m_v in metrics_tuned_lgb.items():
            mlflow.log_metric(m_k, m_v)

        try:
            mlflow.sklearn.log_model(best_lgb, artifact_path="best_tuned_model")
        except Exception:
            pass

        trained_clfs["Tuned LightGBM"] = best_lgb
        test_preds_dict["Tuned LightGBM"] = y_pred_tuned_lgb

        tuned_lgb_record = {
            "Model": "Tuned LightGBM",
            "Stage": "Tuned (GridSearchCV)",
            "Family": "Gradient Boosting",
            "Accuracy": metrics_tuned_lgb["accuracy"],
            "Macro Precision": metrics_tuned_lgb["precision_macro"],
            "Macro Recall": metrics_tuned_lgb["recall_macro"],
            "Macro F1": metrics_tuned_lgb["f1_macro"],
            "Weighted F1": metrics_tuned_lgb["f1_weighted"],
            "Training Time (s)": tune_lgb_time,
            "MLflow Run ID": run_lgb.info.run_id,
            "Best Params": grid_lgb.best_params_
        }
        tuned_results.append(tuned_lgb_record)
        print(f"    Best Params: {grid_lgb.best_params_}")
        print(f"    CV Macro F1: {grid_lgb.best_score_:.4f} -> Test Acc: {metrics_tuned_lgb['accuracy']:.4f} | Test Macro F1: {metrics_tuned_lgb['f1_macro']:.4f}")

    # Combine benchmark DataFrame
    all_results = baseline_results + tuned_results
    benchmark_df = pd.DataFrame(all_results)
    benchmark_csv_path = reports_dir / "model_benchmark_results.csv"
    benchmark_df.to_csv(benchmark_csv_path, index=False)
    print(f"\n[+] Exported complete model benchmark table to: {benchmark_csv_path}")

    # 4. Generate Visualizations & Diagnostic Plots
    print("\n[Step 4/5] Generating Visualizations & Comparative Evaluation Plots...")

    # Plot 1: Model Comparison (Accuracy & Macro F1)
    plt.figure(figsize=(12, 5.5))
    plot_df = benchmark_df.copy()
    df_melted = plot_df.melt(
        id_vars=["Model", "Stage"],
        value_vars=["Accuracy", "Macro F1", "Weighted F1"],
        var_name="Metric",
        value_name="Score"
    )
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(
        data=df_melted,
        x="Model",
        y="Score",
        hue="Metric",
        palette=["#2b5c8f", "#d95f02", "#7570b3"]
    )
    plt.title("Experiment 4: Model Performance Benchmark (Baseline vs. Tuned Architectures)", fontsize=13, fontweight='bold', pad=14)
    plt.xlabel("Model Architecture", fontsize=11, fontweight='bold')
    plt.ylabel("Score", fontsize=11, fontweight='bold')
    plt.ylim(0.40, 1.0)
    plt.xticks(rotation=20, ha='right', fontsize=9.5)
    plt.legend(title="Metric", loc="lower right", frameon=True)

    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{h:.3f}", (p.get_x() + p.get_width() / 2., h),
                        ha='center', va='bottom', fontsize=8, fontweight='bold', xytext=(0, 2),
                        textcoords='offset points')

    plt.tight_layout()
    comparison_plot_path = plots_dir / "exp4_model_comparison.png"
    plt.savefig(comparison_plot_path, dpi=300)
    plt.close()
    print(f"  [+] Saved benchmark plot to: {comparison_plot_path}")

    # Plot 2: Confusion Matrices for Top Models
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    
    # Baseline Top Model (e.g. Linear SVM or Logistic Regression)
    cm_base = confusion_matrix(feats["y_test"], test_preds_dict["Linear Support Vector Machine"], labels=EMOTION_CLASSES)
    sns.heatmap(
        cm_base, annot=True, fmt='d', cmap='Blues', ax=axes[0],
        xticklabels=[c.split('/')[0].strip() for c in EMOTION_CLASSES],
        yticklabels=[c.split('/')[0].strip() for c in EMOTION_CLASSES]
    )
    axes[0].set_title("Baseline: Linear SVM Confusion Matrix", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Predicted Label", fontsize=10.5)
    axes[0].set_ylabel("True Label", fontsize=10.5)

    # Best Tuned Model (e.g. Tuned Logistic Regression)
    cm_tuned = confusion_matrix(feats["y_test"], test_preds_dict["Tuned Logistic Regression"], labels=EMOTION_CLASSES)
    sns.heatmap(
        cm_tuned, annot=True, fmt='d', cmap='Greens', ax=axes[1],
        xticklabels=[c.split('/')[0].strip() for c in EMOTION_CLASSES],
        yticklabels=[c.split('/')[0].strip() for c in EMOTION_CLASSES]
    )
    axes[1].set_title("Tuned: Logistic Regression (GridSearchCV)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Predicted Label", fontsize=10.5)
    axes[1].set_ylabel("True Label", fontsize=10.5)

    plt.suptitle("Experiment 4: Confusion Matrix Analysis Across Emotion Categories", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    cm_plot_path = plots_dir / "exp4_confusion_matrices.png"
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    print(f"  [+] Saved confusion matrices to: {cm_plot_path}")

    # Plot 3: Tuning Comparison (Delta Improvement)
    plt.figure(figsize=(9, 5))
    tuning_delta_data = [
        {"Model": "Logistic Regression", "Metric": "Accuracy", "Baseline": baseline_results[1]["Accuracy"], "Tuned": tuned_results[0]["Accuracy"]},
        {"Model": "Logistic Regression", "Metric": "Macro F1", "Baseline": baseline_results[1]["Macro F1"], "Tuned": tuned_results[0]["Macro F1"]},
        {"Model": "LightGBM", "Metric": "Accuracy", "Baseline": baseline_results[4]["Accuracy"], "Tuned": tuned_results[1]["Accuracy"]},
        {"Model": "LightGBM", "Metric": "Macro F1", "Baseline": baseline_results[4]["Macro F1"], "Tuned": tuned_results[1]["Macro F1"]}
    ]
    tune_df = pd.DataFrame(tuning_delta_data)
    tune_melt = tune_df.melt(id_vars=["Model", "Metric"], value_vars=["Baseline", "Tuned"], var_name="Condition", value_name="Score")
    
    ax3 = sns.barplot(data=tune_melt, x="Model", y="Score", hue="Condition", palette=["#e74c3c", "#2ecc71"])
    plt.title("Hyperparameter Tuning Impact: Baseline vs. GridSearchCV Optimized", fontsize=12, fontweight='bold', pad=12)
    plt.ylim(0.50, 0.95)
    plt.ylabel("Score", fontsize=11)
    
    for p in ax3.patches:
        h = p.get_height()
        if h > 0:
            ax3.annotate(f"{h:.3f}", (p.get_x() + p.get_width() / 2., h),
                         ha='center', va='bottom', fontsize=9, fontweight='bold', xytext=(0, 2),
                         textcoords='offset points')

    plt.tight_layout()
    tuning_plot_path = plots_dir / "exp4_tuning_comparison.png"
    plt.savefig(tuning_plot_path, dpi=300)
    plt.close()
    print(f"  [+] Saved tuning comparison plot to: {tuning_plot_path}")

    # Plot 4: MLflow Dashboard Architecture & Run Summary
    fig, ax_ml = plt.subplots(figsize=(11, 5.5))
    ax_ml.axis('off')
    runs_summary = [
        f"Experiment Name : {experiment_name}",
        f"Total Tracked Runs : {len(all_results)} (5 Baselines + 2 Tuned GridSearches)",
        f"Artifact Location  : {mlflow_tracking_dir}",
        "",
        f"{'Run Name':<38} | {'Stage':<14} | {'Accuracy':<8} | {'Macro F1':<8} | {'Run ID':<32}",
        "-" * 110
    ]
    for r in all_results:
        runs_summary.append(
            f"{r['Model']:<38} | {r['Stage']:<14} | {r['Accuracy']:.4f}   | {r['Macro F1']:.4f}   | {r['MLflow Run ID'][:28]}..."
        )
    
    ax_ml.text(
        0.02, 0.95, "\n".join(runs_summary),
        fontsize=9.5, fontfamily="monospace", va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#f8f9fa', edgecolor='#ced4da')
    )
    plt.title("MLflow Tracking Dashboard Logs: Experiment Run Summary", fontsize=13, fontweight='bold', pad=10)
    plt.tight_layout()
    mlflow_plot_path = plots_dir / "exp4_mlflow_dashboard.png"
    plt.savefig(mlflow_plot_path, dpi=300)
    plt.close()
    print(f"  [+] Saved MLflow run dashboard plot to: {mlflow_plot_path}")

    # 5. Model Selection & Saving
    print("\n[Step 5/5] Model Selection & Production Artifact Serialization...")
    # Best model selection based on highest Macro F1
    best_record = max(all_results, key=lambda x: x["Macro F1"])
    best_model_name = best_record["Model"]
    best_clf = trained_clfs[best_model_name]
    
    print(f"  Champion Model Selected: '{best_model_name}'")
    print(f"    - Macro F1 : {best_record['Macro F1']:.4f}")
    print(f"    - Accuracy : {best_record['Accuracy']:.4f}")
    print(f"    - Run ID   : {best_record['MLflow Run ID']}")

    # Build production inference pipeline
    production_pipeline = EndToEndEmotionPipeline(
        vectorizer=feats["vectorizer"],
        scaler=feats["scaler"],
        classifier=best_clf,
        emotion_classes=EMOTION_CLASSES
    )
    
    pipeline_joblib_path = models_dir / "best_emotion_model_exp4.joblib"
    joblib.dump(production_pipeline, pipeline_joblib_path)
    print(f"  [+] Serialized production model pipeline to: {pipeline_joblib_path}")

    # Log champion pipeline to MLflow under a dedicated champion run
    with mlflow.start_run(run_name="Production_Champion_Model") as champ_run:
        mlflow.set_tag("stage", "production_selected")
        mlflow.set_tag("champion_model", best_model_name)
        mlflow.log_metric("champion_accuracy", best_record["Accuracy"])
        mlflow.log_metric("champion_macro_f1", best_record["Macro F1"])
        mlflow.log_artifact(str(pipeline_joblib_path), artifact_path="serialized_pipeline")
        mlflow.log_artifact(str(comparison_plot_path), artifact_path="visualizations")
        mlflow.log_artifact(str(cm_plot_path), artifact_path="visualizations")
        mlflow.log_artifact(str(tuning_plot_path), artifact_path="visualizations")

    # Export Summary JSON Metadata
    summary_metadata = {
        "experiment_number": 4,
        "experiment_title": "ML Modeling & Experiment Tracking",
        "dataset": "Customer Support on Twitter (TWCS)",
        "sample_size": len(train_df) + len(test_df),
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "emotion_classes": EMOTION_CLASSES,
        "champion_model": best_model_name,
        "champion_metrics": {
            "accuracy": best_record["Accuracy"],
            "macro_f1": best_record["Macro F1"],
            "weighted_f1": best_record["Weighted F1"],
            "training_time_sec": best_record["Training Time (s)"]
        },
        "all_benchmarks": all_results,
        "mlflow_experiment_name": experiment_name,
        "mlflow_tracking_uri": str(mlflow_tracking_dir),
        "total_runtime_sec": round(time.time() - start_time, 2)
    }
    
    metadata_path = models_dir / "exp4_model_summary.json"
    with open(metadata_path, "w") as f:
        json.dump(summary_metadata, f, indent=4)
    print(f"  [+] Saved experiment summary metadata to: {metadata_path}")

    # Copy generated plots to reports/experiment_4/plots/ for self-contained LaTeX compilation
    rep_plots_dir = reports_dir / "plots"
    os.makedirs(rep_plots_dir, exist_ok=True)
    for p_file in [comparison_plot_path, cm_plot_path, tuning_plot_path, mlflow_plot_path]:
        dest = rep_plots_dir / p_file.name
        with open(p_file, "rb") as sf, open(dest, "wb") as df:
            df.write(sf.read())
    print(f"  [+] Synced plots to: {rep_plots_dir}")

    total_time = time.time() - start_time
    print("\n" + "=" * 85)
    print(f" EXPERIMENT 4 TRAINING COMPLETED IN {total_time:.2f} SECONDS!")
    print("=" * 85)
    return summary_metadata


if __name__ == '__main__':
    train_and_track_experiments()
