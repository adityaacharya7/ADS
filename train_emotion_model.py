import os
import json
import time
import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import joblib

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from emotion_labeler import EmotionLabeler

# Ensure plots and models directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

class EmotionPipeline:
    """
    Combined feature engineering and classifier container for inference.
    """
    def __init__(self, vectorizer, scaler, label_encoder, classifier, feature_names=None):
        self.vectorizer = vectorizer
        self.scaler = scaler
        self.label_encoder = label_encoder
        self.classifier = classifier
        self.feature_names = feature_names

    def transform_features(self, texts, vader_feats):
        X_tfidf = self.vectorizer.transform(texts)
        X_vader_scaled = self.scaler.transform(vader_feats)
        X_combined = sp.hstack([X_tfidf, X_vader_scaled], format='csr')
        return X_combined

    def predict(self, texts, vader_feats):
        X = self.transform_features(texts, vader_feats)
        preds = self.classifier.predict(X)
        return self.label_encoder.inverse_transform(preds)

    def predict_proba(self, texts, vader_feats):
        X = self.transform_features(texts, vader_feats)
        return self.classifier.predict_proba(X)


def plot_emotion_distribution(df, output_path="plots/emotion_distribution.png"):
    """Plot and save class distribution chart."""
    plt.figure(figsize=(10, 5))
    emotion_counts = df['emotion'].value_counts()
    ax = sns.barplot(x=emotion_counts.index, y=emotion_counts.values, hue=emotion_counts.index, palette="viridis", legend=False)
    plt.title("TWCS Emotion Class Distribution", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Emotion Category", fontsize=12)
    plt.ylabel("Number of Tweets", fontsize=12)
    plt.xticks(rotation=15)
    
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height()):,}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 8), 
                    textcoords='offset points', fontsize=10)
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[+] Saved emotion distribution plot to '{output_path}'")


def plot_model_comparison(results_df, output_path="plots/model_comparison.png"):
    """Plot model accuracy and F1 score comparison."""
    plt.figure(figsize=(10, 5))
    df_melt = results_df.melt(id_vars=["Model"], value_vars=["Accuracy", "Weighted F1"], 
                              var_name="Metric", value_name="Score")
    ax = sns.barplot(data=df_melt, x="Model", y="Score", hue="Metric", palette="mako")
    plt.title("Emotion Model Benchmark Performance Comparison", fontsize=14, fontweight='bold', pad=15)
    plt.ylim(0.70, 1.0)
    plt.ylabel("Score", fontsize=12)
    
    for p in ax.patches:
        val = p.get_height()
        if not np.isnan(val) and val > 0:
            ax.annotate(f'{val:.4f}', 
                        (p.get_x() + p.get_width() / 2., val),
                        ha='center', va='center', xytext=(0, 6), 
                        textcoords='offset points', fontsize=9, fontweight='bold')
            
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[+] Saved model comparison plot to '{output_path}'")


def plot_confusion_matrix_heatmap(cm, classes, model_name, output_path="plots/confusion_matrix.png"):
    """Plot confusion matrix heatmap for winning model."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f"Confusion Matrix - {model_name}", fontsize=14, fontweight='bold', pad=15)
    plt.ylabel("Actual Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.xticks(rotation=20)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[+] Saved confusion matrix heatmap to '{output_path}'")


def train_and_evaluate(dataset_path: str, sample_size: int = 0):
    start_time = time.time()
    print("=" * 70)
    print("      TWCS EMOTION DETECTION MODEL TRAINING & EVALUATION PIPELINE")
    print("=" * 70)
    
    # 1. Load Dataset
    print(f"\n[Step 1/5] Loading cleaned tweets dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f" Raw Dataset Size: {len(df):,} rows")
    
    if sample_size > 0 and len(df) > sample_size:
        print(f" Sampling {sample_size:,} rows for rapid training & benchmarking...")
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        
    df['clean_text'] = df['clean_text'].fillna("")
    
    # 2. Label Emotions using EmotionLabeler
    print("\n[Step 2/5] Annotating tweets with Sentiment and Emotion Labels...")
    labeler = EmotionLabeler()
    df_labeled = labeler.label_dataframe(df, text_column='clean_text')
    
    print("\nClass Distribution:")
    for em, cnt in df_labeled['emotion'].value_counts().items():
        print(f"  - {em:25s}: {cnt:7,} ({cnt/len(df_labeled)*100:.1f}%)")
        
    plot_emotion_distribution(df_labeled)

    # 3. Feature Engineering
    print("\n[Step 3/5] Performing Feature Extraction (TF-IDF N-grams + VADER Polarity)...")
    X_texts = np.array(df_labeled['clean_text'].astype(str).tolist())
    vader_cols = ['vader_compound', 'vader_pos', 'vader_neg', 'vader_neu']
    X_vader = np.array(df_labeled[vader_cols].values, dtype=np.float32)

    # Encode target labels
    label_enc = LabelEncoder()
    y = label_enc.fit_transform(df_labeled['emotion'])
    class_names = label_enc.classes_

    # Train/Test Split (80/20 Stratified)
    X_text_train, X_text_test, X_vader_train, X_vader_test, y_train, y_test = train_test_split(
        X_texts, X_vader, y, test_size=0.20, random_state=42, stratify=y
    )

    # Vectorize text features
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True, stop_words='english')
    X_tfidf_train = vectorizer.fit_transform(X_text_train)
    X_tfidf_test = vectorizer.transform(X_text_test)

    # Scale VADER numerical features
    scaler = StandardScaler()
    X_vader_train_scaled = scaler.fit_transform(X_vader_train)
    X_vader_test_scaled = scaler.transform(X_vader_test)

    # Stack sparse TF-IDF and dense VADER features
    X_train_full = sp.hstack([X_tfidf_train, X_vader_train_scaled], format='csr')
    X_test_full = sp.hstack([X_tfidf_test, X_vader_test_scaled], format='csr')

    print(f" Feature Matrix Shape: Train {X_train_full.shape}, Test {X_test_full.shape}")

    # 4. Model Training & Comparison
    print("\n[Step 4/5] Training and Benchmarking Candidate ML Classifiers...")
    candidate_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=2.0, random_state=42, n_jobs=-1),
        "LightGBM": LGBMClassifier(n_estimators=150, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1),
        "XGBoost": XGBClassifier(n_estimators=150, learning_rate=0.1, random_state=42, n_jobs=-1, eval_metric='mlogloss')
    }

    benchmark_results = []
    trained_clf_objects = {}

    for name, clf in candidate_models.items():
        m_start = time.time()
        print(f" Training {name:20s} ...", end="", flush=True)
        clf.fit(X_train_full, y_train)
        preds = clf.predict(X_test_full)
        elapsed = time.time() - m_start

        acc = accuracy_score(y_test, preds)
        f1_w = f1_score(y_test, preds, average='weighted')
        prec_w = precision_score(y_test, preds, average='weighted')
        rec_w = recall_score(y_test, preds, average='weighted')

        print(f" Done in {elapsed:.2f}s | Acc: {acc:.4f} | F1: {f1_w:.4f}")

        benchmark_results.append({
            "Model": name,
            "Accuracy": acc,
            "Weighted F1": f1_w,
            "Precision": prec_w,
            "Recall": rec_w,
            "Training Time (s)": elapsed
        })
        trained_clf_objects[name] = clf

    results_df = pd.DataFrame(benchmark_results)
    plot_model_comparison(results_df)

    # Find best model based on F1 score
    best_row = results_df.sort_values(by="Weighted F1", ascending=False).iloc[0]
    best_name = best_row["Model"]
    best_clf = trained_clf_objects[best_name]

    print("\n" + "=" * 70)
    print(f" BEST PERFORMING MODEL: {best_name.upper()}")
    print("=" * 70)
    
    y_pred_best = best_clf.predict(X_test_full)
    report_str = classification_report(y_test, y_pred_best, target_names=class_names, digits=4)
    print("\nClassification Report:\n")
    print(report_str)

    cm = confusion_matrix(y_test, y_pred_best)
    plot_confusion_matrix_heatmap(cm, class_names, best_name)

    # 5. Model Serialization
    print("\n[Step 5/5] Serializing Model Pipeline & Metadata...")
    pipeline = EmotionPipeline(
        vectorizer=vectorizer,
        scaler=scaler,
        label_encoder=label_enc,
        classifier=best_clf
    )
    
    pipeline_path = "models/emotion_pipeline.joblib"
    joblib.dump(pipeline, pipeline_path)
    print(f" Saved full model pipeline artifact to: '{pipeline_path}'")

    metadata = {
        "best_model": best_name,
        "sample_size": len(df),
        "train_samples": int(X_train_full.shape[0]),
        "test_samples": int(X_test_full.shape[0]),
        "vocab_size": len(vectorizer.vocabulary_),
        "classes": list(class_names),
        "benchmark": results_df.to_dict(orient="records"),
        "training_time_seconds": time.time() - start_time
    }
    
    metadata_path = "models/model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f" Saved model metadata to: '{metadata_path}'")
    print(f"\nPipeline training complete in {time.time() - start_time:.2f} seconds!")
    print("=" * 70)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Emotion Detection ML Models on TWCS Dataset")
    parser.add_argument("--input", "-i", type=str, default="twcs_cleaned.csv", help="Path to clean CSV")
    parser.add_argument("--sample-size", "-s", type=int, default=0, help="Sample size for training (0 for all rows)")
    args = parser.parse_args()
    
    train_and_evaluate(args.input, args.sample_size)
