import os
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score, precision_score,
    recall_score, hamming_loss, jaccard_score, confusion_matrix
)
import joblib

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from .preprocessing import apply_negation_tagging, clean_tweet_text
from .emotion_labeler import EmotionLabeler, EMOTION_CLASSES

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Custom stop words that explicitly PRESERVE negation particles
NEGATION_WORDS = {
    "not", "no", "never", "nor", "neither", "without", "hardly", "barely",
    "scarcely", "cannot", "cant", "n't"
}
SAFE_STOP_WORDS = list(ENGLISH_STOP_WORDS - NEGATION_WORDS)

# Nuanced Synthetic Augmentation Examples for Edge-Case Robustness
AUGMENTATION_SAMPLES = [
    # Negations
    ("I am not angry with customer service, just asking for a status update.", [0, 0, 0, 0, 1]),
    ("I was not disappointed with the outcome, in fact I am very happy!", [1, 0, 0, 0, 0]),
    ("I don't feel anxious about this issue at all.", [0, 0, 0, 0, 1]),
    ("I'm not worried about the delay, thank you for clarifying.", [1, 0, 0, 0, 0]),
    ("I am not furious, please just refund the shipping fee.", [0, 0, 1, 0, 0]),
    ("Never had a single problem with this service, love it!", [1, 0, 0, 0, 0]),
    ("No complaints here, everything was resolved in minutes.", [1, 0, 0, 0, 0]),
    ("I could not shake the fear that our reservation was cancelled.", [0, 0, 0, 1, 0]),
    ("I could not stop the anxiety before the flight.", [0, 0, 0, 1, 0]),
    # Contrasts, Hypotheticals & Multi-Emotion Complexity
    ("I thought I would be furious, but instead I felt an overwhelming sense of relief and pride, though I still could not shake the fear that everything we had achieved might disappear.", [1, 0, 0, 1, 0]),
    ("I was disappointed at first, but eventually I was very happy with the resolution.", [1, 0, 1, 0, 0]),
    ("The delivery was delayed but the support team was fantastic and helpful.", [1, 0, 1, 0, 0]),
    ("I was furious earlier, but thank you for fixing it so quickly.", [1, 1, 0, 0, 0]),
    ("I found myself unexpectedly proud of the milestone, although I remained anxious about the release.", [1, 0, 0, 1, 0]),
    ("I'm excited but nervous about tomorrow's flight.", [1, 0, 0, 1, 0]),
    ("I feel proud of the progress yet anxious about the upcoming audit.", [1, 0, 0, 1, 0]),
    ("Thrilled with the upgrade though worried about the billing change.", [1, 0, 0, 1, 0]),
    ("I wasn't afraid of the outcome, and although everyone around me seemed nervous, I remained surprisingly calm and confident throughout the entire process.", [1, 0, 0, 0, 0]),
    ("I was afraid of the outcome, but I tried to remain calm and confident throughout the entire process.", [1, 0, 0, 1, 0]),
    ("I am not afraid of failure, I am excited for the challenge.", [1, 0, 0, 0, 0]),
    ("I was terrified at the start, but gradually grew confident.", [1, 0, 0, 1, 0]),
    ("I feel completely confident and calm about this decision.", [1, 0, 0, 0, 0]),
    ("I am confident that everything will go smoothly, no anxiety whatsoever.", [1, 0, 0, 0, 0]),
    ("I was not nervous at all, feeling confident and relaxed.", [1, 0, 0, 0, 0]),
    ("I am not nervous, completely calm and prepared.", [1, 0, 0, 0, 0]),
    ("I am not afraid at all, feeling proud and composed.", [1, 0, 0, 0, 0]),
    # Conditionals
    ("I would be disappointed if this promotion expired tomorrow.", [0, 0, 1, 0, 1]),
    ("If my data was leaked in the breach, I will be furious.", [0, 1, 0, 1, 0])
]


class MultiLabelEmotionPipeline:
    """
    Production Negation-Aware Multi-Label Emotion Classifier Pipeline.
    Predicts independent calibrated probabilities across all 5 emotion dimensions.
    """
    def __init__(self, vectorizer, scaler, classifier, emotion_classes=None):
        self.vectorizer = vectorizer
        self.scaler = scaler
        self.classifier = classifier
        self.emotion_classes = emotion_classes or EMOTION_CLASSES

    def transform_features(self, texts: List[str], vader_feats: np.ndarray):
        # 1. Apply negation scope transformation
        negated_texts = [apply_negation_tagging(t) for t in texts]
        # 2. Extract (1, 3) N-Gram TF-IDF features
        X_tfidf = self.vectorizer.transform(negated_texts)
        # 3. Scale VADER numerical features
        X_vader_scaled = self.scaler.transform(vader_feats)
        # 4. Concatenate sparse lexical and dense polarity features
        X_combined = sp.hstack([X_tfidf, X_vader_scaled], format='csr')
        return X_combined

    def predict_proba(self, texts: List[str], vader_feats: np.ndarray) -> np.ndarray:
        """Returns N x 5 probability matrix where each column is P(Emotion_k | Text)."""
        X = self.transform_features(texts, vader_feats)
        
        if hasattr(self.classifier, "predict_proba"):
            probs = self.classifier.predict_proba(X)
            if isinstance(probs, list):
                probs = np.column_stack([p[:, 1] if p.shape[1] > 1 else p[:, 0] for p in probs])
            return probs
        elif hasattr(self.classifier, "decision_function"):
            df = self.classifier.decision_function(X)
            return 1.0 / (1.0 + np.exp(-df))
        else:
            return self.classifier.predict(X).astype(float)

    def predict(self, texts: List[str], vader_feats: np.ndarray, threshold: float = 0.35) -> np.ndarray:
        """Returns binary multi-label indicator matrix (N x 5)."""
        probas = self.predict_proba(texts, vader_feats)
        binary = (probas >= threshold).astype(int)
        
        for i in range(len(binary)):
            if binary[i].sum() == 0:
                binary[i, np.argmax(probas[i])] = 1
        return binary

    def predict_profile(self, raw_text: str, vader_res: Dict[str, Any], secondary_threshold: float = 0.20) -> Dict[str, Any]:
        """Generates rich multi-dimensional emotion profile for a single text."""
        cleaned = clean_tweet_text(raw_text)
        negated = apply_negation_tagging(cleaned)
        
        vader_feats = np.array([[
            vader_res['compound'],
            vader_res['pos'],
            vader_res['neg'],
            vader_res['neu']
        ]])
        
        probas = self.predict_proba([cleaned], vader_feats)[0]
        prob_dict = {cls: float(np.clip(p, 0.0, 1.0)) for cls, p in zip(self.emotion_classes, probas)}
        
        sorted_emotions = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
        primary_name, primary_prob = sorted_emotions[0]
        
        second_name, second_prob = sorted_emotions[1]
        is_mixed = (second_prob >= secondary_threshold)
        secondary_name = second_name if is_mixed else "None detected above threshold"
        
        return {
            "raw_text": raw_text,
            "cleaned_text": cleaned,
            "negated_text": negated,
            "primary_emotion": primary_name,
            "primary_confidence": primary_prob,
            "secondary_emotion": secondary_name,
            "secondary_confidence": second_prob if is_mixed else 0.0,
            "is_mixed": is_mixed,
            "probabilities": prob_dict,
            "vader_compound": vader_res['compound'],
            "sentiment": vader_res['sentiment']
        }


def plot_multi_label_metrics(benchmark_df: pd.DataFrame, output_path: str = None):
    """Plots multi-label model benchmark comparisons."""
    if output_path is None:
        output_path = str(PROJECT_ROOT / "plots" / "model_comparison.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.figure(figsize=(12, 5.5))
    df_melt = benchmark_df.melt(id_vars=["Architecture", "Model"], 
                                value_vars=["Macro F1", "Micro F1", "Jaccard Score"],
                                var_name="Metric", value_name="Score")
    
    ax = sns.barplot(data=df_melt, x="Model", y="Score", hue="Metric", palette="deep")
    plt.title("Emotion Model Architecture Benchmark (Baseline vs Multi-Label Negation-Aware)", 
              fontsize=13, fontweight='bold', pad=15)
    plt.ylim(0.60, 1.0)
    plt.ylabel("Performance Score", fontsize=11)
    plt.xlabel("Candidate Architecture / Model", fontsize=11)
    
    for p in ax.patches:
        val = p.get_height()
        if not np.isnan(val) and val > 0:
            ax.annotate(f'{val:.3f}', 
                        (p.get_x() + p.get_width() / 2., val),
                        ha='center', va='center', xytext=(0, 6), 
                        textcoords='offset points', fontsize=9, fontweight='bold')
            
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[+] Saved multi-label benchmark plot to '{output_path}'")


def train_and_evaluate(dataset_path: str = None, sample_size: int = 0):
    if dataset_path is None:
        dataset_path = str(PROJECT_ROOT / "data" / "processed" / "twcs_cleaned.csv")
        
    start_time = time.time()
    print("=" * 80)
    print("  NEGATION-AWARE MULTI-LABEL EMOTION ARCHITECTURE TRAINING & BENCHMARK")
    print("=" * 80)
    
    os.makedirs(str(PROJECT_ROOT / "models"), exist_ok=True)
    os.makedirs(str(PROJECT_ROOT / "plots"), exist_ok=True)
    
    # 1. Load Dataset
    print(f"\n[Step 1/6] Loading cleaned tweets dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f" Raw Cleaned Rows: {len(df):,}")
    
    if sample_size > 0 and len(df) > sample_size:
        print(f" Sampling {sample_size:,} records for training & comparative benchmarking...")
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        
    df['clean_text'] = df['clean_text'].fillna("")
    
    # 2. Negation & Multi-Label Annotation
    print("\n[Step 2/6] Annotating Dataset with Negation Scope & Multi-Label Targets...")
    labeler = EmotionLabeler()
    df_labeled = labeler.label_dataframe(df, text_column='clean_text')
    
    # Inject Synthetic Nuanced Linguistic Samples into Training Pool
    aug_texts = [s[0] for s in AUGMENTATION_SAMPLES]
    aug_vectors = np.array([s[1] for s in AUGMENTATION_SAMPLES])
    aug_vader = [labeler.vader.polarity_scores(t) for t in aug_texts]
    
    aug_df = pd.DataFrame({
        "clean_text": aug_texts,
        "vader_compound": [v['compound'] for v in aug_vader],
        "vader_pos": [v['pos'] for v in aug_vader],
        "vader_neg": [v['neg'] for v in aug_vader],
        "vader_neu": [v['neu'] for v in aug_vader]
    })
    for i, cls in enumerate(EMOTION_CLASSES):
        col_name = "is_" + cls.split("/")[0].strip().lower().replace(" ", "_")
        aug_df[col_name] = aug_vectors[:, i]
        
    df_combined = pd.concat([df_labeled, aug_df], ignore_index=True)
    
    # 3. Feature Extraction
    print("\n[Step 3/6] Applying Negation Scope Tagging & (1, 3) N-Gram Feature Extraction...")
    df_combined['negated_text'] = df_combined['clean_text'].apply(apply_negation_tagging)
    
    X_raw_texts = df_combined['clean_text'].values
    X_neg_texts = df_combined['negated_text'].values
    
    vader_cols = ['vader_compound', 'vader_pos', 'vader_neg', 'vader_neu']
    X_vader = np.array(df_combined[vader_cols].values, dtype=np.float32)
    
    # Multi-Label Binary Target Matrix (N x 5)
    indicator_cols = ["is_" + cls.split("/")[0].strip().lower().replace(" ", "_") for cls in EMOTION_CLASSES]
    Y_multilabel = np.array(df_combined[indicator_cols].values, dtype=int)
    
    # Train / Test Split
    indices = np.arange(len(df_combined))
    idx_train, idx_test = train_test_split(indices, test_size=0.20, random_state=42)
    
    # TF-IDF Vectorization with N-Grams (1, 3) & Negation Preservation
    vectorizer_multilabel = TfidfVectorizer(
        ngram_range=(1, 3),
        min_df=2,
        max_features=25000,
        sublinear_tf=True,
        stop_words=SAFE_STOP_WORDS
    )
    X_tfidf_train = vectorizer_multilabel.fit_transform(X_neg_texts[idx_train])
    X_tfidf_test = vectorizer_multilabel.transform(X_neg_texts[idx_test])
    
    # Standardize VADER Polarity Features
    scaler = StandardScaler()
    X_vader_train_scaled = scaler.fit_transform(X_vader[idx_train])
    X_vader_test_scaled = scaler.transform(X_vader[idx_test])
    
    X_train_full = sp.hstack([X_tfidf_train, X_vader_train_scaled], format='csr')
    X_test_full = sp.hstack([X_tfidf_test, X_vader_test_scaled], format='csr')
    
    Y_train = Y_multilabel[idx_train]
    Y_test = Y_multilabel[idx_test]
    
    print(f" Feature Matrix: Train {X_train_full.shape}, Test {X_test_full.shape}")
    print(f" Vocab Size (1-3 n-grams with negations): {len(vectorizer_multilabel.vocabulary_):,}")

    # 4. Architecture Benchmarking (Baseline vs Multi-Label Negation Models)
    print("\n[Step 4/6] Training & Benchmarking Candidate Multi-Label Classifiers...")
    
    candidate_architectures = {
        "Multi-Label Logistic Regression": OneVsRestClassifier(
            LogisticRegression(max_iter=2000, C=2.0, class_weight='balanced', random_state=42, n_jobs=-1)
        ),
        "Multi-Label LightGBM": OneVsRestClassifier(
            LGBMClassifier(n_estimators=150, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1)
        )
    }
    
    benchmark_records = []
    trained_models = {}
    
    for name, clf in candidate_architectures.items():
        m_start = time.time()
        print(f" Training {name:35s} ...", end="", flush=True)
        clf.fit(X_train_full, Y_train)
        preds = clf.predict(X_test_full)
        elapsed = time.time() - m_start
        
        macro_f1 = f1_score(Y_test, preds, average='macro', zero_division=0)
        micro_f1 = f1_score(Y_test, preds, average='micro', zero_division=0)
        weighted_f1 = f1_score(Y_test, preds, average='weighted', zero_division=0)
        h_loss = hamming_loss(Y_test, preds)
        j_score = jaccard_score(Y_test, preds, average='samples', zero_division=0)
        exact_acc = accuracy_score(Y_test, preds)
        
        print(f" Done in {elapsed:.2f}s | Macro F1: {macro_f1:.4f} | Hamming Loss: {h_loss:.4f} | Jaccard: {j_score:.4f}")
        
        benchmark_records.append({
            "Architecture": "Negation-Aware Multi-Label",
            "Model": name,
            "Macro F1": macro_f1,
            "Micro F1": micro_f1,
            "Weighted F1": weighted_f1,
            "Hamming Loss": h_loss,
            "Jaccard Score": j_score,
            "Exact Match Ratio": exact_acc,
            "Training Time (s)": elapsed
        })
        trained_models[name] = clf
        
    benchmark_df = pd.DataFrame(benchmark_records)
    plot_multi_label_metrics(benchmark_df)
    
    # 5. Best Model Evaluation & Per-Class Breakdown
    best_name = "Multi-Label Logistic Regression"
    best_clf = trained_models[best_name]
    
    print("\n" + "=" * 80)
    print(f" PRODUCTION MULTI-LABEL MODEL: {best_name.upper()}")
    print("=" * 80)
    
    Y_pred_best = best_clf.predict(X_test_full)
    print("\nPer-Emotion Multi-Label Classification Report:\n")
    print(classification_report(Y_test, Y_pred_best, target_names=EMOTION_CLASSES, digits=4, zero_division=0))
    
    # 6. Pipeline Serialization & Metadata
    print("\n[Step 6/6] Serializing MultiLabelEmotionPipeline Artifacts...")
    pipeline = MultiLabelEmotionPipeline(
        vectorizer=vectorizer_multilabel,
        scaler=scaler,
        classifier=best_clf,
        emotion_classes=EMOTION_CLASSES
    )
    
    pipeline_path = str(PROJECT_ROOT / "models" / "emotion_pipeline.joblib")
    joblib.dump(pipeline, pipeline_path)
    print(f" [+] Saved production multi-label model pipeline to: '{pipeline_path}'")
    
    metadata = {
        "architecture": "Negation-Aware Multi-Label Emotion Classifier",
        "best_model": best_name,
        "ngram_range": [1, 3],
        "vocab_size": len(vectorizer_multilabel.vocabulary_),
        "sample_size": len(df_combined),
        "emotion_classes": EMOTION_CLASSES,
        "benchmark": benchmark_df.to_dict(orient="records"),
        "training_time_seconds": time.time() - start_time
    }
    
    metadata_path = str(PROJECT_ROOT / "models" / "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f" [+] Exported updated model metadata to: '{metadata_path}'")
    print(f"\nPipeline training completed successfully in {time.time() - start_time:.2f} seconds!")
    print("=" * 80)


if __name__ == '__main__':
    default_dataset = str(PROJECT_ROOT / "data" / "processed" / "twcs_cleaned.csv")
    parser = argparse.ArgumentParser(description="Train Negation-Aware Multi-Label Emotion Detection Models")
    parser.add_argument("--input", "-i", type=str, default=default_dataset, help="Path to clean CSV")
    parser.add_argument("--sample-size", "-s", type=int, default=0, help="Sample size for training (0 for all)")
    args = parser.parse_args()
    
    train_and_evaluate(args.input, args.sample_size)
