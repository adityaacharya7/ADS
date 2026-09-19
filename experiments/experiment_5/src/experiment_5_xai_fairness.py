"""
Experiment 5: Explainable AI (XAI) with SHAP & LIME and Fairness Auditing with Fairlearn
Course: Applied Data Science (ADS)
Dataset: Adult Census Income (Demographic & Socioeconomic Benchmark)

Implements the complete Responsible Machine Learning lifecycle:
1. Benchmark Tabular Data Ingestion with Sensitive Demographic Attributes (Sex & Race)
2. High-Performance Machine Learning Modeling (LightGBM Champion & Random Forest Baseline)
3. Global and Local Model Interpretability via SHAP (Summary, Beeswarm, Dependence, Waterfall)
4. Local Surrogate Explanations via LIME (lime.lime_tabular) with Cross-XAI Concordance Analysis
5. Comprehensive Algorithmic Bias Auditing via Fairlearn (Demographic Parity, Equalized Odds, Disparate Impact)
6. Tri-Modal Bias Mitigation Strategies:
   - Pre-processing: Sample Reweighting across Demographic Intersections
   - In-processing: Fairlearn ExponentiatedGradient with Fairness Constraints
   - Post-processing: Fairlearn ThresholdOptimizer for Calibrated Decision Boundaries
7. Fairness-Performance Pareto Trade-off Benchmarking & Artifact Serialization
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, confusion_matrix
)
from lightgbm import LGBMClassifier

import shap
from lime import lime_tabular

import fairlearn
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    equalized_odds_ratio,
    selection_rate,
    true_positive_rate,
    false_positive_rate
)
from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds
from fairlearn.postprocessing import ThresholdOptimizer

# Path configurations
EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = EXPERIMENT_DIR.parent.parent
PLOTS_DIR = EXPERIMENT_DIR / "plots"
MODELS_DIR = EXPERIMENT_DIR / "models"
REPORTS_DIR = EXPERIMENT_DIR / "reports"
DATA_DIR = EXPERIMENT_DIR / "data"

for d in [PLOTS_DIR, MODELS_DIR, REPORTS_DIR, DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})


# ------------------------------------------------------------------------------
# 1. DATA INGESTION & SENSITIVE ATTRIBUTE PREPARATION
# ------------------------------------------------------------------------------

def load_and_prepare_census_data() -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, Dict[str, Any]]:
    """
    Loads and prepares the Adult Census Income dataset with explicit sensitive features.
    Standardizes feature names (removing spaces) and preserves categorical mappings.
    """
    print("\n" + "=" * 70)
    print(" [STAGE 1/6] INGESTING BENCHMARK DATASET & SENSITIVE DEMOGRAPHICS")
    print("=" * 70)

    # Ingest numerical and display representations from SHAP benchmark repository
    X_raw, y_raw = shap.datasets.adult()
    X_disp, _ = shap.datasets.adult(display=True)

    # Standardize column names (replacing spaces and hyphens with underscores)
    col_mapping = {col: col.replace(" ", "_").replace("-", "_") for col in X_raw.columns}
    X = X_raw.rename(columns=col_mapping).copy()
    X_disp = X_disp.rename(columns=col_mapping).copy()

    # Target: 1 for income >$50k, 0 for <=$50k
    y = y_raw.astype(int)

    # Extract sensitive features
    # Sex: 0 = Female, 1 = Male
    sex_labels = {0: "Female", 1: "Male"}
    race_labels = {
        0: "Amer-Indian-Eskimo",
        1: "Asian-Pac-Islander",
        2: "Black",
        3: "Other",
        4: "White"
    }

    metadata = {
        "dataset_name": "Adult Census Income (UCI ML / US Census Bureau)",
        "total_instances": len(X),
        "feature_names": list(X.columns),
        "target_name": "Income_Over_50K",
        "class_distribution": {
            "<=50K (0)": int((y == 0).sum()),
            ">50K (1)": int((y == 1).sum()),
            "positive_rate": float(y.mean())
        },
        "sensitive_attributes": {
            "Sex": {
                "categories": sex_labels,
                "counts": {sex_labels[k]: int((X["Sex"] == k).sum()) for k in sex_labels}
            },
            "Race": {
                "categories": race_labels,
                "counts": {race_labels[k]: int((X["Race"] == k).sum()) for k in race_labels}
            }
        }
    }

    # Cache dataset locally for offline reproducibility
    cached_csv = DATA_DIR / "census_income_clean.csv"
    if not cached_csv.exists():
        df_save = X.copy()
        df_save["Income_Over_50K"] = y
        df_save.to_csv(cached_csv, index=False)
        print(f"[+] Dataset cached locally to: {cached_csv.name}")

    print(f"[*] Total Records   : {len(X):,} rows, {X.shape[1]} features")
    print(f"[*] Target Class    : <=50K: {(y == 0).sum():,} ({(y == 0).mean()*100:.1f}%), >50K: {(y == 1).sum():,} ({(y == 1).mean()*100:.1f}%)")
    print(f"[*] Sensitive (Sex) : Male: {(X['Sex'] == 1).sum():,} ({(X['Sex'] == 1).mean()*100:.1f}%), Female: {(X['Sex'] == 0).sum():,} ({(X['Sex'] == 0).mean()*100:.1f}%)")
    print(f"[*] Sensitive (Race): White: {(X['Race'] == 4).sum():,} ({(X['Race'] == 4).mean()*100:.1f}%), Non-White: {(X['Race'] != 4).sum():,} ({(X['Race'] != 4).mean()*100:.1f}%)")

    return X, y, X_disp, metadata


# ------------------------------------------------------------------------------
# 2. MODEL TRAINING & BASELINE BENCHMARKING
# ------------------------------------------------------------------------------

def train_unmitigated_models(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray
) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Trains baseline Random Forest and champion LightGBM classifiers.
    Evaluates standard accuracy, precision, recall, F1, and ROC-AUC metrics.
    """
    print("\n" + "=" * 70)
    print(" [STAGE 2/6] TRAINING CLASSIFICATION BENCHMARK & SELECTING CHAMPION")
    print("=" * 70)

    # 1. Random Forest Baseline
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_probs = rf.predict_proba(X_test)[:, 1]

    rf_metrics = {
        "model": "Random Forest Baseline",
        "accuracy": float(accuracy_score(y_test, rf_preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, rf_preds)),
        "precision": float(precision_score(y_test, rf_preds, zero_division=0)),
        "recall": float(recall_score(y_test, rf_preds)),
        "f1": float(f1_score(y_test, rf_preds)),
        "roc_auc": float(roc_auc_score(y_test, rf_probs))
    }

    # 2. LightGBM Champion Classifier
    lgb = LGBMClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.08,
        num_leaves=31,
        random_state=42,
        verbose=-1
    )
    lgb.fit(X_train, y_train)
    lgb_preds = lgb.predict(X_test)
    lgb_probs = lgb.predict_proba(X_test)[:, 1]

    lgb_metrics = {
        "model": "Champion LightGBM (Unmitigated)",
        "accuracy": float(accuracy_score(y_test, lgb_preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, lgb_preds)),
        "precision": float(precision_score(y_test, lgb_preds, zero_division=0)),
        "recall": float(recall_score(y_test, lgb_preds)),
        "f1": float(f1_score(y_test, lgb_preds)),
        "roc_auc": float(roc_auc_score(y_test, lgb_probs))
    }

    print(f"[*] Random Forest -> Acc: {rf_metrics['accuracy']:.4f} | F1: {rf_metrics['f1']:.4f} | AUC: {rf_metrics['roc_auc']:.4f}")
    print(f"[*] Champion LGBM -> Acc: {lgb_metrics['accuracy']:.4f} | F1: {lgb_metrics['f1']:.4f} | AUC: {lgb_metrics['roc_auc']:.4f}")

    benchmark_summary = {
        "random_forest": rf_metrics,
        "champion_lightgbm": lgb_metrics
    }

    # Save champion model
    joblib.dump(lgb, MODELS_DIR / "unmitigated_champion_model.joblib")
    print(f"[+] Champion model saved to: {MODELS_DIR / 'unmitigated_champion_model.joblib'}")

    return lgb, rf, benchmark_summary


# ------------------------------------------------------------------------------
# 3. EXPLAINABILITY WITH SHAP (GLOBAL & LOCAL)
# ------------------------------------------------------------------------------

def run_shap_explainability(
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    feature_names: List[str]
) -> Dict[str, Any]:
    """
    Executes SHAP game-theoretic explainability suite:
    - Global feature importance (Mean |SHAP|)
    - Beeswarm summary plot (Directional impact & feature value dispersion)
    - SHAP dependence & interaction plot
    - Local waterfall explanation for representative prediction instances
    """
    print("\n" + "=" * 70)
    print(" [STAGE 3/6] GENERATING SHAP GLOBAL & LOCAL EXPLANATIONS")
    print("=" * 70)

    # Fit TreeExplainer
    explainer = shap.TreeExplainer(model)
    # Use 800 test instances for dense, high-fidelity visualizations
    sample_size = min(800, len(X_test))
    X_sample = X_test.iloc[:sample_size].copy()
    
    print(f"[*] Calculating SHAP values for {sample_size} validation samples...")
    shap_values = explainer(X_sample)

    # Extract 2D array of shap values for binary positive class
    if len(shap_values.shape) == 3:
        # Some versions output (samples, features, classes)
        shap_vals_arr = shap_values.values[:, :, 1]
    else:
        shap_vals_arr = shap_values.values

    # Compute global mean absolute SHAP values
    mean_abs_shap = np.mean(np.abs(shap_vals_arr), axis=0)
    feat_importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Mean_Absolute_SHAP": mean_abs_shap
    }).sort_values(by="Mean_Absolute_SHAP", ascending=False).reset_index(drop=True)

    # 1. Global Feature Importance Bar Plot
    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = sns.color_palette("Blues_r", n_colors=len(feat_importance_df))
    bars = ax.barh(
        feat_importance_df["Feature"][::-1],
        feat_importance_df["Mean_Absolute_SHAP"][::-1],
        color=colors[::-1],
        edgecolor="#1E3A8A",
        height=0.65
    )
    ax.set_xlabel("Mean Absolute SHAP Value (Impact on Model Output Magnitude)", fontsize=11, fontweight="bold")
    ax.set_title("Global Feature Importance (SHAP TreeExplainer Ranking)", fontsize=13, fontweight="bold", pad=12)
    for bar in bars:
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.3f}",
            va="center", ha="left", fontsize=9, color="#1E293B", fontweight="bold"
        )
    ax.set_xlim(0, max(feat_importance_df["Mean_Absolute_SHAP"]) * 1.15)
    plt.tight_layout()
    bar_plot_path = PLOTS_DIR / "exp5_shap_summary_bar.png"
    plt.savefig(bar_plot_path)
    plt.close()
    print(f"[+] Saved SHAP summary bar chart to: {bar_plot_path.name}")

    # 2. SHAP Beeswarm Plot
    plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_values, X_sample, show=False, max_display=12)
    plt.title("SHAP Beeswarm Summary Plot (Feature Values vs. Impact on Log-Odds)", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    beeswarm_path = PLOTS_DIR / "exp5_shap_beeswarm.png"
    plt.savefig(beeswarm_path)
    plt.close()
    print(f"[+] Saved SHAP beeswarm plot to: {beeswarm_path.name}")

    # 3. SHAP Dependence Plot (Age vs. Hours_per_week)
    plt.figure(figsize=(8.5, 5.5))
    shap.dependence_plot(
        "Age",
        shap_vals_arr,
        X_sample,
        interaction_index="Hours_per_week",
        show=False
    )
    plt.title("SHAP Dependence Plot: Age Non-Linearity & Interaction with Hours per Week", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    dependence_path = PLOTS_DIR / "exp5_shap_dependence.png"
    plt.savefig(dependence_path)
    plt.close()
    print(f"[+] Saved SHAP dependence plot to: {dependence_path.name}")

    # 4. SHAP Waterfall Plot (Local Sample Explanation)
    # Pick a high-confidence prediction sample
    sample_idx = 0
    plt.figure(figsize=(8.5, 6))
    if hasattr(shap.plots, "waterfall"):
        if len(shap_values.shape) == 3:
            shap.plots.waterfall(shap_values[sample_idx, :, 1], show=False)
        else:
            shap.plots.waterfall(shap_values[sample_idx], show=False)
    else:
        shap.summary_plot(shap_vals_arr[sample_idx:sample_idx+1], X_sample.iloc[sample_idx:sample_idx+1], plot_type="bar", show=False)
    plt.title(f"SHAP Waterfall Attribution: Local Explanation for Sample #{sample_idx}", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    waterfall_path = PLOTS_DIR / "exp5_shap_waterfall.png"
    plt.savefig(waterfall_path)
    plt.close()
    print(f"[+] Saved SHAP waterfall plot to: {waterfall_path.name}")

    shap_results = {
        "top_features_ranked": feat_importance_df.to_dict(orient="records"),
        "sample_analyzed": sample_size,
        "sample_0_actual": int(y_test[0]),
        "sample_0_predicted": int(model.predict(X_test.iloc[[0]])[0]),
        "sample_0_prob": float(model.predict_proba(X_test.iloc[[0]])[0, 1])
    }

    return shap_results


# ------------------------------------------------------------------------------
# 4. EXPLAINABILITY WITH LIME (LOCAL SURROGATE MODELS)
# ------------------------------------------------------------------------------

def run_lime_explainability(
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    feature_names: List[str]
) -> Dict[str, Any]:
    """
    Executes LIME local surrogate interpretability suite using lime.lime_tabular.
    Explains individual positive, negative, and borderline predictions.
    Computes cross-method concordance with SHAP.
    """
    print("\n" + "=" * 70)
    print(" [STAGE 4/6] GENERATING LIME LOCAL EXPLANATIONS & CONCORDANCE AUDIT")
    print("=" * 70)

    # Initialize LimeTabularExplainer
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_names,
        class_names=["<=50K", ">50K"],
        mode="classification",
        random_state=42
    )

    # Select representative samples:
    # 1. High-confidence positive (>50K)
    # 2. High-confidence negative (<=50K)
    # 3. Borderline prediction (~0.50 probability)
    test_probs = model.predict_proba(X_test)[:, 1]
    idx_high_pos = int(np.argmax(test_probs))
    idx_high_neg = int(np.argmin(test_probs))
    idx_borderline = int(np.argmin(np.abs(test_probs - 0.50)))

    samples_to_explain = [
        ("High-Confidence Positive (>50K)", idx_high_pos),
        ("High-Confidence Negative (<=50K)", idx_high_neg),
        ("Borderline Prediction (p~0.50)", idx_borderline)
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=False)
    lime_summaries = []

    for ax, (title, s_idx) in zip(axes, samples_to_explain):
        exp = explainer.explain_instance(
            data_row=X_test.iloc[s_idx].values,
            predict_fn=model.predict_proba,
            num_features=6
        )
        exp_list = exp.as_list()
        
        rules = [item[0] for item in exp_list][::-1]
        weights = [item[1] for item in exp_list][::-1]
        colors = ["#10B981" if w >= 0 else "#EF4444" for w in weights]

        ax.barh(range(len(rules)), weights, color=colors, edgecolor="#1E293B", height=0.6)
        ax.set_yticks(range(len(rules)))
        ax.set_yticklabels(rules, fontsize=9, fontweight="medium")
        ax.axvline(0, color="#64748B", linestyle="--", linewidth=1)
        prob_val = test_probs[s_idx]
        actual_val = "<=50K" if y_test[s_idx] == 0 else ">50K"
        ax.set_title(f"{title}\nP(>50K)={prob_val:.2f} | Actual: {actual_val}", fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("LIME Feature Weight", fontsize=10, fontweight="bold")

        lime_summaries.append({
            "profile": title,
            "sample_index": s_idx,
            "predicted_prob": float(prob_val),
            "actual_label": actual_val,
            "top_rules": exp_list
        })

    plt.suptitle("LIME Local Surrogate Explanations Across Distinct Prediction Profiles", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    lime_plot_path = PLOTS_DIR / "exp5_lime_local_explanations.png"
    plt.savefig(lime_plot_path)
    plt.close()
    print(f"[+] Saved LIME local explanations figure to: {lime_plot_path.name}")

    # 5. SHAP vs LIME Concordance Comparison on Sample #0
    # Comparing directional feature attributions on sample #0
    sample_0 = X_test.iloc[0]
    tree_explainer = shap.TreeExplainer(model)
    shap_0 = tree_explainer(X_test.iloc[:1])
    if len(shap_0.shape) == 3:
        shap_0_vals = shap_0.values[0, :, 1]
    else:
        shap_0_vals = shap_0.values[0]

    lime_0 = explainer.explain_instance(sample_0.values, model.predict_proba, num_features=len(feature_names))
    lime_dict = dict(lime_0.as_list())

    # Map LIME rules back to feature names
    lime_feat_weights = {}
    for feat in feature_names:
        matched_weight = 0.0
        for rule_str, w in lime_dict.items():
            if feat in rule_str:
                matched_weight = w
                break
        lime_feat_weights[feat] = matched_weight

    comp_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP_Attribution": shap_0_vals,
        "LIME_Attribution": [lime_feat_weights[f] for f in feature_names]
    }).sort_values(by="SHAP_Attribution", key=abs, ascending=False).head(7).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x_pos = np.arange(len(comp_df))
    width = 0.35

    ax.bar(x_pos - width/2, comp_df["SHAP_Attribution"], width, label="SHAP (Shapley Additive Attribution)", color="#3B82F6", edgecolor="#1E3A8A")
    ax.bar(x_pos + width/2, comp_df["LIME_Attribution"], width, label="LIME (Local Surrogate Attribution)", color="#10B981", edgecolor="#065F46")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(comp_df["Feature"], rotation=25, ha="right", fontsize=9, fontweight="bold")
    ax.axhline(0, color="#64748B", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Attribution Weight", fontsize=11, fontweight="bold")
    ax.set_title(f"Cross-XAI Concordance: SHAP vs. LIME Feature Attribution (Sample #0)\nModel P(>50K) = {test_probs[0]:.2f}", fontsize=12, fontweight="bold", pad=12)
    ax.legend(frameon=True, facecolor="white", edgecolor="#CBD5E1")
    plt.tight_layout()
    comp_plot_path = PLOTS_DIR / "exp5_xai_shap_vs_lime_comparison.png"
    plt.savefig(comp_plot_path)
    plt.close()
    print(f"[+] Saved SHAP vs. LIME concordance chart to: {comp_plot_path.name}")

    return {
        "profiles": lime_summaries,
        "sample_0_comparison": comp_df.to_dict(orient="records")
    }


# ------------------------------------------------------------------------------
# 5. FAIRNESS AUDIT WITH FAIRLEARN
# ------------------------------------------------------------------------------

def audit_model_fairness(
    model: Any,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    sex_test: pd.Series,
    race_test: pd.Series,
    model_name: str = "Champion LightGBM (Unmitigated)",
    is_post_processed: bool = False
) -> Dict[str, Any]:
    """
    Computes rigorous demographic parity, equalized odds, and sub-group error metrics
    across sensitive attributes (Sex and Race) using Fairlearn MetricFrame.
    """
    if is_post_processed:
        preds = model.predict(X_test, sensitive_features=sex_test)
        probs = preds.astype(float)
    else:
        preds = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)[:, 1]
        else:
            probs = preds.astype(float)

    # Map numeric sensitive codes to readable names
    sex_labels = sex_test.map({0: "Female", 1: "Male"})
    race_labels = race_test.map({
        0: "Amer-Indian-Eskimo",
        1: "Asian-Pac-Islander",
        2: "Black",
        3: "Other",
        4: "White"
    })
    race_binary = race_test.map(lambda r: "White" if r == 4 else "Non-White")

    # Define core metric suite for MetricFrame
    metric_funcs = {
        "accuracy": accuracy_score,
        "selection_rate": selection_rate,
        "true_positive_rate": true_positive_rate,
        "false_positive_rate": false_positive_rate,
        "balanced_accuracy": balanced_accuracy_score,
        "f1_score": f1_score
    }

    # 1. Audit across Sex
    mf_sex = MetricFrame(
        metrics=metric_funcs,
        y_true=y_test,
        y_pred=preds,
        sensitive_features=sex_labels
    )

    dp_diff_sex = float(demographic_parity_difference(y_test, preds, sensitive_features=sex_labels))
    dp_ratio_sex = float(demographic_parity_ratio(y_test, preds, sensitive_features=sex_labels))
    eo_diff_sex = float(equalized_odds_difference(y_test, preds, sensitive_features=sex_labels))
    eo_ratio_sex = float(equalized_odds_ratio(y_test, preds, sensitive_features=sex_labels))

    # 2. Audit across Race
    mf_race = MetricFrame(
        metrics=metric_funcs,
        y_true=y_test,
        y_pred=preds,
        sensitive_features=race_labels
    )

    dp_diff_race = float(demographic_parity_difference(y_test, preds, sensitive_features=race_labels))
    dp_ratio_race = float(demographic_parity_ratio(y_test, preds, sensitive_features=race_labels))
    eo_diff_race = float(equalized_odds_difference(y_test, preds, sensitive_features=race_labels))
    eo_ratio_race = float(equalized_odds_ratio(y_test, preds, sensitive_features=race_labels))

    # 3. Audit across Binary Race (White vs Non-White)
    mf_race_bin = MetricFrame(
        metrics=metric_funcs,
        y_true=y_test,
        y_pred=preds,
        sensitive_features=race_binary
    )

    audit_result = {
        "model_name": model_name,
        "overall": {
            "accuracy": float(accuracy_score(y_test, preds)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, preds)),
            "f1_score": float(f1_score(y_test, preds)),
            "selection_rate": float(selection_rate(y_test, preds)),
            "roc_auc": float(roc_auc_score(y_test, probs)) if (hasattr(model, "predict_proba") and not is_post_processed) else None
        },
        "sex_audit": {
            "by_group": mf_sex.by_group.to_dict(orient="index"),
            "demographic_parity_difference": dp_diff_sex,
            "demographic_parity_ratio": dp_ratio_sex,
            "equalized_odds_difference": eo_diff_sex,
            "equalized_odds_ratio": eo_ratio_sex
        },
        "race_audit": {
            "by_group": mf_race.by_group.to_dict(orient="index"),
            "demographic_parity_difference": dp_diff_race,
            "demographic_parity_ratio": dp_ratio_race,
            "equalized_odds_difference": eo_diff_race,
            "equalized_odds_ratio": eo_ratio_race
        },
        "race_binary_audit": {
            "by_group": mf_race_bin.by_group.to_dict(orient="index"),
            "demographic_parity_difference": float(demographic_parity_difference(y_test, preds, sensitive_features=race_binary)),
            "equalized_odds_difference": float(equalized_odds_difference(y_test, preds, sensitive_features=race_binary))
        }
    }

    return audit_result


# ------------------------------------------------------------------------------
# 6. BIAS MITIGATION STRATEGIES & PARETO TRADEOFF ANALYSIS
# ------------------------------------------------------------------------------

def run_bias_mitigation(
    unmitigated_model: Any,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    sex_train: pd.Series,
    sex_test: pd.Series,
    race_test: pd.Series
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Implements 3 comprehensive mitigation techniques:
    1. Pre-processing: Sample Reweighting across demographic groups
    2. In-processing: Fairlearn ExponentiatedGradient with EqualizedOdds constraints
    3. Post-processing: Fairlearn ThresholdOptimizer calibrating group thresholds
    Evaluates fairness-accuracy trade-offs and generates Pareto plots.
    """
    print("\n" + "=" * 70)
    print(" [STAGE 5/6] IMPLEMENTING TRI-MODAL BIAS MITIGATION & BENCHMARKING")
    print("=" * 70)

    # 1. Unmitigated Baseline Audit
    baseline_audit = audit_model_fairness(
        unmitigated_model, X_test, y_test, sex_test, race_test,
        model_name="1. Baseline Unmitigated (LightGBM)"
    )

    # --------------------------------------------------------------------------
    # A. PRE-PROCESSING: Sample Reweighting
    # --------------------------------------------------------------------------
    print("[*] Training Pre-processing Mitigation: Demographic Sample Reweighting...")
    # Calculate sample weights: Weight = P(Y) / P(Y | Sensitive)
    train_df = pd.DataFrame({"y": y_train, "sex": sex_train})
    p_y = train_df["y"].value_counts(normalize=True)
    p_y_given_sex = train_df.groupby(["sex", "y"]).size() / train_df.groupby("sex").size()
    
    weights = []
    for _, row in train_df.iterrows():
        w = p_y[row["y"]] / p_y_given_sex[row["sex"], row["y"]]
        weights.append(w)
    sample_weights = np.array(weights)

    model_pre = LGBMClassifier(
        n_estimators=150, max_depth=6, learning_rate=0.08,
        num_leaves=31, random_state=42, verbose=-1
    )
    model_pre.fit(X_train, y_train, sample_weight=sample_weights)
    pre_audit = audit_model_fairness(
        model_pre, X_test, y_test, sex_test, race_test,
        model_name="2. Pre-Processing (Reweighted)"
    )
    joblib.dump(model_pre, MODELS_DIR / "mitigated_reweighted_model.joblib")

    # --------------------------------------------------------------------------
    # B. IN-PROCESSING: Exponentiated Gradient with Equalized Odds
    # --------------------------------------------------------------------------
    print("[*] Training In-processing Mitigation: Fairlearn ExponentiatedGradient...")
    # Base estimator for reductions
    in_base = LGBMClassifier(n_estimators=30, max_depth=4, learning_rate=0.1, random_state=42, verbose=-1)
    mitigated_in = ExponentiatedGradient(
        estimator=in_base,
        constraints=EqualizedOdds(),
        max_iter=15
    )
    mitigated_in.fit(X_train, y_train, sensitive_features=sex_train)
    in_audit = audit_model_fairness(
        mitigated_in, X_test, y_test, sex_test, race_test,
        model_name="3. In-Processing (ExponentiatedGradient)"
    )
    joblib.dump(mitigated_in, MODELS_DIR / "mitigated_exponentiated_model.joblib")

    # --------------------------------------------------------------------------
    # C. POST-PROCESSING: Threshold Optimizer (Group Threshold Calibration)
    # --------------------------------------------------------------------------
    print("[*] Training Post-processing Mitigation: Fairlearn ThresholdOptimizer...")
    mitigated_post = ThresholdOptimizer(
        estimator=unmitigated_model,
        constraints="equalized_odds",
        prefit=True,
        predict_method="predict_proba"
    )
    mitigated_post.fit(X_train, y_train, sensitive_features=sex_train)
    post_audit = audit_model_fairness(
        mitigated_post, X_test, y_test, sex_test, race_test,
        model_name="4. Post-Processing (ThresholdOptimizer)",
        is_post_processed=True
    )
    joblib.dump(mitigated_post, MODELS_DIR / "mitigated_threshold_model.joblib")

    # --------------------------------------------------------------------------
    # Comparative Benchmark Summary Table
    # --------------------------------------------------------------------------
    all_audits = [baseline_audit, pre_audit, in_audit, post_audit]
    benchmark_rows = []
    for a in all_audits:
        benchmark_rows.append({
            "Model Pipeline": a["model_name"],
            "Accuracy": a["overall"]["accuracy"],
            "Balanced Accuracy": a["overall"]["balanced_accuracy"],
            "F1-Score": a["overall"]["f1_score"],
            "Demographic Parity Diff (Sex)": a["sex_audit"]["demographic_parity_difference"],
            "Equalized Odds Diff (Sex)": a["sex_audit"]["equalized_odds_difference"],
            "Disparate Impact Ratio (Sex)": a["sex_audit"]["demographic_parity_ratio"],
            "Demographic Parity Diff (Race)": a["race_audit"]["demographic_parity_difference"],
            "Equalized Odds Diff (Race)": a["race_audit"]["equalized_odds_difference"]
        })

    benchmark_df = pd.DataFrame(benchmark_rows)
    benchmark_df.to_csv(REPORTS_DIR / "bias_mitigation_benchmark.csv", index=False)
    print(f"[+] Saved mitigation benchmark table to: {REPORTS_DIR / 'bias_mitigation_benchmark.csv'}")

    # Print summary to console
    print("\n" + "=" * 70)
    print(" BIAS MITIGATION BENCHMARK MATRIX (PERFORMANCE VS. FAIRNESS)")
    print("=" * 70)
    print(benchmark_df.to_string(index=False))

    # --------------------------------------------------------------------------
    # 6. VISUALIZATION 1: Subgroup Disparity Bar Charts (Baseline Audit)
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Sex Subgroup Disparities
    sex_groups = ["Female", "Male"]
    sex_sr = [baseline_audit["sex_audit"]["by_group"][g]["selection_rate"] for g in sex_groups]
    sex_tpr = [baseline_audit["sex_audit"]["by_group"][g]["true_positive_rate"] for g in sex_groups]
    sex_acc = [baseline_audit["sex_audit"]["by_group"][g]["accuracy"] for g in sex_groups]

    x = np.arange(len(sex_groups))
    w = 0.25
    axes[0].bar(x - w, sex_sr, w, label="Selection Rate P(>50K)", color="#3B82F6", edgecolor="#1E3A8A")
    axes[0].bar(x, sex_tpr, w, label="True Positive Rate (TPR)", color="#10B981", edgecolor="#065F46")
    axes[0].bar(x + w, sex_acc, w, label="Accuracy", color="#6366F1", edgecolor="#3730A3")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(sex_groups, fontsize=11, fontweight="bold")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title(f"Fairness Disparity by Gender (Sex)\nDP Diff = {baseline_audit['sex_audit']['demographic_parity_difference']:.3f} | EO Diff = {baseline_audit['sex_audit']['equalized_odds_difference']:.3f}", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Metric Value", fontsize=10, fontweight="bold")
    axes[0].legend(frameon=True, facecolor="white")

    # Race Subgroup Disparities
    race_groups = ["Amer-Ind", "Asian", "Black", "Other", "White"]
    race_sr = [baseline_audit["race_audit"]["by_group"][k]["selection_rate"] for k in baseline_audit["race_audit"]["by_group"]]
    race_tpr = [baseline_audit["race_audit"]["by_group"][k]["true_positive_rate"] for k in baseline_audit["race_audit"]["by_group"]]

    x_r = np.arange(len(race_groups))
    w_r = 0.35
    axes[1].bar(x_r - w_r/2, race_sr, w_r, label="Selection Rate P(>50K)", color="#3B82F6", edgecolor="#1E3A8A")
    axes[1].bar(x_r + w_r/2, race_tpr, w_r, label="True Positive Rate (TPR)", color="#10B981", edgecolor="#065F46")
    axes[1].set_xticks(x_r)
    axes[1].set_xticklabels(race_groups, fontsize=10, fontweight="bold")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title(f"Fairness Disparity by Race\nDP Diff = {baseline_audit['race_audit']['demographic_parity_difference']:.3f} | EO Diff = {baseline_audit['race_audit']['equalized_odds_difference']:.3f}", fontsize=11, fontweight="bold")
    axes[1].legend(frameon=True, facecolor="white")

    plt.suptitle("Fairlearn Baseline Bias Audit: Disparity Across Sensitive Demographic Features", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    disparity_plot_path = PLOTS_DIR / "exp5_fairness_audit_disparity.png"
    plt.savefig(disparity_plot_path)
    plt.close()
    print(f"[+] Saved fairness audit disparity plot to: {disparity_plot_path.name}")

    # --------------------------------------------------------------------------
    # 7. VISUALIZATION 2: Fairness-Accuracy Pareto Tradeoff Plot
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    colors = ["#EF4444", "#F59E0B", "#3B82F6", "#10B981"]
    markers = ["o", "s", "^", "D"]

    for i, row in benchmark_df.iterrows():
        ax.scatter(
            row["Equalized Odds Diff (Sex)"],
            row["Accuracy"],
            color=colors[i],
            marker=markers[i],
            s=180,
            label=row["Model Pipeline"],
            edgecolor="#1E293B",
            linewidth=1.5,
            zorder=4
        )
        ax.annotate(
            row["Model Pipeline"].split("(")[0].strip(),
            (row["Equalized Odds Diff (Sex)"] + 0.005, row["Accuracy"] + 0.001),
            fontsize=9, fontweight="bold", color="#1E293B"
        )

    ax.set_xlabel("Equalized Odds Difference (Sex) [Lower is Fairer]", fontsize=11, fontweight="bold")
    ax.set_ylabel("Classification Accuracy [Higher is Better]", fontsize=11, fontweight="bold")
    ax.set_title("Fairness vs. Performance Pareto Frontier (Mitigation Comparison)", fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True, facecolor="white", loc="lower left")
    plt.tight_layout()
    tradeoff_plot_path = PLOTS_DIR / "exp5_fairness_mitigation_tradeoff.png"
    plt.savefig(tradeoff_plot_path)
    plt.close()
    print(f"[+] Saved fairness-performance Pareto tradeoff plot to: {tradeoff_plot_path.name}")

    mitigation_results = {
        "benchmark_matrix": benchmark_df.to_dict(orient="records"),
        "audits": all_audits
    }

    return mitigation_results, benchmark_df


# ------------------------------------------------------------------------------
# 7. MASTER EXPERIMENT 5 PIPELINE ORCHESTRATOR
# ------------------------------------------------------------------------------

def run_experiment_5_pipeline() -> Dict[str, Any]:
    """
    Master pipeline orchestrator for Experiment 5:
    Loads dataset, trains models, runs SHAP, LIME, Fairlearn audit & mitigation,
    and serializes comprehensive metadata summary.
    """
    start_time = time.time()
    print("=" * 80)
    print(" EXPERIMENT 5: EXPLAINABLE AI (XAI) & FAIRNESS AUDITING PIPELINE")
    print("=" * 80)

    # Stage 1: Ingest Data
    X, y, X_disp, data_meta = load_and_prepare_census_data()

    # Stratified 80/20 Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    sex_train = X_train["Sex"]
    sex_test = X_test["Sex"]
    race_train = X_train["Race"]
    race_test = X_test["Race"]

    # Stage 2: Train Unmitigated Benchmark Models
    lgb_champion, rf_baseline, model_summary = train_unmitigated_models(
        X_train, y_train, X_test, y_test
    )

    # Stage 3: Explainability with SHAP
    shap_summary = run_shap_explainability(
        lgb_champion, X_train, X_test, y_test, list(X.columns)
    )

    # Stage 4: Explainability with LIME
    lime_summary = run_lime_explainability(
        lgb_champion, X_train, X_test, y_test, list(X.columns)
    )

    # Stage 5 & 6: Fairness Audit & Bias Mitigation
    mitigation_summary, benchmark_df = run_bias_mitigation(
        lgb_champion, X_train, y_train, X_test, y_test,
        sex_train, sex_test, race_test
    )

    # Save detailed fairness audit CSV
    audit_rows = []
    for audit_item in mitigation_summary["audits"]:
        m_name = audit_item["model_name"]
        for grp, vals in audit_item["sex_audit"]["by_group"].items():
            row = {"Model": m_name, "Attribute": "Sex", "Group": grp}
            row.update(vals)
            audit_rows.append(row)
        for grp, vals in audit_item["race_audit"]["by_group"].items():
            row = {"Model": m_name, "Attribute": "Race", "Group": grp}
            row.update(vals)
            audit_rows.append(row)
    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(REPORTS_DIR / "fairness_audit_metrics.csv", index=False)
    print(f"[+] Saved granular fairness audit metrics to: {REPORTS_DIR / 'fairness_audit_metrics.csv'}")

    elapsed_time = time.time() - start_time

    # Master Experiment Summary Metadata
    experiment_summary = {
        "experiment_title": "Experiment 5: Explainable AI (SHAP & LIME) & Algorithmic Fairness (Fairlearn)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_seconds": round(elapsed_time, 2),
        "dataset_metadata": data_meta,
        "modeling_benchmarks": model_summary,
        "shap_explainability": shap_summary,
        "lime_explainability": lime_summary,
        "fairness_and_mitigation": mitigation_summary["benchmark_matrix"],
        "generated_artifacts": {
            "plots": [f.name for f in PLOTS_DIR.glob("*.png")],
            "models": [f.name for f in MODELS_DIR.glob("*.joblib")],
            "reports": [f.name for f in REPORTS_DIR.glob("*.csv")]
        }
    }

    summary_path = MODELS_DIR / "exp5_xai_fairness_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(experiment_summary, f, indent=2)
    print(f"[+] Master experiment summary serialized to: {summary_path}")

    print("\n" + "=" * 80)
    print(f" [SUCCESS] EXPERIMENT 5 PIPELINE COMPLETED IN {elapsed_time:.2f} SECONDS!")
    print("=" * 80)

    return experiment_summary


def load_experiment_5_summary() -> Dict[str, Any]:
    """Loads previously saved experiment 5 summary metadata."""
    summary_path = MODELS_DIR / "exp5_xai_fairness_summary.json"
    if not summary_path.exists():
        return run_experiment_5_pipeline()
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    run_experiment_5_pipeline()
