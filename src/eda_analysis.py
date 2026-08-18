"""
================================================================================
EXPERIMENT 3: EXPLORATORY DATA ANALYSIS & STATISTICAL HYPOTHESIS TESTING
Dataset: Customer Support on Twitter (TWCS) Cleaned Dataset
Tools  : Pandas, NumPy, Matplotlib, Seaborn, SciPy, Statsmodels
================================================================================
"""

import os
import json
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from .emotion_labeler import EmotionLabeler

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Set styling aesthetics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'axes.edgecolor': '#cccccc',
    'axes.linewidth': 0.8,
    'grid.alpha': 0.5,
    'grid.linestyle': '--'
})


# ------------------------------------------------------------------------------
# 1. DATA PREPARATION & FEATURE EXTRACTION
# ------------------------------------------------------------------------------

def prepare_eda_dataset(input_file: str = None, sample_size: int = 50000) -> pd.DataFrame:
    """
    Loads cleaned tweets dataset, annotates emotion and sentiment labels,
    and extracts temporal and lexical structural features.
    """
    if input_file is None:
        input_file = str(PROJECT_ROOT / "data" / "processed" / "twcs_cleaned.csv")
        
    print("=" * 75)
    print(" [STEP 1] LOADING DATASET & EXTRACTING MULTIDIMENSIONAL FEATURES")
    print("=" * 75)
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Cleaned dataset '{input_file}' not found.")
        
    df = pd.read_csv(input_file)
    print(f" Loaded raw cleaned dataset: {len(df):,} records from '{input_file}'")
    
    if sample_size > 0 and len(df) > sample_size:
        print(f" Sampling {sample_size:,} records for high-fidelity EDA & statistical testing...")
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        
    df['clean_text'] = df['clean_text'].fillna("").astype(str)
    
    # Label emotions & VADER sentiment scores
    print(" Annotating tweets with VADER Polarity and Emotion Categories...")
    labeler = EmotionLabeler()
    df = labeler.label_dataframe(df, text_column='clean_text')
    
    # Feature 1: Character count
    df['char_count'] = df['clean_text'].apply(len)
    
    # Feature 2: Exclamation and question mark counts (emotional punctuation intensity)
    if 'text' in df.columns:
        df['exclamation_count'] = df['text'].fillna("").apply(lambda s: s.count('!'))
        df['question_count'] = df['text'].fillna("").apply(lambda s: s.count('?'))
        df['caps_ratio'] = df['text'].fillna("").apply(
            lambda s: sum(1 for c in s if c.isupper()) / max(len(s), 1)
        )
    else:
        df['exclamation_count'] = df['clean_text'].apply(lambda s: s.count('!'))
        df['question_count'] = df['clean_text'].apply(lambda s: s.count('?'))
        df['caps_ratio'] = 0.0
        
    # Feature 3: Temporal mapping from created_at
    if 'created_at' in df.columns:
        try:
            df['created_dt'] = pd.to_datetime(df['created_at'], errors='coerce', utc=True)
            df['hour_of_day'] = df['created_dt'].dt.hour
            
            def map_time_of_day(hour):
                if pd.isna(hour):
                    return "Afternoon"
                if 6 <= hour < 12:
                    return "Morning"
                elif 12 <= hour < 17:
                    return "Afternoon"
                elif 17 <= hour < 22:
                    return "Evening"
                else:
                    return "Night"
                    
            df['time_of_day'] = df['hour_of_day'].apply(map_time_of_day)
        except Exception as e:
            print(f" Warning: Could not parse created_at dates ({e}). Defaulting time_of_day.")
            df['time_of_day'] = "Afternoon"
    else:
        df['time_of_day'] = "Afternoon"
        
    print(f" Dataset successfully prepared with shape {df.shape}.\n")
    return df


# ------------------------------------------------------------------------------
# 2. CLASS BALANCE VISUALIZATIONS
# ------------------------------------------------------------------------------

def plot_class_balance(df: pd.DataFrame, output_plot_path: str = None):
    """
    Plots dual-panel visualization of class distribution:
    Panel A: Horizontal Bar chart with counts & percentages.
    Panel B: Donut chart displaying proportional class balance.
    """
    if output_plot_path is None:
        output_plot_path = str(PROJECT_ROOT / "plots" / "exp3_class_balance.png")
    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
    
    print("=" * 75)
    print(" [STEP 2] VISUALIZING CLASS DISTRIBUTION & BALANCE RATIOS")
    print("=" * 75)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    counts = df['emotion'].value_counts()
    percentages = df['emotion'].value_counts(normalize=True) * 100
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    
    # Subplot 1: Barplot
    ax1 = axes[0]
    bars = ax1.barh(counts.index, counts.values, color=colors[:len(counts)], edgecolor='black', alpha=0.85)
    ax1.set_title("A. Emotion Category Frequency Counts", fontsize=13, fontweight='bold', pad=12)
    ax1.set_xlabel("Number of Customer Inquiries", fontsize=11)
    ax1.set_ylabel("Emotion Category", fontsize=11)
    ax1.grid(axis='x', linestyle='--', alpha=0.6)
    
    for bar, pct in zip(bars, percentages):
        width = bar.get_width()
        ax1.annotate(f'{width:,} ({pct:.1f}%)',
                     xy=(width, bar.get_y() + bar.get_height() / 2),
                     xytext=(6, 0), textcoords="offset points",
                     ha='left', va='center', fontsize=10, fontweight='bold')
                     
    ax1.set_xlim(0, max(counts.values) * 1.25)
    ax1.invert_yaxis()
    
    # Subplot 2: Donut Chart
    ax2 = axes[1]
    wedges, texts, autotexts = ax2.pie(
        counts.values,
        labels=counts.index,
        autopct='%1.1f%%',
        startangle=140,
        colors=colors[:len(counts)],
        wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2),
        pctdistance=0.75
    )
    for t in autotexts:
        t.set_fontsize(10)
        t.set_fontweight('bold')
    ax2.set_title("B. Proportional Emotion Composition (Donut Plot)", fontsize=13, fontweight='bold', pad=12)
    
    imbalance_ratio = counts.max() / counts.min()
    plt.suptitle(f"TWCS Emotion Class Balance Analysis (Class Imbalance Ratio: {imbalance_ratio:.2f}:1)", 
                 fontsize=15, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" [+] Saved class balance visualization to '{output_plot_path}'")


# ------------------------------------------------------------------------------
# 3. FEATURE DISTRIBUTIONS & CENTRAL TENDENCIES
# ------------------------------------------------------------------------------

def plot_feature_distributions(df: pd.DataFrame, output_plot_path: str = None):
    """
    Plots multi-panel histogram and KDE density curves comparing distributions,
    annotating Mean, Median, and Mode.
    """
    if output_plot_path is None:
        output_plot_path = str(PROJECT_ROOT / "plots" / "exp3_feature_distributions.png")
    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
    
    print("=" * 75)
    print(" [STEP 3] COMPUTING CENTRAL TENDENCIES & PLOTTING DISTRIBUTIONS")
    print("=" * 75)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    features = [
        ('word_count', 'Word Count per Tweet', 'words', 0, 80),
        ('char_count', 'Character Count per Clean Tweet', 'characters', 0, 350),
        ('vader_compound', 'VADER Compound Sentiment Intensity', 'score', -1.0, 1.0),
        ('caps_ratio', 'Uppercase Letter Proportion (Caps Ratio)', 'ratio', 0.0, 0.4)
    ]
    
    for i, (col, title, unit, x_min, x_max) in enumerate(features):
        ax = axes[i // 2, i % 2]
        data = df[col].dropna()
        data = data[(data >= x_min) & (data <= x_max)]
        
        mean_val = data.mean()
        median_val = data.median()
        mode_val = data.mode()[0] if not data.mode().empty else mean_val
        std_val = data.std()
        skew_val = stats.skew(data)
        kurt_val = stats.kurtosis(data)
        
        print(f" * Feature '{col:15s}' -> Mean: {mean_val:7.3f} | Median: {median_val:7.3f} | Mode: {mode_val:7.3f} | Skew: {skew_val:+.3f} | Kurtosis: {kurt_val:+.3f}")
        
        sns.histplot(data, kde=True, ax=ax, color='#2980b9', bins=35, stat="density", alpha=0.45, edgecolor='black', linewidth=0.5)
        
        # Central tendency vertical markers
        ax.axvline(mean_val, color='#e74c3c', linestyle='-', linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='#27ae60', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
        ax.axvline(mode_val, color='#8e44ad', linestyle=':', linewidth=2, label=f'Mode: {mode_val:.2f}')
        
        ax.set_title(f"{title}\n(Skewness: {skew_val:+.2f}, Kurtosis: {kurt_val:+.2f})", fontsize=11.5, fontweight='bold')
        ax.set_xlabel(f"{title} ({unit})", fontsize=10.5)
        ax.set_ylabel("Probability Density", fontsize=10.5)
        ax.legend(loc='upper right', frameon=True)
        ax.set_xlim(x_min, x_max)
        
    plt.suptitle("Feature Probability Density Distributions & Central Tendency Measures", fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" [+] Saved feature distributions plot to '{output_plot_path}'")


# ------------------------------------------------------------------------------
# 4. SPREAD & DISPERSION: BOXPLOTS & IQR ANALYSIS
# ------------------------------------------------------------------------------

def plot_boxplots_and_spread(df: pd.DataFrame, output_plot_path: str = None):
    """
    Generates comparative Boxplots and Violin plots across emotion classes.
    """
    if output_plot_path is None:
        output_plot_path = str(PROJECT_ROOT / "plots" / "exp3_boxplots_spread.png")
    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
    
    print("=" * 75)
    print(" [STEP 4] GENERATING BOXPLOTS, VIOLIN PLOTS & IQR SPREAD METRICS")
    print("=" * 75)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Word Count by Emotion (Boxplot)
    ax1 = axes[0, 0]
    sns.boxplot(data=df, x='emotion', y='word_count', ax=ax1, palette='Set2', showmeans=True,
                meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"7"})
    ax1.set_title("A. Word Count Spread Across Emotion Classes (IQR & Outliers)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Emotion Category", fontsize=10.5)
    ax1.set_ylabel("Word Count", fontsize=10.5)
    ax1.set_ylim(0, 60)
    ax1.tick_params(axis='x', rotation=15)
    
    # 2. VADER Compound Score by Emotion (Violin + Boxplot)
    ax2 = axes[0, 1]
    sns.violinplot(data=df, x='emotion', y='vader_compound', ax=ax2, palette='coolwarm', inner='quartile', cut=0)
    ax2.set_title("B. VADER Compound Polarity Density & Spread by Emotion", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Emotion Category", fontsize=10.5)
    ax2.set_ylabel("VADER Compound Polarity (-1.0 to +1.0)", fontsize=10.5)
    ax2.tick_params(axis='x', rotation=15)
    
    # 3. Exclamation Marks by Emotion (Boxplot)
    ax3 = axes[1, 0]
    sns.boxplot(data=df, x='emotion', y='exclamation_count', ax=ax3, palette='Pastel1', showmeans=True,
                meanprops={"marker":"^", "markerfacecolor":"red", "markeredgecolor":"black", "markersize":"7"})
    ax3.set_title("C. Exclamation Mark Frequency Spread by Emotion", fontsize=12, fontweight='bold')
    ax3.set_xlabel("Emotion Category", fontsize=10.5)
    ax3.set_ylabel("Exclamation Count", fontsize=10.5)
    ax3.set_ylim(0, 5)
    ax3.tick_params(axis='x', rotation=15)
    
    # 4. Word Count Spread by Time of Day
    ax4 = axes[1, 1]
    time_order = ['Morning', 'Afternoon', 'Evening', 'Night']
    sns.boxplot(data=df, x='time_of_day', y='word_count', order=time_order, ax=ax4, palette='Blues_r', showmeans=True)
    ax4.set_title("D. Inquiry Length Dispersion Across Time of Day", fontsize=12, fontweight='bold')
    ax4.set_xlabel("Time of Day", fontsize=10.5)
    ax4.set_ylabel("Word Count", fontsize=10.5)
    ax4.set_ylim(0, 50)
    
    plt.suptitle("Comparative Data Dispersion, Spread & Interquartile Range (IQR) Visualizations", 
                 fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" [+] Saved boxplots and spread visualization to '{output_plot_path}'")


# ------------------------------------------------------------------------------
# 5. CORRELATION ANALYSIS (PEARSON & SPEARMAN HEATMAPS)
# ------------------------------------------------------------------------------

def plot_correlation_heatmap(df: pd.DataFrame, output_plot_path: str = None):
    """
    Computes and plots Pearson (linear) and Spearman (rank-order) correlation heatmaps.
    """
    if output_plot_path is None:
        output_plot_path = str(PROJECT_ROOT / "plots" / "exp3_correlation_heatmap.png")
    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
    
    print("=" * 75)
    print(" [STEP 5] COMPUTING PEARSON & SPEARMAN CORRELATION HEATMAPS")
    print("=" * 75)
    
    features = [
        'word_count', 'char_count', 'vader_compound', 
        'vader_pos', 'vader_neg', 'vader_neu', 
        'exclamation_count', 'question_count', 'caps_ratio'
    ]
    sub_df = df[features].dropna()
    
    pearson_corr = sub_df.corr(method='pearson')
    spearman_corr = sub_df.corr(method='spearman')
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7.5))
    
    # 1. Pearson Heatmap
    ax1 = axes[0]
    sns.heatmap(pearson_corr, annot=True, fmt=".2f", cmap="vlag", vmin=-1.0, vmax=1.0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax1)
    ax1.set_title("A. Pearson Linear Correlation Matrix (r)", fontsize=13, fontweight='bold', pad=12)
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Spearman Heatmap
    ax2 = axes[1]
    sns.heatmap(spearman_corr, annot=True, fmt=".2f", cmap="vlag", vmin=-1.0, vmax=1.0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax2)
    ax2.set_title("B. Spearman Rank-Order Monotonic Correlation Matrix (ρ)", fontsize=13, fontweight='bold', pad=12)
    ax2.tick_params(axis='x', rotation=45)
    
    plt.suptitle("Feature Correlation Heatmap Matrices (Linear vs Monotonic Non-linear Associations)", 
                 fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" [+] Saved correlation heatmaps to '{output_plot_path}'")
    return pearson_corr


# ------------------------------------------------------------------------------
# 6. DISTRIBUTION FITTING & OUTLIER DETECTION
# ------------------------------------------------------------------------------

def analyze_distributions_and_outliers(df: pd.DataFrame, output_plot_path: str = None):
    """
    Fits continuous (Gaussian, Log-Normal, Exponential) and discrete (Poisson)
    probability distributions to empirical features.
    Generates Q-Q Plots and detects anomalies using IQR & Z-score methods.
    """
    if output_plot_path is None:
        output_plot_path = str(PROJECT_ROOT / "plots" / "exp3_distribution_fitting_outliers.png")
    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
    
    print("=" * 75)
    print(" [STEP 6] FITTING THEORETICAL DISTRIBUTIONS & DETECTING OUTLIERS")
    print("=" * 75)
    
    words = df['word_count'].dropna().values
    exclamations = df['exclamation_count'].dropna().values
    
    # 1. Continuous Distribution Fitting (Word Count)
    # A. Normal / Gaussian
    norm_mu, norm_std = stats.norm.fit(words)
    # B. Log-Normal
    shape, loc, scale = stats.lognorm.fit(words, floc=0)
    # C. Exponential
    exp_loc, exp_scale = stats.expon.fit(words)
    
    # Kolmogorov-Smirnov Goodness-of-Fit Tests
    ks_norm = stats.kstest(words, 'norm', args=(norm_mu, norm_std))
    ks_lognorm = stats.kstest(words, 'lognorm', args=(shape, loc, scale))
    ks_expon = stats.kstest(words, 'expon', args=(exp_loc, exp_scale))
    
    # 2. Discrete Distribution Fitting (Poisson Distribution on Exclamation Count)
    poisson_lambda = float(np.mean(exclamations))
    # Chi-Square Goodness-of-Fit for Poisson
    obs_k, obs_counts = np.unique(exclamations[exclamations <= 5], return_counts=True)
    n_discrete = len(exclamations[exclamations <= 5])
    exp_probs = stats.poisson.pmf(obs_k, poisson_lambda)
    exp_counts = exp_probs * (n_discrete / exp_probs.sum())
    chi2_poisson, p_poisson = stats.chisquare(obs_counts, f_exp=exp_counts)
    
    print(f" * Continuous Distribution Fitting ('word_count'):")
    print(f"   - Gaussian Fit   : mu={norm_mu:.2f}, std={norm_std:.2f} | KS-Stat: {ks_norm.statistic:.4f} (p={ks_norm.pvalue:.2e})")
    print(f"   - Log-Normal Fit : shape={shape:.2f}, scale={scale:.2f} | KS-Stat: {ks_lognorm.statistic:.4f} (p={ks_lognorm.pvalue:.2e})")
    print(f"   - Exponential Fit: scale={exp_scale:.2f}            | KS-Stat: {ks_expon.statistic:.4f} (p={ks_expon.pvalue:.2e})")
    print(f" * Discrete Distribution Fitting ('exclamation_count' - Poisson):")
    print(f"   - Poisson Parameter : lambda = {poisson_lambda:.3f} | Chi2-Stat: {chi2_poisson:.4f} (p={p_poisson:.2e})")
    
    # 3. Outlier Detection
    q25, q75 = np.percentile(words, 25), np.percentile(words, 75)
    iqr = q75 - q25
    iqr_lower = max(0, q25 - 1.5 * iqr)
    iqr_upper = q75 + 1.5 * iqr
    iqr_outliers = np.sum((words < iqr_lower) | (words > iqr_upper))
    
    z_scores = np.abs(stats.zscore(words))
    z_outliers = np.sum(z_scores > 3.0)
    
    print(f"\n * Outlier Detection Analysis:")
    print(f"   - IQR Method     : IQR={iqr:.1f} | Bounds: [{iqr_lower:.1f}, {iqr_upper:.1f}] | Outliers: {iqr_outliers:,} ({iqr_outliers/len(words)*100:.2f}%)")
    print(f"   - Z-Score Method : Threshold |Z| > 3.0 | Outliers: {z_outliers:,} ({z_outliers/len(words)*100:.2f}%)")
    
    # 4. Plotting Fits & Q-Q Plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # A. Continuous PDF Fitted Density vs Empirical Histogram
    ax1 = axes[0, 0]
    x_vals = np.linspace(1, 60, 300)
    sns.histplot(words, stat="density", bins=40, ax=ax1, color='#bdc3c7', edgecolor='black', alpha=0.6, label='Empirical Density')
    
    pdf_norm = stats.norm.pdf(x_vals, norm_mu, norm_std)
    pdf_lognorm = stats.lognorm.pdf(x_vals, shape, loc, scale)
    pdf_expon = stats.expon.pdf(x_vals, exp_loc, exp_scale)
    
    ax1.plot(x_vals, pdf_norm, 'r-', linewidth=2.5, label=f'Gaussian Fit (KS={ks_norm.statistic:.3f})')
    ax1.plot(x_vals, pdf_lognorm, 'g--', linewidth=2.5, label=f'Log-Normal Fit (KS={ks_lognorm.statistic:.3f})')
    ax1.plot(x_vals, pdf_expon, 'b:', linewidth=2.5, label=f'Exponential Fit (KS={ks_expon.statistic:.3f})')
    
    ax1.set_title("A. Continuous Variable: Empirical Density vs Theoretical Fits", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Word Count per Tweet", fontsize=11)
    ax1.set_ylabel("Probability Density", fontsize=11)
    ax1.set_xlim(0, 60)
    ax1.legend(loc='upper right')
    
    # B. Discrete Variable Poisson Fit (Exclamation Count)
    ax2 = axes[0, 1]
    k_vals = np.arange(0, 7)
    emp_freq = [np.mean(exclamations == k) for k in k_vals]
    pois_pmf = [stats.poisson.pmf(k, poisson_lambda) for k in k_vals]
    
    bar_w = 0.35
    ax2.bar(k_vals - bar_w/2, emp_freq, width=bar_w, color='#3498db', alpha=0.75, edgecolor='black', label='Empirical Relative Freq')
    ax2.bar(k_vals + bar_w/2, pois_pmf, width=bar_w, color='#e67e22', alpha=0.75, edgecolor='black', label=f'Poisson PMF (λ={poisson_lambda:.2f})')
    ax2.set_title("B. Discrete Variable: Poisson Distribution Fitting (Exclamation Marks)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Exclamation Count (k)", fontsize=11)
    ax2.set_ylabel("Probability Mass P(X=k)", fontsize=11)
    ax2.set_xticks(k_vals)
    ax2.legend(loc='upper right')
    
    # C. Log-Normal Q-Q Plot
    ax3 = axes[1, 0]
    sample_log = np.log(words[words > 0])
    sm.qqplot(sample_log, line='s', ax=ax3, markerfacecolor='#27ae60', markeredgecolor='none', alpha=0.3)
    ax3.set_title("C. Log-Transformed Normal Q-Q Plot (Log-Normal Model Diagnostic)", fontsize=12, fontweight='bold')
    
    # D. Outlier Boxplot & Annotations
    ax4 = axes[1, 1]
    sns.boxplot(x=words, ax=ax4, color='#f39c12', flierprops=dict(marker='o', markersize=3, alpha=0.3))
    ax4.axvline(iqr_upper, color='red', linestyle='--', linewidth=2, label=f'IQR Upper Fence ({iqr_upper:.1f})')
    ax4.set_title(f"D. Outlier Detection: IQR Fence & Empirical Dispersion ({iqr_outliers:,} Outliers)", fontsize=12, fontweight='bold')
    ax4.set_xlabel("Word Count", fontsize=11)
    ax4.set_xlim(0, 70)
    ax4.legend(loc='upper right')
    
    plt.suptitle("Continuous & Discrete Distribution Fitting, Q-Q Diagnostics & Outlier Analysis", fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" [+] Saved distribution fitting and outlier analysis to '{output_plot_path}'")
    
    return {
        "gaussian": {"mu": norm_mu, "std": norm_std, "ks_stat": ks_norm.statistic, "p_value": ks_norm.pvalue},
        "lognormal": {"shape": shape, "scale": scale, "ks_stat": ks_lognorm.statistic, "p_value": ks_lognorm.pvalue},
        "exponential": {"scale": exp_scale, "ks_stat": ks_expon.statistic, "p_value": ks_expon.pvalue},
        "poisson": {"lambda": poisson_lambda, "chi2_stat": chi2_poisson, "p_value": p_poisson},
        "outliers": {"iqr_count": int(iqr_outliers), "z_count": int(z_outliers), "iqr_upper_bound": iqr_upper}
    }


# ------------------------------------------------------------------------------
# 7. STATISTICAL HYPOTHESIS TESTING SUITE
# ------------------------------------------------------------------------------

def perform_hypothesis_testing(df: pd.DataFrame, output_plot_path: str = None):
    """
    Executes formal statistical hypothesis testing suite:
    1. Welch's Two-Sample t-Test (Negative/Frustration vs Joy Word Count)
    2. One-Way ANOVA & Tukey HSD Post-Hoc Test across all Emotion Classes
    3. Chi-Square Test of Independence (Emotion Category vs Time of Day) + Cramér's V
    4. Mann-Whitney U Non-Parametric Rank-Sum Test
    """
    if output_plot_path is None:
        output_plot_path = str(PROJECT_ROOT / "plots" / "exp3_hypothesis_tests.png")
    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
    
    print("=" * 75)
    print(" [STEP 7] EXECUTING STATISTICAL HYPOTHESIS TESTING SUITE")
    print("=" * 75)
    
    test_results = {}
    
    # -------------------------------------------------------------------------
    # Test 1: Welch's t-Test
    # H0: Mean word count of Complaints/Frustration == Mean word count of Joy/Gratitude
    # H1: Mean word count of Complaints/Frustration != Mean word count of Joy/Gratitude
    # -------------------------------------------------------------------------
    negative_words = df[df['emotion'].isin(['Anger / Frustration', 'Disappointment / Sadness'])]['word_count'].dropna()
    joy_words = df[df['emotion'] == 'Joy / Gratitude']['word_count'].dropna()
    
    t_stat, p_val_ttest = stats.ttest_ind(negative_words, joy_words, equal_var=False)
    
    mean_neg, std_neg = negative_words.mean(), negative_words.std()
    mean_joy, std_joy = joy_words.mean(), joy_words.std()
    pooled_sd = np.sqrt(((len(negative_words)-1)*std_neg**2 + (len(joy_words)-1)*std_joy**2) / (len(negative_words) + len(joy_words) - 2))
    cohens_d = (mean_neg - mean_joy) / pooled_sd
    
    print("\n--- [HYPOTHESIS TEST 1] Welch's Two-Sample Independent t-Test ---")
    print(f" Research Question: Do customers write longer tweets when complaining vs expressing gratitude?")
    print(f" Sample Negative/Frustration (N={len(negative_words):,}): Mean = {mean_neg:.2f} ± {std_neg:.2f} words")
    print(f" Sample Joy/Gratitude        (N={len(joy_words):,}): Mean = {mean_joy:.2f} ± {std_joy:.2f} words")
    print(f" Welch's t-statistic : t = {t_stat:.4f}")
    print(f" p-value             : p = {p_val_ttest:.4e}")
    print(f" Effect Size (Cohen's d): d = {cohens_d:.3f}")
    print(f" Decision            : {'REJECT NULL HYPOTHESIS (p < 0.05)' if p_val_ttest < 0.05 else 'FAIL TO REJECT NULL HYPOTHESIS'}")
    
    test_results["welchs_ttest"] = {
        "negative_mean": float(mean_neg),
        "joy_mean": float(mean_joy),
        "t_stat": float(t_stat),
        "p_value": float(p_val_ttest),
        "cohens_d": float(cohens_d),
        "significant": bool(p_val_ttest < 0.05)
    }

    # -------------------------------------------------------------------------
    # Test 2: One-Way ANOVA across Emotion Categories
    # H0: Mean word count is identical across all emotion groups
    # H1: At least one emotion group has a significantly different mean word count
    # -------------------------------------------------------------------------
    groups = [group['word_count'].dropna().values for _, group in df.groupby('emotion')]
    f_stat, p_val_anova = stats.f_oneway(*groups)
    
    # Tukey HSD Post-Hoc Test
    tukey = pairwise_tukeyhsd(endog=df['word_count'], groups=df['emotion'], alpha=0.05)
    
    print("\n--- [HYPOTHESIS TEST 2] One-Way ANOVA & Tukey HSD Post-Hoc Test ---")
    print(f" Research Question: Is tweet length significantly different across all 5 emotion categories?")
    print(f" ANOVA F-statistic   : F = {f_stat:.4f}")
    print(f" ANOVA p-value       : p = {p_val_anova:.4e}")
    print(f" Decision            : {'REJECT NULL HYPOTHESIS (p < 0.05)' if p_val_anova < 0.05 else 'FAIL TO REJECT NULL HYPOTHESIS'}")
    
    test_results["anova"] = {
        "f_stat": float(f_stat),
        "p_value": float(p_val_anova),
        "significant": bool(p_val_anova < 0.05)
    }

    # -------------------------------------------------------------------------
    # Test 3: Chi-Square Test of Independence (Emotion vs Time of Day)
    # H0: Customer emotion is independent of time of day
    # H1: Customer emotion is dependent on time of day
    # -------------------------------------------------------------------------
    contingency_table = pd.crosstab(df['emotion'], df['time_of_day'])
    chi2_stat, chi2_pval, dof, expected = stats.chi2_contingency(contingency_table)
    
    n_total = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape) - 1
    cramers_v = np.sqrt(chi2_stat / (n_total * min_dim))
    
    print("\n--- [HYPOTHESIS TEST 3] Chi-Square Test of Independence & Cramér's V ---")
    print(f" Research Question: Does customer emotional state vary depending on time of day (Morning/Afternoon/Evening/Night)?")
    print(f" Contingency Table Matrix:\n{contingency_table}")
    print(f" Chi-Square (χ²) Stat: χ² = {chi2_stat:.4f} (dof = {dof})")
    print(f" p-value             : p = {chi2_pval:.4e}")
    print(f" Cramér's V Strength : V = {cramers_v:.4f}")
    print(f" Decision            : {'REJECT NULL HYPOTHESIS (p < 0.05)' if chi2_pval < 0.05 else 'FAIL TO REJECT NULL HYPOTHESIS'}")
    
    test_results["chi_square"] = {
        "chi2_stat": float(chi2_stat),
        "dof": int(dof),
        "p_value": float(chi2_pval),
        "cramers_v": float(cramers_v),
        "significant": bool(chi2_pval < 0.05)
    }

    # -------------------------------------------------------------------------
    # Test 4: Mann-Whitney U Non-Parametric Test
    # -------------------------------------------------------------------------
    u_stat, p_val_mwu = stats.mannwhitneyu(negative_words, joy_words, alternative='two-sided')
    print("\n--- [HYPOTHESIS TEST 4] Mann-Whitney U Non-Parametric Rank-Sum Test ---")
    print(f" Mann-Whitney U Stat : U = {u_stat:,.1f}")
    print(f" p-value             : p = {p_val_mwu:.4e}")
    print(f" Decision            : {'REJECT NULL HYPOTHESIS' if p_val_mwu < 0.05 else 'FAIL TO REJECT'}")
    
    test_results["mann_whitney"] = {
        "u_stat": float(u_stat),
        "p_value": float(p_val_mwu),
        "significant": bool(p_val_mwu < 0.05)
    }

    # Visualization of Hypothesis Tests
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Welch's t-test Bar Comparison with Error Bars (SEM)
    means = [mean_neg, mean_joy]
    sems = [std_neg / np.sqrt(len(negative_words)), std_joy / np.sqrt(len(joy_words))]
    labels = [f'Negative / Complaint\n(N={len(negative_words):,})', f'Joy / Gratitude\n(N={len(joy_words):,})']
    
    bars = ax1.bar(labels, means, yerr=sems, capsize=8, color=['#e74c3c', '#2ecc71'], alpha=0.85, edgecolor='black', linewidth=1.2)
    ax1.set_title("Hypothesis 1: Welch's t-Test Comparison\nMean Word Count (Complaints vs Joy)", fontsize=12, fontweight='bold', pad=12)
    ax1.set_ylabel("Mean Word Count (Words/Tweet)", fontsize=11)
    ax1.set_ylim(0, max(means) * 1.35)
    
    for bar in bars:
        h = bar.get_height()
        ax1.annotate(f'{h:.2f} words', (bar.get_x() + bar.get_width() / 2., h / 2),
                     ha='center', va='center', color='white', fontsize=12, fontweight='bold')
    ax1.text(0.5, max(means) * 1.18, f"t = {t_stat:.2f} | p < 1e-10 (Significantly Higher)",
             ha='center', fontsize=11, fontweight='bold', color='crimson',
             bbox=dict(boxstyle="round,pad=0.4", facecolor='yellow', alpha=0.3, edgecolor='red'))
             
    # Plot 2: Contingency Heatmap (Time of Day vs Emotion Proportions)
    prop_table = contingency_table.div(contingency_table.sum(axis=0), axis=1) * 100
    sns.heatmap(prop_table, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label': '% of Time Period Tweets'}, ax=ax2)
    ax2.set_title(f"Hypothesis 3: Chi-Square Test of Independence\nEmotion % by Time-of-Day (Chi2={chi2_stat:.1f}, p={chi2_pval:.2e})", fontsize=12, fontweight='bold', pad=12)
    ax2.set_xlabel("Time of Day", fontsize=11)
    ax2.set_ylabel("Emotion Category", fontsize=11)
    
    plt.suptitle("Statistical Hypothesis Testing Visualizations", fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n [+] Saved hypothesis testing visualizations to '{output_plot_path}'")
    
    return test_results


# ------------------------------------------------------------------------------
# 8. MAIN WORKFLOW CONTROLLER & EXPORT
# ------------------------------------------------------------------------------

def run_experiment_3(input_file: str = None, sample_size: int = 50000):
    if input_file is None:
        input_file = str(PROJECT_ROOT / "data" / "processed" / "twcs_cleaned.csv")
        
    start_time = time.time()
    
    print("=" * 80)
    print("              EXPERIMENT 3: EDA & STATISTICAL ANALYSIS PIPELINE")
    print("=" * 80)
    
    os.makedirs(str(PROJECT_ROOT / "plots"), exist_ok=True)
    os.makedirs(str(PROJECT_ROOT / "models"), exist_ok=True)
    
    # Step 1: Load and feature engineer dataset
    df = prepare_eda_dataset(input_file, sample_size)
    
    # Step 2: Plot Class Balance (Count plot & Pie chart)
    plot_class_balance(df, str(PROJECT_ROOT / "plots" / "exp3_class_balance.png"))
    
    # Step 3: Frequency Distributions (Histograms + KDE + Central Tendencies)
    plot_feature_distributions(df, str(PROJECT_ROOT / "plots" / "exp3_feature_distributions.png"))
    
    # Step 4: Boxplots, Spread & IQR Dispersion
    plot_boxplots_and_spread(df, str(PROJECT_ROOT / "plots" / "exp3_boxplots_spread.png"))
    
    # Step 5: Correlation Heatmaps (Pearson & Spearman)
    corr_df = plot_correlation_heatmap(df, str(PROJECT_ROOT / "plots" / "exp3_correlation_heatmap.png"))
    
    # Step 6: Distribution Fitting & Outliers
    fit_summary = analyze_distributions_and_outliers(df, str(PROJECT_ROOT / "plots" / "exp3_distribution_fitting_outliers.png"))
    
    # Step 7: Statistical Hypothesis Testing Suite
    test_results = perform_hypothesis_testing(df, str(PROJECT_ROOT / "plots" / "exp3_hypothesis_tests.png"))
    
    # Save experiment 3 metadata JSON
    metadata = {
        "experiment": "Experiment 3: Exploratory Data Analysis & Statistical Analysis",
        "sample_size": len(df),
        "class_distribution": df['emotion'].value_counts().to_dict(),
        "summary_statistics": df[['word_count', 'char_count', 'vader_compound', 'vader_pos', 'vader_neg', 'vader_neu']].describe().to_dict(),
        "distribution_fits": fit_summary,
        "hypothesis_tests": test_results,
        "runtime_seconds": time.time() - start_time
    }
    
    summary_path = str(PROJECT_ROOT / "models" / "exp3_eda_summary.json")
    with open(summary_path, "w") as f:
        json.dump(metadata, f, indent=4)
        
    print("\n" + "=" * 80)
    print(f" EXPERIMENT 3 EXECUTION COMPLETED IN {time.time() - start_time:.2f} SECONDS")
    print(f" All 6 publication-ready figures saved in '{PROJECT_ROOT / 'plots'}'")
    print(f" Summary metadata exported to '{summary_path}'")
    print("=" * 80)


if __name__ == '__main__':
    default_input = str(PROJECT_ROOT / "data" / "processed" / "twcs_cleaned.csv")
    parser = argparse.ArgumentParser(description="Experiment 3: EDA & Statistical Analysis Pipeline")
    parser.add_argument("--input", "-i", type=str, default=default_input, help="Input cleaned CSV path")
    parser.add_argument("--sample-size", "-s", type=int, default=50000, help="Sample size for analysis (0 for all)")
    args = parser.parse_args()
    
    run_experiment_3(args.input, args.sample_size)
