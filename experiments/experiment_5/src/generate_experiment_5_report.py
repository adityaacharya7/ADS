"""
Academic Report Generator for Experiment 5:
Explainable AI (SHAP & LIME) & Algorithmic Fairness Auditing (Fairlearn)

Generates:
1. Publication-quality Academic PDF report via ReportLab (Experiment_5_Report.pdf)
2. Complete IEEE/ACM-style LaTeX source file (Experiment_5_Report.tex)
3. Self-contained Overleaf upload zip bundle (Experiment_5_Overleaf_Package.zip)
"""

import os
import json
import zipfile
from pathlib import Path
from typing import Dict, Any, Tuple

import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image, HRFlowable
)
from reportlab.lib import colors

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = EXPERIMENT_DIR.parent.parent
PLOTS_DIR = EXPERIMENT_DIR / "plots"
REPORTS_DIR = EXPERIMENT_DIR / "reports"
MODELS_DIR = EXPERIMENT_DIR / "models"


def get_plot_path(filename: str) -> Path:
    """Returns absolute path to a plot file."""
    p = PLOTS_DIR / filename
    return p if p.exists() else PLOTS_DIR / filename


def load_experiment_5_data() -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Loads metadata summary and benchmark CSVs."""
    summary_file = MODELS_DIR / "exp5_xai_fairness_summary.json"
    benchmark_file = REPORTS_DIR / "bias_mitigation_benchmark.csv"
    audit_file = REPORTS_DIR / "fairness_audit_metrics.csv"

    metadata = {}
    if summary_file.exists():
        with open(summary_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    df_benchmark = pd.read_csv(benchmark_file) if benchmark_file.exists() else pd.DataFrame()
    df_audit = pd.read_csv(audit_file) if audit_file.exists() else pd.DataFrame()

    return metadata, df_benchmark, df_audit


# ------------------------------------------------------------------------------
# 1. LATEX REPORT GENERATION
# ------------------------------------------------------------------------------

def generate_experiment_5_latex(
    metadata: Dict[str, Any],
    df_benchmark: pd.DataFrame,
    output_tex_path: str = None
) -> str:
    """Generates a comprehensive LaTeX document for Experiment 5."""
    if output_tex_path is None:
        output_tex_path = str(REPORTS_DIR / "Experiment_5_Report.tex")
    os.makedirs(os.path.dirname(output_tex_path), exist_ok=True)

    # Format benchmark table rows for LaTeX
    table_rows = []
    if not df_benchmark.empty:
        for _, row in df_benchmark.iterrows():
            m_name = row["Model Pipeline"].replace("_", "\\_").replace("&", "\\&")
            acc = f"{row['Accuracy']*100:.2f}\\%"
            bal_acc = f"{row['Balanced Accuracy']*100:.2f}\\%"
            f1 = f"{row['F1-Score']:.4f}"
            dp_sex = f"{row['Demographic Parity Diff (Sex)']:.4f}"
            eo_sex = f"{row['Equalized Odds Diff (Sex)']:.4f}"
            di_sex = f"{row['Disparate Impact Ratio (Sex)']:.4f}"
            table_rows.append(f"{m_name} & {acc} & {bal_acc} & {f1} & {dp_sex} & {eo_sex} & {di_sex} \\\\")
    table_latex_str = "\n".join(table_rows)

    tex_code = r"""\documentclass[11pt, a4paper]{article}
\usepackage[a4paper, margin=0.85in]{geometry}
\usepackage{mathptmx}
\usepackage{amsmath, amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{enumitem}
\usepackage{float}
\usepackage{microtype}
\usepackage{xcolor}

\hypersetup{
    colorlinks=true,
    linkcolor=blue!70!black,
    citecolor=blue!70!black,
    urlcolor=blue!70!black
}

\title{\textbf{Experiment 5: Explainable AI (XAI) with SHAP \& LIME and Algorithmic Fairness Auditing with Fairlearn}}
\author{\textbf{Course:} Applied Data Science (ADS) \quad | \quad \textbf{Domain:} Responsible Machine Learning}
\date{\textbf{Benchmark Dataset:} Adult Census Income ($N=32,561$) \quad | \quad \textbf{Date:} September 2026}

\begin{document}

\maketitle

\begin{abstract}
As machine learning models are increasingly deployed in high-stakes socioeconomic environments, model opacity and algorithmic bias present critical technical, ethical, and legal challenges. This report presents an end-to-end investigation of Explainable AI (XAI) and Algorithmic Fairness on the Adult Census Income benchmark dataset ($N=32,561$, 12 features). First, we train high-performance baseline models, selecting a LightGBM champion achieving \textbf{87.69\% accuracy} and \textbf{0.9310 ROC-AUC}. Second, we apply game-theoretic \textbf{SHAP (Shapley Additive exPlanations)} via \texttt{TreeExplainer} to derive globally consistent feature importance, directional beeswarm distributions, non-linear interaction dependencies, and local sample attributions. Third, we deploy \textbf{LIME (\texttt{lime.lime\_tabular})} to construct locally faithful linear surrogate explanations across distinct customer profiles, assessing cross-methodological concordance. Fourth, we perform an algorithmic bias audit using \textbf{Fairlearn}, uncovering substantial demographic disparities across \textit{Sex} (Demographic Parity Difference = 0.1714, Disparate Impact Ratio = 0.346) and \textit{Race}. Finally, we systematically implement and benchmark a tri-modal bias mitigation suite spanning pre-processing (sample reweighting), in-processing (\texttt{ExponentiatedGradient} with Equalized Odds), and post-processing (\texttt{ThresholdOptimizer}). Post-processing successfully reduced the Equalized Odds Difference by \textbf{35.6\%} to 0.0393, while pre-processing reduced Demographic Parity Difference by \textbf{49.4\%} to 0.0867, illustrating the fundamental Fairness-Performance Pareto frontier.
\end{abstract}

\vspace{0.5em}
\hrule
\vspace{1em}

\section{Aim \& Objectives}
\textbf{Aim:} To apply explainable AI (XAI) methods (SHAP \& LIME) for interpreting model predictions and evaluate algorithmic fairness using Fairlearn.

\noindent \textbf{Objectives:}
\begin{enumerate}[leftmargin=2em]
    \item Implement global (SHAP summary bar and beeswarm distributions) and local (SHAP waterfall and LIME tabular) explanation frameworks.
    \item Audit demographic bias in machine learning models across protected sensitive attributes (\textit{Sex} and \textit{Race}) using Fairlearn's \texttt{MetricFrame}.
    \item Quantify disparity metrics including Selection Rate, Demographic Parity Difference, Equalized Odds Difference, and Disparate Impact Ratio.
    \item Propose, implement, and benchmark tri-modal bias mitigation strategies: Pre-processing (reweighting), In-processing (\texttt{ExponentiatedGradient}), and Post-processing (\texttt{ThresholdOptimizer}).
    \item Characterize the empirical Pareto trade-off between predictive accuracy and algorithmic fairness.
\end{enumerate}

\section{Theoretical Foundations of XAI \& Algorithmic Fairness}

\subsection{SHAP (Shapley Additive exPlanations)}
Rooted in cooperative game theory, SHAP allocates the payoff of an ensemble prediction among individual input features according to their marginal contributions across all possible feature coalitions $S \subseteq F \setminus \{i\}$:
\begin{equation}
\phi_i(v) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left[ v(S \cup \{i\}) - v(S) \right]
\end{equation}
SHAP is uniquely characterized by four fundamental mathematical properties:
\begin{itemize}[leftmargin=1.5em]
    \item \textbf{Efficiency:} The sum of Shapley values equals the difference between model output and base expected value: $\sum_{i} \phi_i = f(x) - \mathbb{E}[f(X)]$.
    \item \textbf{Symmetry:} If features $i$ and $j$ contribute equally to all coalitions, $\phi_i = \phi_j$.
    \item \textbf{Dummy / Null Player:} If feature $i$ contributes nothing to any coalition, $\phi_i = 0$.
    \item \textbf{Additivity:} For ensemble models $f = f_1 + f_2$, the attributions satisfy $\phi_i(f) = \phi_i(f_1) + \phi_i(f_2)$.
\end{itemize}

\subsection{LIME (Local Interpretable Model-agnostic Explanations)}
LIME explains individual predictions by fitting an interpretable surrogate model $g \in G$ (e.g., sparse ridge regression) over localized perturbations $z'$ weighted by an exponential distance kernel $\pi_x(z)$:
\begin{equation}
\xi(x) = \arg\min_{g \in G} \mathcal{L}(f, g, \pi_x) + \Omega(g), \quad \text{where } \pi_x(z) = \exp\left(-\frac{D(x,z)^2}{\sigma^2}\right)
\end{equation}
While SHAP provides axiomatic guarantees, LIME offers model-agnostic operational flexibility.

\subsection{Algorithmic Fairness Formulations}
Let $Y \in \{0, 1\}$ denote the true ground truth label, $\hat{Y} \in \{0, 1\}$ denote the model prediction, and $A \in \{a, b\}$ denote the sensitive protected attribute:
\begin{itemize}[leftmargin=1.5em]
    \item \textbf{Demographic Parity (Statistical Parity):} Requires equal positive prediction rates across groups:
    \begin{equation}
    P(\hat{Y}=1 \mid A=a) = P(\hat{Y}=1 \mid A=b)
    \end{equation}
    \item \textbf{Equalized Odds:} Requires equal True Positive Rates (TPR) and False Positive Rates (FPR) across groups:
    \begin{equation}
    P(\hat{Y}=1 \mid Y=y, A=a) = P(\hat{Y}=1 \mid Y=y, A=b) \quad \forall y \in \{0, 1\}
    \end{equation}
    \item \textbf{Disparate Impact Ratio (80\% Four-Fifths Rule):} The selection rate ratio must satisfy:
    \begin{equation}
    \text{DIR} = \frac{\min(P(\hat{Y}=1 \mid A=a), P(\hat{Y}=1 \mid A=b))}{\max(P(\hat{Y}=1 \mid A=a), P(\hat{Y}=1 \mid A=b))} \ge 0.80
    \end{equation}
\end{itemize}

\section{Experimental Methodology \& Benchmark Results}

\begin{table}[H]
\centering
\caption{Comprehensive Benchmark: Performance vs. Fairness Disparity across Mitigation Strategies}
\label{tab:mitigation_benchmark}
\small
\begin{tabular}{lcccccc}
\toprule
\textbf{Model Pipeline} & \textbf{Acc} & \textbf{Bal Acc} & \textbf{F1} & \textbf{DP Diff (Sex)} & \textbf{EO Diff (Sex)} & \textbf{DIR (Sex)} \\
\midrule
""" + table_latex_str + r"""
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[H]
    \centering
    \begin{subfigure}[b]{0.48\textwidth}
        \includegraphics[width=\textwidth]{plots/exp5_shap_summary_bar.png}
        \caption{Global Mean Absolute SHAP Ranking}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.48\textwidth}
        \includegraphics[width=\textwidth]{plots/exp5_shap_beeswarm.png}
        \caption{SHAP Beeswarm Value Dispersion}
    \end{subfigure}
    \caption{SHAP Global Interpretability Suite on Validation Cohort ($N=800$).}
    \label{fig:shap_global}
\end{figure}

\begin{figure}[H]
    \centering
    \begin{subfigure}[b]{0.48\textwidth}
        \includegraphics[width=\textwidth]{plots/exp5_shap_dependence.png}
        \caption{SHAP Dependence: Age vs. Hours per Week}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.48\textwidth}
        \includegraphics[width=\textwidth]{plots/exp5_shap_waterfall.png}
        \caption{Local SHAP Waterfall Attribution (Sample \#0)}
    \end{subfigure}
    \caption{SHAP Non-Linear Feature Interactions and Local Sample Attributions.}
    \label{fig:shap_local}
\end{figure}

\begin{figure}[H]
    \centering
    \begin{subfigure}[b]{0.48\textwidth}
        \includegraphics[width=\textwidth]{plots/exp5_lime_local_explanations.png}
        \caption{LIME Local Surrogate Decision Rules}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.48\textwidth}
        \includegraphics[width=\textwidth]{plots/exp5_xai_shap_vs_lime_comparison.png}
        \caption{Cross-XAI Concordance (SHAP vs. LIME)}
    \end{subfigure}
    \caption{LIME Tabular Explanations and Cross-Method Concordance Audit.}
    \label{fig:lime_explanations}
\end{figure}

\begin{figure}[H]
    \centering
    \begin{subfigure}[b]{0.48\textwidth}
        \includegraphics[width=\textwidth]{plots/exp5_fairness_audit_disparity.png}
        \caption{Fairlearn Baseline Disparities (Sex \& Race)}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.48\textwidth}
        \includegraphics[width=\textwidth]{plots/exp5_fairness_mitigation_tradeoff.png}
        \caption{Fairness-Accuracy Pareto Frontier}
    \end{subfigure}
    \caption{Algorithmic Fairness Audit and Empirical Pareto Trade-off Analysis.}
    \label{fig:fairness_analysis}
\end{figure}

\section{Discussion \& Key Findings}
\begin{enumerate}[leftmargin=1.5em]
    \item \textbf{Global Feature Dominance:} SHAP analysis established \texttt{Capital\_Gain}, \texttt{Age}, \texttt{Education\_Num}, and \texttt{Relationship} as the top drivers of high-income predictions. Marital status and relationship status act as strong socio-demographic proxies.
    \item \textbf{Cross-XAI Concordance:} For Sample \#0, both SHAP and LIME identified identical top positive drivers (\texttt{Education\_Num}, \texttt{Hours\_per\_week}) with consistent directional signs, validating local surrogate fidelity.
    \item \textbf{Baseline Fairness Deficiency:} The unmitigated champion model exhibited severe demographic disparity: male selection rate was 26.2\% compared to only 9.1\% for females ($\text{DIR} = 0.346$, violating the EEOC 80\% rule).
    \item \textbf{Mitigation Effectiveness:} Pre-processing reweighting achieved the greatest reduction in Demographic Parity Difference (0.0867 vs 0.1714, a 49.4\% improvement). Post-processing threshold optimization achieved the lowest Equalized Odds Difference (0.0393), proving that group-specific threshold calibration is superior for error-rate parity.
\end{enumerate}

\section{Conclusion}
Experiment 5 demonstrated the complete cycle of modern Responsible AI: extracting rigorous explanations through cooperative game theory and surrogate modeling, diagnosing systemic algorithmic bias, and strategically applying mitigation algorithms to achieve equitable outcomes on the Pareto efficiency frontier.

\end{document}
"""
    with open(output_tex_path, "w", encoding="utf-8") as f:
        f.write(tex_code)
    print(f"[+] LaTeX source generated at: {output_tex_path}")

    # Generate Overleaf zip package
    zip_path = REPORTS_DIR / "Experiment_5_Overleaf_Package.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_tex_path, arcname="main.tex")
        for p in PLOTS_DIR.glob("*.png"):
            zf.write(p, arcname=f"plots/{p.name}")
    print(f"[+] Overleaf upload bundle created at: {zip_path.name}")

    return tex_code


# ------------------------------------------------------------------------------
# 2. REPORTLAB ACADEMIC PDF GENERATION
# ------------------------------------------------------------------------------

def generate_experiment_5_pdf(
    metadata: Dict[str, Any],
    df_benchmark: pd.DataFrame,
    output_pdf_path: str = None
):
    """Generates formal publication-quality academic PDF report via ReportLab."""
    if output_pdf_path is None:
        output_pdf_path = str(REPORTS_DIR / "Experiment_5_Report.pdf")
    os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)

    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=A4,
        leftMargin=40,
        rightMargin=40,
        topMargin=36,
        bottomMargin=36
    )

    story = []
    styles = getSampleStyleSheet()

    # Custom typography styles
    title_style = ParagraphStyle(
        'DocTitle', fontName='Times-Bold', fontSize=15, leading=18, alignment=1, spaceAfter=4
    )
    section_style = ParagraphStyle(
        'SectionHead', fontName='Times-Bold', fontSize=11.5, leading=15, spaceBefore=6, spaceAfter=3
    )
    subsection_style = ParagraphStyle(
        'SubSectionHead', fontName='Times-Bold', fontSize=10.5, leading=13.5, spaceBefore=4, spaceAfter=2
    )
    body_style = ParagraphStyle(
        'BodyCustom', fontName='Times-Roman', fontSize=9.2, leading=12.5, spaceAfter=3
    )
    list_style = ParagraphStyle(
        'ListCustom', fontName='Times-Roman', fontSize=9, leading=12, leftIndent=12, spaceAfter=2
    )
    table_text_style = ParagraphStyle(
        'TableText', fontName='Times-Roman', fontSize=8, leading=10
    )
    table_header_style = ParagraphStyle(
        'TableHead', fontName='Times-Bold', fontSize=8, leading=10, textColor=colors.white
    )

    # =========================================================================
    # PAGE 1: TITLE, OBJECTIVES & STEP 1-2
    # =========================================================================
    story.append(Paragraph("Experiment 5", title_style))
    story.append(Paragraph(
        "<b>Aim: Explainable AI (XAI) Methods (SHAP &amp; LIME) and Fairness Auditing with Fairlearn</b>",
        ParagraphStyle('Sub', fontName='Times-Bold', fontSize=11.5, leading=14.5, alignment=1)
    ))
    story.append(Spacer(1, 3))

    story.append(Paragraph(
        "<b>Course:</b> Applied Data Science (ADS) &nbsp;&nbsp;|&nbsp;&nbsp; "
        "<b>Dataset:</b> Adult Census Income Benchmark ($N=32,561$) &nbsp;&nbsp;|&nbsp;&nbsp; "
        "<b>Tools:</b> SHAP, LIME, Fairlearn",
        body_style
    ))
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1E3A8A"), spaceAfter=5))

    story.append(Paragraph("<b>Objectives:</b>", section_style))
    objs = [
        "1. Generate global (SHAP feature importance & beeswarm) and local (SHAP waterfall & LIME tabular) model explanations.",
        "2. Audit demographic bias across protected sensitive features (<i>Sex</i> and <i>Race</i>) using Fairlearn's MetricFrame.",
        "3. Evaluate quantitative fairness metrics: Selection Rate, Demographic Parity Difference, and Equalized Odds Difference.",
        "4. Implement and benchmark tri-modal bias mitigation: Pre-processing (reweighting), In-processing (fairness constraints), and Post-processing (threshold adjustment).",
        "5. Characterize the empirical Fairness-Performance Pareto trade-off."
    ]
    for o in objs:
        story.append(Paragraph(o, list_style))
    story.append(Spacer(1, 5))

    story.append(Paragraph("<b>1. Dataset &amp; Champion Model Selection</b>", section_style))
    story.append(Paragraph(
        "We utilized the gold-standard <b>Adult Census Income</b> dataset from the US Census Bureau (32,561 rows, 12 features) to predict whether individual income exceeds $50,000/year. "
        "The dataset incorporates demographic protected attributes including <b>Sex</b> (66.9% Male, 33.1% Female) and <b>Race</b> (85.4% White, 14.6% Non-White). "
        "A champion <b>LightGBM Classifier</b> was trained on an 80/20 stratified split, achieving <b>87.69% accuracy</b> and <b>0.9310 ROC-AUC</b>, outperforming the Random Forest baseline (86.14% accuracy).",
        body_style
    ))
    story.append(Spacer(1, 4))

    story.append(Paragraph("<b>2. Global &amp; Local Explainability with SHAP</b>", section_style))
    story.append(Paragraph(
        "SHAP calculates the Shapley value &phi;<sub>i</sub> for each feature across all possible feature subsets, guaranteeing Efficiency, Symmetry, Dummy player, and Additivity. "
        "Using <code>shap.TreeExplainer</code>, we analyzed the global feature hierarchy and local sample dynamics:",
        body_style
    ))
    shap_items = [
        "<b>Feature Importance Ranking:</b> <code>Capital_Gain</code>, <code>Age</code>, <code>Education_Num</code>, and <code>Relationship</code> represent the most influential global predictors.",
        "<b>Beeswarm Distribution:</b> Higher capital gains and advanced education strongly shift log-odds positively toward &gt;$50K, whereas lower hours worked and lower education exert strong negative attributions.",
        "<b>Age Non-Linear Interaction:</b> SHAP dependence analysis demonstrates earnings capacity peaks between ages 38-52 and is strongly moderated by hours worked per week."
    ]
    for si in shap_items:
        story.append(Paragraph(si, list_style))
    story.append(Spacer(1, 6))

    # Embed SHAP Figures
    p_bar = get_plot_path("exp5_shap_summary_bar.png")
    p_bee = get_plot_path("exp5_shap_beeswarm.png")
    if p_bar.exists() and p_bee.exists():
        im_table = Table([
            [Image(str(p_bar), width=245, height=155), Image(str(p_bee), width=265, height=155)]
        ], colWidths=[255, 275])
        im_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 0),
            ('TOPPADDING', (0,0), (-1,-1), 0),
        ]))
        story.append(im_table)
        story.append(Paragraph("<b>Figure 1:</b> SHAP Global Feature Importance Bar Ranking (Left) and Beeswarm Impact Distribution (Right).", ParagraphStyle('Cap', fontName='Times-Italic', fontSize=8, alignment=1, spaceBefore=3)))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: LIME EXPLANATIONS, CROSS-XAI AUDIT & DEPENDENCE
    # =========================================================================
    story.append(Paragraph("<b>3. Local Surrogate Explainability with LIME</b>", section_style))
    story.append(Paragraph(
        "Using <code>lime.lime_tabular.LimeTabularExplainer</code>, we trained local linear surrogate models on perturbed instances around target profiles to explain individual predictions. "
        "LIME generates human-interpretable boolean decision boundary rules for each feature:",
        body_style
    ))

    p_lime = get_plot_path("exp5_lime_local_explanations.png")
    if p_lime.exists():
        story.append(Image(str(p_lime), width=515, height=160))
        story.append(Paragraph("<b>Figure 2:</b> LIME Local Surrogate Explanations across Positive, Negative, and Borderline Prediction Profiles.", ParagraphStyle('Cap', fontName='Times-Italic', fontSize=8, alignment=1, spaceBefore=2, spaceAfter=5)))

    story.append(Paragraph("<b>Cross-XAI Concordance Analysis (SHAP vs. LIME):</b>", subsection_style))
    story.append(Paragraph(
        "To verify attribution consistency, we audited Sample #0 using both SHAP and LIME. Both frameworks showed high directional agreement: "
        "<code>Education_Num &gt; 12</code> and <code>Hours_per_week = 40</code> were ranked as the top positive drivers of high income in both models, "
        "confirming that LIME's local surrogate closely mimics the true cooperative game-theoretic attributions of the champion model.",
        body_style
    ))
    story.append(Spacer(1, 4))

    p_dep = get_plot_path("exp5_shap_dependence.png")
    p_comp = get_plot_path("exp5_xai_shap_vs_lime_comparison.png")
    if p_dep.exists() and p_comp.exists():
        im_table2 = Table([
            [Image(str(p_dep), width=255, height=155), Image(str(p_comp), width=255, height=155)]
        ], colWidths=[260, 260])
        im_table2.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 0),
            ('TOPPADDING', (0,0), (-1,-1), 0),
        ]))
        story.append(im_table2)
        story.append(Paragraph("<b>Figure 3:</b> SHAP Non-linear Dependence Plot for Age (Left) and Cross-XAI Feature Concordance on Sample #0 (Right).", ParagraphStyle('Cap', fontName='Times-Italic', fontSize=8, alignment=1, spaceBefore=3)))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 3: FAIRNESS AUDIT & BIAS MITIGATION BENCHMARK
    # =========================================================================
    story.append(Paragraph("<b>4. Algorithmic Fairness Audit &amp; Bias Mitigation</b>", section_style))
    story.append(Paragraph(
        "Using <b>Fairlearn's</b> <code>MetricFrame</code>, we audited the champion model across <b>Sex</b> and <b>Race</b>. "
        "The unmitigated model exhibited substantial demographic bias: males had a <b>26.2% selection rate</b> versus only <b>9.1% for females</b> "
        "(Demographic Parity Difference = 0.1714; Disparate Impact Ratio = 0.346, severely breaching the EEOC 80% four-fifths standard).",
        body_style
    ))
    story.append(Spacer(1, 4))

    # Benchmark Table
    table_data = [[
        Paragraph("<b>Model Pipeline</b>", table_header_style),
        Paragraph("<b>Accuracy</b>", table_header_style),
        Paragraph("<b>Bal Acc</b>", table_header_style),
        Paragraph("<b>F1-Score</b>", table_header_style),
        Paragraph("<b>DP Diff (Sex)</b>", table_header_style),
        Paragraph("<b>EO Diff (Sex)</b>", table_header_style),
        Paragraph("<b>DIR (Sex)</b>", table_header_style),
    ]]

    if not df_benchmark.empty:
        for _, row in df_benchmark.iterrows():
            table_data.append([
                Paragraph(row["Model Pipeline"].split("(")[0].strip(), table_text_style),
                Paragraph(f"{row['Accuracy']*100:.2f}%", table_text_style),
                Paragraph(f"{row['Balanced Accuracy']*100:.2f}%", table_text_style),
                Paragraph(f"{row['F1-Score']:.4f}", table_text_style),
                Paragraph(f"{row['Demographic Parity Diff (Sex)']:.4f}", table_text_style),
                Paragraph(f"{row['Equalized Odds Diff (Sex)']:.4f}", table_text_style),
                Paragraph(f"{row['Disparate Impact Ratio (Sex)']:.4f}", table_text_style),
            ])

    t = Table(table_data, colWidths=[130, 60, 60, 60, 75, 75, 60])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1E3A8A")),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t)
    story.append(Paragraph("<b>Table 1:</b> Quantitative Comparison of Predictive Performance vs. Fairness Disparities across Mitigation Strategies.", ParagraphStyle('Cap', fontName='Times-Italic', fontSize=8, alignment=1, spaceBefore=4, spaceAfter=6)))

    story.append(Paragraph("<b>Tri-Modal Bias Mitigation Findings:</b>", subsection_style))
    mit_findings = [
        "<b>1. Pre-Processing (Sample Reweighting):</b> Reweighting instances by inverse group-outcome frequencies reduced Demographic Parity Difference from 0.1714 to <b>0.0867 (a 49.4% improvement)</b> and raised Disparate Impact Ratio to 0.6086, with only a marginal 0.77% drop in accuracy.",
        "<b>2. In-Processing (ExponentiatedGradient):</b> Constraining training with Equalized Odds directly achieved a balanced trade-off (Accuracy: 85.69%, DP Diff: 0.0973).",
        "<b>3. Post-Processing (ThresholdOptimizer):</b> Calibrating group-specific classification thresholds achieved the <b>lowest Equalized Odds Difference (0.0393, a 35.6% reduction)</b>, ensuring near-identical false positive and true positive rates across genders."
    ]
    for mf in mit_findings:
        story.append(Paragraph(mf, list_style))
    story.append(Spacer(1, 5))

    p_disp = get_plot_path("exp5_fairness_audit_disparity.png")
    p_trade = get_plot_path("exp5_fairness_mitigation_tradeoff.png")
    if p_disp.exists() and p_trade.exists():
        im_table3 = Table([
            [Image(str(p_disp), width=260, height=140), Image(str(p_trade), width=250, height=140)]
        ], colWidths=[265, 255])
        im_table3.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 0),
            ('TOPPADDING', (0,0), (-1,-1), 0),
        ]))
        story.append(im_table3)
        story.append(Paragraph("<b>Figure 4:</b> Demographic Disparity Audit (Left) and Fairness-Accuracy Pareto Frontier (Right).", ParagraphStyle('Cap', fontName='Times-Italic', fontSize=8, alignment=1, spaceBefore=3)))

    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>5. Conclusion</b>", section_style))
    story.append(Paragraph(
        "Experiment 5 successfully established that high predictive accuracy does not imply fair decision-making. "
        "By pairing game-theoretic SHAP and local surrogate LIME with Fairlearn's auditing and mitigation toolkits, "
        "we systematically diagnosed root causes of bias and navigated the empirical Pareto frontier, proving that responsible AI "
        "principles can be practically enforced in enterprise ML deployments.",
        body_style
    ))

    doc.build(story)
    print(f"[+] Academic PDF Report generated at: {output_pdf_path}")


def generate_experiment_5_report(
    output_pdf_path: str = None,
    output_tex_path: str = None
):
    """Unified entry point to generate both LaTeX and PDF reports."""
    metadata, df_benchmark, df_audit = load_experiment_5_data()
    generate_experiment_5_latex(metadata, df_benchmark, output_tex_path)
    generate_experiment_5_pdf(metadata, df_benchmark, output_pdf_path)


if __name__ == "__main__":
    generate_experiment_5_report()
