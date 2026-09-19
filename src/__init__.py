"""
Root package shim re-exporting core modules across experiments for backwards compatibility.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Re-export Experiment 2
from experiments.experiment_2.src.emotion_labeler import EmotionLabeler, EMOTION_CLASSES
from experiments.experiment_2.src.preprocessing import clean_tweet_text, preprocess_twcs, apply_negation_tagging
from experiments.experiment_2.src.train import MultiLabelEmotionPipeline, train_and_evaluate
from experiments.experiment_2.src.predict import EmotionPredictor

# Backward compatibility alias
EmotionPipeline = MultiLabelEmotionPipeline

# Re-export Experiment 3
from experiments.experiment_3.src.eda_analysis import run_experiment_3, prepare_eda_dataset
from experiments.experiment_3.src.generate_report import generate_detailed_pdf

# Re-export Experiment 4
from experiments.experiment_4.src.experiment_4_modeling import train_and_track_experiments
from experiments.experiment_4.src.generate_experiment_4_report import generate_experiment_4_report
from experiments.experiment_4.src.generate_experiment_4_docx import generate_experiment_4_docx

# Re-export Experiment 5
from experiments.experiment_5.src.experiment_5_xai_fairness import (
    run_experiment_5_pipeline,
    load_and_prepare_census_data,
    train_unmitigated_models,
    audit_model_fairness,
    run_shap_explainability,
    run_lime_explainability,
    run_bias_mitigation,
    load_experiment_5_summary
)
from experiments.experiment_5.src.generate_experiment_5_report import generate_experiment_5_report as generate_experiment_5_pdf_report
from experiments.experiment_5.src.generate_experiment_5_docx import generate_experiment_5_docx

# Re-export Experiment 6
from experiments.experiment_6.src.app import app
from experiments.experiment_6.src.test_api import run_api_tests
from experiments.experiment_6.src.generate_experiment_6_report import generate_experiment_6_report as generate_experiment_6_pdf_report
from experiments.experiment_6.src.generate_experiment_6_docx import generate_experiment_6_docx

__all__ = [
    "PROJECT_ROOT",
    # Experiment 2
    "EmotionLabeler",
    "EMOTION_CLASSES",
    "clean_tweet_text",
    "apply_negation_tagging",
    "preprocess_twcs",
    "MultiLabelEmotionPipeline",
    "EmotionPipeline",
    "train_and_evaluate",
    "EmotionPredictor",
    # Experiment 3
    "run_experiment_3",
    "prepare_eda_dataset",
    "generate_detailed_pdf",
    # Experiment 4
    "train_and_track_experiments",
    "generate_experiment_4_report",
    "generate_experiment_4_docx",
    # Experiment 5
    "run_experiment_5_pipeline",
    "load_and_prepare_census_data",
    "train_unmitigated_models",
    "audit_model_fairness",
    "run_shap_explainability",
    "run_lime_explainability",
    "run_bias_mitigation",
    "load_experiment_5_summary",
    "generate_experiment_5_pdf_report",
    "generate_experiment_5_docx",
    # Experiment 6
    "app",
    "run_api_tests",
    "generate_experiment_6_pdf_report",
    "generate_experiment_6_docx",
]


