"""
Experiment 5: Explainable AI (XAI) with SHAP & LIME and Fairness Auditing with Fairlearn
Course: Applied Data Science (ADS)
"""

from experiments.experiment_5.src.experiment_5_xai_fairness import (
    run_experiment_5_pipeline,
    load_and_prepare_census_data,
    train_unmitigated_models,
    audit_model_fairness,
    run_shap_explainability,
    run_lime_explainability,
    run_bias_mitigation,
    load_experiment_5_summary,
)

__all__ = [
    "run_experiment_5_pipeline",
    "load_and_prepare_census_data",
    "train_unmitigated_models",
    "audit_model_fairness",
    "run_shap_explainability",
    "run_lime_explainability",
    "run_bias_mitigation",
    "load_experiment_5_summary",
]
