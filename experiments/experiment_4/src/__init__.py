"""
Experiment 4 Source Module
"""
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = EXPERIMENT_DIR.parent.parent

from .experiment_4_modeling import (
    train_and_track_experiments,
    load_and_prepare_data,
    extract_features,
    EndToEndEmotionPipeline,
)
from .generate_experiment_4_report import generate_experiment_4_report, generate_experiment_4_latex
from .generate_experiment_4_docx import generate_experiment_4_docx

__all__ = [
    "EXPERIMENT_DIR",
    "WORKSPACE_ROOT",
    "train_and_track_experiments",
    "load_and_prepare_data",
    "extract_features",
    "EndToEndEmotionPipeline",
    "generate_experiment_4_report",
    "generate_experiment_4_latex",
    "generate_experiment_4_docx",
]
