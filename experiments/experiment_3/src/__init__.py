"""
Experiment 3 Source Module
"""
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = EXPERIMENT_DIR.parent.parent

from .eda_analysis import run_experiment_3, prepare_eda_dataset
from .generate_report import generate_detailed_pdf

__all__ = [
    "EXPERIMENT_DIR",
    "WORKSPACE_ROOT",
    "run_experiment_3",
    "prepare_eda_dataset",
    "generate_detailed_pdf",
]
