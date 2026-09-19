"""
Experiment 2 Source Module
"""
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = EXPERIMENT_DIR.parent.parent

from .emotion_labeler import EmotionLabeler, EMOTION_CLASSES
from .preprocessing import clean_tweet_text, preprocess_twcs, apply_negation_tagging
from .train import MultiLabelEmotionPipeline, train_and_evaluate
from .predict import EmotionPredictor

# Backward compatibility alias
EmotionPipeline = MultiLabelEmotionPipeline

__all__ = [
    "EXPERIMENT_DIR",
    "WORKSPACE_ROOT",
    "EmotionLabeler",
    "EMOTION_CLASSES",
    "clean_tweet_text",
    "apply_negation_tagging",
    "preprocess_twcs",
    "MultiLabelEmotionPipeline",
    "EmotionPipeline",
    "train_and_evaluate",
    "EmotionPredictor",
]
