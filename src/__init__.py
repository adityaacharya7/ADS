"""
Applied Data Science (ADS) - Twitter Customer Support Emotion & Sentiment Analysis Toolkit
"""

from pathlib import Path

# Base project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

from .emotion_labeler import EmotionLabeler, EMOTION_CLASSES
from .preprocessing import clean_tweet_text, preprocess_twcs, apply_negation_tagging
from .train import MultiLabelEmotionPipeline, train_and_evaluate
from .predict import EmotionPredictor

# Backward compatibility alias
EmotionPipeline = MultiLabelEmotionPipeline

__all__ = [
    "PROJECT_ROOT",
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
