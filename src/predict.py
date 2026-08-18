import sys
import os
import argparse
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Reconfigure stdout to utf-8 if possible on Windows
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from .emotion_labeler import EmotionLabeler, EMOTION_CLASSES
from .preprocessing import clean_tweet_text, apply_negation_tagging
from .train import MultiLabelEmotionPipeline

# Unpickling compatibility mappings across namespace environments
sys.modules['__main__'].MultiLabelEmotionPipeline = MultiLabelEmotionPipeline
if 'train_emotion_model' not in sys.modules:
    import types
    fake_mod = types.ModuleType('train_emotion_model')
    fake_mod.MultiLabelEmotionPipeline = MultiLabelEmotionPipeline
    sys.modules['train_emotion_model'] = fake_mod

EMOJI_MAP = {
    "Joy / Gratitude": "😊",
    "Anger / Frustration": "😡",
    "Disappointment / Sadness": "😞",
    "Fear / Anxiety": "😨",
    "Neutral / Inquiry": "😐"
}

ASCII_MAP = {
    "Joy / Gratitude": "[JOY]",
    "Anger / Frustration": "[ANGER]",
    "Disappointment / Sadness": "[SADNESS]",
    "Fear / Anxiety": "[FEAR]",
    "Neutral / Inquiry": "[NEUTRAL]"
}


class EmotionPredictor:
    """
    Inference engine for Negation-Aware Multi-Label Emotion Profiling.
    """
    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = str(PROJECT_ROOT / "models" / "emotion_pipeline.joblib")
            
        self.labeler = EmotionLabeler()
        self.pipeline = None
        
        if os.path.exists(model_path):
            try:
                self.pipeline = joblib.load(model_path)
            except Exception as e:
                print(f"[*] Note: Loading model with labeler engine fallback ({e})")
                self.pipeline = None
        else:
            print(f"[*] Note: Model '{model_path}' not found yet. Using rule & lexicon engine.")

    def predict_one(self, raw_text: str):
        cleaned = clean_tweet_text(raw_text)
        negated = apply_negation_tagging(cleaned)
        vader_res = self.labeler.analyze_tweet(cleaned)
        
        if self.pipeline is not None and hasattr(self.pipeline, "predict_profile"):
            profile = self.pipeline.predict_profile(raw_text, vader_res)
            return profile
        elif self.pipeline is not None and hasattr(self.pipeline, "predict_proba"):
            vader_feats = np.array([[
                vader_res['compound'],
                vader_res['pos'],
                vader_res['neg'],
                vader_res['neu']
            ]])
            probas = self.pipeline.predict_proba([cleaned], vader_feats)[0]
            classes = getattr(self.pipeline, "emotion_classes", EMOTION_CLASSES)
            prob_dict = {cls: float(np.clip(p, 0.0, 1.0)) for cls, p in zip(classes, probas)}
            
            sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
            primary_name, primary_prob = sorted_probs[0]
            second_name, second_prob = sorted_probs[1]
            is_mixed = (second_prob >= 0.20)
            
            return {
                "raw_text": raw_text,
                "cleaned_text": cleaned,
                "negated_text": negated,
                "primary_emotion": primary_name,
                "primary_confidence": primary_prob,
                "secondary_emotion": second_name if is_mixed else "None detected above threshold",
                "secondary_confidence": second_prob if is_mixed else 0.0,
                "is_mixed": is_mixed,
                "probabilities": prob_dict,
                "vader_compound": vader_res['compound'],
                "sentiment": vader_res['sentiment']
            }
        else:
            # Fallback to EmotionLabeler engine
            scores = vader_res['emotion_scores']
            return {
                "raw_text": raw_text,
                "cleaned_text": cleaned,
                "negated_text": negated,
                "primary_emotion": vader_res['primary_emotion'],
                "primary_confidence": scores[vader_res['primary_emotion']],
                "secondary_emotion": vader_res['secondary_emotion'] if vader_res['is_mixed'] else "None detected above threshold",
                "secondary_confidence": scores.get(vader_res['secondary_emotion'], 0.0) if vader_res['is_mixed'] else 0.0,
                "is_mixed": vader_res['is_mixed'],
                "probabilities": scores,
                "vader_compound": vader_res['compound'],
                "sentiment": vader_res['sentiment']
            }

    def print_prediction(self, raw_text: str):
        res = self.predict_one(raw_text)
        
        # Test emoji encoding support
        use_emoji = True
        try:
            "😊".encode(sys.stdout.encoding or 'utf-8')
        except Exception:
            use_emoji = False

        bar_char = "█" if use_emoji else "="
        empty_char = "░" if use_emoji else "-"

        prim_icon = EMOJI_MAP.get(res['primary_emotion'], " ") if use_emoji else ASCII_MAP.get(res['primary_emotion'], "")
        has_secondary = res['secondary_emotion'] != "None detected above threshold"
        sec_icon = EMOJI_MAP.get(res['secondary_emotion'], " ") if (use_emoji and has_secondary) else ""

        print("=" * 68)
        print(f" Input Text : \"{res['raw_text']}\"")
        if res['cleaned_text'] != res['raw_text']:
            print(f" Cleaned    : \"{res['cleaned_text']}\"")
        if res['negated_text'] != res['cleaned_text'].lower():
            print(f" Negated    : \"{res['negated_text']}\"")
        print("-" * 68)
        
        print(f" Dominant Emotion                     : {prim_icon} {res['primary_emotion']} ({res['primary_confidence']*100:.1f}%)")
        if has_secondary:
            print(f" Significant Co-existing Emotion(s)   : {sec_icon} {res['secondary_emotion']} ({res['secondary_confidence']*100:.1f}%)")
        else:
            print(f" Significant Co-existing Emotion(s)   : None detected above threshold (threshold: 20.0%)")
            
        print(f" Sentiment Polarity                   : {res['sentiment']} (Compound Score: {res['vader_compound']:+.3f})")
        if res['is_mixed']:
            print(f" Emotional State                      : Multi-Dimensional / Co-existing Emotions")
        else:
            print(f" Emotional State                      : Single Dominant State")
            
        print("\n Multi-Dimensional Independent Activation Profile:")
        sorted_probs = sorted(res['probabilities'].items(), key=lambda x: x[1], reverse=True)
        for cls, prob in sorted_probs:
            cls_icon = EMOJI_MAP.get(cls, ' ') if use_emoji else ASCII_MAP.get(cls, ' ')
            bar_len = int(np.clip(prob, 0.0, 1.0) * 30)
            bar = bar_char * bar_len + empty_char * (30 - bar_len)
            print(f"  {cls_icon:4s} {cls:25s} | {bar} | {prob*100:5.1f}%")
        print("=" * 68)


def run_demo():
    predictor = EmotionPredictor()
    test_suites = {
        "1. Negation Handling & Scope Boundary Tests": [
            "I am disappointed with this service.",
            "I am not disappointed with this service.",
            "I am furious about this billing issue.",
            "I am not angry at all, just asking for a status update.",
            "I don't feel anxious about this error code anymore."
        ],
        "2. Complex Bounded Negation & Contrast Tests": [
            "I wasn't afraid of the outcome, and although everyone around me seemed nervous, I remained surprisingly calm and confident throughout the entire process.",
            "I was afraid of the outcome, but I tried to remain calm and confident throughout the entire process.",
            "I thought I would be furious, but instead I felt an overwhelming sense of relief and pride, though I still could not shake the fear that everything we had achieved might disappear."
        ],
        "3. Real Customer Support Queries": [
            "Flight IB7449 was cancelled suddenly without notice. I lost all my hotel bookings, so sad.",
            "I received an email stating my password was compromised in a security breach. Am I hacked?!",
            "Shoutout to the amazing Twitter support representative who resolved my refund issue in 5 minutes! Thank you so much!"
        ]
    }
    
    print("=" * 68)
    print(" NEGATION-AWARE MULTI-LABEL EMOTION ANALYSIS COMPREHENSIVE SUITE ")
    print("=" * 68)
    
    for section_name, samples in test_suites.items():
        print(f"\n>>> {section_name}\n")
        for sample in samples:
            predictor.print_prediction(sample)
            print()


if __name__ == '__main__':
    default_model = str(PROJECT_ROOT / "models" / "emotion_pipeline.joblib")
    
    parser = argparse.ArgumentParser(description="Real-Time Negation-Aware Multi-Label Emotion Profiler")
    parser.add_argument("pos_text", nargs="?", type=str, default=None, help="Tweet text (positional)")
    parser.add_argument("--text", "-t", type=str, default=None, help="Tweet text to analyze")
    parser.add_argument("--model", "-m", type=str, default=default_model, help="Path to trained model pipeline")
    parser.add_argument("--interactive", action="store_true", help="Start interactive prediction CLI session")
    
    args = parser.parse_args()
    text = args.text or args.pos_text
    
    predictor = EmotionPredictor(args.model)
    
    if args.interactive:
        print("Starting interactive Emotion Prediction session (type 'exit' or 'quit' to stop):")
        while True:
            try:
                user_input = input("\nEnter tweet text: ").strip()
                if user_input.lower() in ('exit', 'quit', 'q'):
                    break
                if user_input:
                    predictor.print_prediction(user_input)
            except (KeyboardInterrupt, EOFError):
                break
    elif text:
        predictor.print_prediction(text)
    else:
        run_demo()
