import sys
import os
import argparse
import joblib
import pandas as pd
import numpy as np

from emotion_labeler import EmotionLabeler
from preprocess_twcs import clean_tweet_text
from train_emotion_model import EmotionPipeline

EMOJI_MAP = {
    "Anger / Frustration": "😡",
    "Disappointment / Sadness": "😞",
    "Fear / Anxiety": "😨",
    "Joy / Gratitude": "😊",
    "Neutral / Inquiry": "😐"
}

class EmotionPredictor:
    """
    Inference interface for loading trained Emotion Pipeline and predicting emotions.
    """
    def __init__(self, model_path: str = "models/emotion_pipeline.joblib"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file '{model_path}' not found! Please run `python train_emotion_model.py` first."
            )
        self.pipeline = joblib.load(model_path)
        self.labeler = EmotionLabeler()

    def predict_one(self, raw_text: str):
        cleaned = clean_tweet_text(raw_text)
        vader_res = self.labeler.analyze_tweet(cleaned)
        
        vader_feats = np.array([[
            vader_res['compound'],
            vader_res['pos'],
            vader_res['neg'],
            vader_res['neu']
        ]])
        
        predicted_class = self.pipeline.predict([cleaned], vader_feats)[0]
        probas = self.pipeline.predict_proba([cleaned], vader_feats)[0]
        classes = self.pipeline.label_encoder.classes_
        
        prob_dict = {cls: float(prob) for cls, prob in zip(classes, probas)}
        confidence = prob_dict[predicted_class]
        
        return {
            "raw_text": raw_text,
            "cleaned_text": cleaned,
            "predicted_emotion": predicted_class,
            "emoji": EMOJI_MAP.get(predicted_class, "❓"),
            "confidence": confidence,
            "sentiment": vader_res['sentiment'],
            "vader_compound": vader_res['compound'],
            "probabilities": prob_dict
        }

    def print_prediction(self, raw_text: str):
        res = self.predict_one(raw_text)
        print("=" * 65)
        print(f" Input Text  : \"{res['raw_text']}\"")
        if res['cleaned_text'] != res['raw_text']:
            print(f" Cleaned     : \"{res['cleaned_text']}\"")
        print("-" * 65)
        print(f" Predicted Emotion : {res['emoji']}  {res['predicted_emotion']} (Confidence: {res['confidence']*100:.1f}%)")
        print(f" Sentiment Polarity: {res['sentiment']} (Compound Score: {res['vader_compound']:+.3f})")
        print("\n Class Probabilities:")
        
        sorted_probs = sorted(res['probabilities'].items(), key=lambda x: x[1], reverse=True)
        for cls, prob in sorted_probs:
            bar_len = int(prob * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            print(f"  {EMOJI_MAP.get(cls, ' ')} {cls:25s} | {bar} | {prob*100:5.1f}%")
        print("=" * 65)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Real-Time Tweet Sentiment & Emotion Predictor")
    parser.add_argument("text", nargs="?", type=str, help="Text/Tweet to analyze")
    parser.add_argument("--model", type=str, default="models/emotion_pipeline.joblib", help="Path to trained model pipeline")
    
    args = parser.parse_args()
    
    predictor = EmotionPredictor(args.model)
    
    if args.text:
        predictor.print_prediction(args.text)
    else:
        # Run demo on sample customer tweets
        test_samples = [
            "My PlayStation account is giving error code ws-37338-4! What is going on? Please respond!",
            "I've been waiting 3 weeks for my package and customer service keeps hanging up! I am furious!",
            "Flight IB7449 was cancelled suddenly without notice. I lost all my hotel bookings, so sad.",
            "I received an email stating my password was compromised in a security breach. Am I hacked?!",
            "Shoutout to the amazing Twitter support representative who resolved my refund issue in 5 minutes! Thank you so much! ❤️"
        ]
        print("No text provided. Running demo on 5 sample customer support tweets...\n")
        for sample in test_samples:
            predictor.print_prediction(sample)
            print()
