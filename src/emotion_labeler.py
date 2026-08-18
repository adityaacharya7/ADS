import re
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .preprocessing import apply_negation_tagging, expand_contractions

EMOTION_CLASSES = [
    "Joy / Gratitude",
    "Anger / Frustration",
    "Disappointment / Sadness",
    "Fear / Anxiety",
    "Neutral / Inquiry"
]


class EmotionLabeler:
    """
    Negation-aware, multi-label sentiment and emotion labeling engine
    tailored for complex, nuanced customer support and conversational interactions.
    """
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        
        # Unnegated Lexicon Patterns
        self.joy_patterns = re.compile(
            r'\b(thank|thanks|thankyou|awesome|great|amazing|excellent|wonderful|love|loved|'
            r'perfect|happy|kudos|solved|helpful|best|bless|blessed|appreciated|appreciate|'
            r'fantastic|brilliant|rockstar|applause|proud|proudly|delighted|glad|relieved|relief|'
            r'pleased|exceeded|thrilled|excited|exciting|excitement|eager|satisfaction|satisfied|'
            r'calm|confident|confidence|composed|at ease|reassured)\b',
            re.IGNORECASE
        )
        
        self.anger_patterns = re.compile(
            r'\b(angry|anger|furious|fury|terrible|worst|horrible|frustrated|frustrating|unacceptable|garbage|'
            r'scam|ridiculous|useless|hate|broken|fail|fails|failed|sucks|annoyed|annoying|never again|'
            r'disgusting|pathetic|scammed|appalling|trash|shameful|infuriating|livid|outrage|outrageous)\b',
            re.IGNORECASE
        )
        
        self.sadness_patterns = re.compile(
            r'\b(sad|sadness|disappointed|disappointing|disappointment|bummer|upset|regret|missed|missing|'
            r'lost|ruined|refund|cancel|cancelled|cancellation|heartbroken|pity|unfortunate|'
            r'unhappy|loss|losing|hopeless|depressing|down|dismayed|regretful|sorrow|grief)\b',
            re.IGNORECASE
        )
        
        self.fear_patterns = re.compile(
            r'\b(fear|fearful|feared|fears|worried|worry|scared|afraid|anxious|anxiety|hacked|security|breach|compromised|'
            r'stolen|danger|urgent|panic|leak|fraud|warning|threat|locked out|identity theft|'
            r'suspicious|unsafe|vulnerability|nervous|terrified|alarmed|dread|apprehensive|disappear|disappearing)\b',
            re.IGNORECASE
        )
        
        # Contrast & Mixed Emotion Markers
        self.contrast_patterns = re.compile(
            r'\b(but|however|although|though|yet|while|nevertheless|despite|in spite of|on the other hand|instead)\b',
            re.IGNORECASE
        )

    def analyze_tweet(self, text: str) -> Dict[str, Any]:
        """
        Analyzes tweet with negation scope tracking and multi-label scoring.
        Returns independent probability/intensity scores for each of the 5 emotion classes.
        """
        if not isinstance(text, str) or not text.strip():
            return {
                "sentiment": "Neutral",
                "compound": 0.0,
                "pos": 0.0,
                "neg": 0.0,
                "neu": 1.0,
                "emotion": "Neutral / Inquiry",
                "primary_emotion": "Neutral / Inquiry",
                "secondary_emotion": "None detected above threshold",
                "is_mixed": False,
                "emotion_scores": {cls: (1.0 if cls == "Neutral / Inquiry" else 0.0) for cls in EMOTION_CLASSES},
                "multi_labels": {cls: (1 if cls == "Neutral / Inquiry" else 0) for cls in EMOTION_CLASSES},
                "label_vector": [0, 0, 0, 0, 1]
            }

        # Apply negation scope tagging
        negated_text = apply_negation_tagging(text)
        expanded_clean = expand_contractions(text)

        # 1. Compute VADER sentiment scores
        vader_scores = self.vader.polarity_scores(text)
        compound = vader_scores['compound']
        pos = vader_scores['pos']
        neg = vader_scores['neg']
        neu = vader_scores['neu']

        # Determine Valence Sentiment
        if compound >= 0.05:
            sentiment = "Positive"
        elif compound <= -0.05:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        # 2. Extract unnegated and negated keyword matches
        tokens = negated_text.split()
        unnegated_tokens = " ".join([t for t in tokens if not t.endswith("_NEG")])
        negated_tokens = " ".join([t[:-4] for t in tokens if t.endswith("_NEG")])

        joy_unneg = len(self.joy_patterns.findall(unnegated_tokens))
        joy_neg = len(self.joy_patterns.findall(negated_tokens))

        anger_unneg = len(self.anger_patterns.findall(unnegated_tokens))
        anger_neg = len(self.anger_patterns.findall(negated_tokens))

        sadness_unneg = len(self.sadness_patterns.findall(unnegated_tokens))
        sadness_neg = len(self.sadness_patterns.findall(negated_tokens))

        fear_unneg = len(self.fear_patterns.findall(unnegated_tokens))
        fear_neg = len(self.fear_patterns.findall(negated_tokens))

        has_contrast = bool(self.contrast_patterns.search(expanded_clean))

        # Counterfactual/hypothetical check: "thought I would be furious, but..."
        hypo_anger = len(re.findall(r'\b(would be|thought i would be|expected to be|might be)\s+\w*\s*(angry|furious|frustrated|livid)', expanded_clean))
        hypo_sadness = len(re.findall(r'\b(would be|thought i would be|expected to be|might be)\s+\w*\s*(sad|disappointed|upset)', expanded_clean))
        if hypo_anger > 0 and has_contrast:
            anger_unneg = max(0, anger_unneg - hypo_anger)
        if hypo_sadness > 0 and has_contrast:
            sadness_unneg = max(0, sadness_unneg - hypo_sadness)

        # 3. Compute continuous independent emotion intensity scores [0.0 - 1.0]
        # Joy Score
        joy_raw = (joy_unneg * 0.45) + (max(0.0, compound) * 0.45) + (pos * 0.35) - (joy_neg * 0.40)
        if sadness_neg > 0 or anger_neg > 0 or fear_neg > 0:
            joy_raw += 0.20
        joy_score = float(np.clip(joy_raw, 0.0, 1.0))

        # Anger Score
        if anger_neg > 0:
            anger_raw = max(0.0, (anger_unneg - anger_neg * 1.5) * 0.50 + max(0.0, -compound) * 0.15)
        else:
            anger_raw = (anger_unneg * 0.50) + (max(0.0, -compound) * 0.35) + (neg * 0.35)
        anger_score = float(np.clip(anger_raw, 0.0, 1.0))

        # Sadness / Disappointment Score
        if sadness_neg > 0:
            sadness_raw = max(0.0, (sadness_unneg - sadness_neg * 1.5) * 0.50 + max(0.0, -compound) * 0.15)
        else:
            sadness_raw = (sadness_unneg * 0.45) + (max(0.0, -compound) * 0.30) + (neg * 0.30)
        if joy_neg > 0:
            sadness_raw += 0.30
        sadness_score = float(np.clip(sadness_raw, 0.0, 1.0))

        # Fear / Anxiety Score
        if fear_neg > 0:
            fear_raw = max(0.0, (fear_unneg - fear_neg * 1.5) * 0.50 + max(0.0, -compound) * 0.15)
        else:
            fear_raw = (fear_unneg * 0.55) + (max(0.0, -compound) * 0.20)
        fear_score = float(np.clip(fear_raw, 0.0, 1.0))

        # Neutral / Inquiry Score
        active_emotion_mass = max(joy_score, anger_score, sadness_score, fear_score)
        if active_emotion_mass < 0.25:
            neutral_score = float(np.clip(neu * 0.85 + (1.0 - abs(compound)) * 0.40, 0.50, 1.0))
        else:
            neutral_score = float(np.clip((1.0 - active_emotion_mass) * 0.50, 0.0, 0.40))

        emotion_scores = {
            "Joy / Gratitude": round(joy_score, 4),
            "Anger / Frustration": round(anger_score, 4),
            "Disappointment / Sadness": round(sadness_score, 4),
            "Fear / Anxiety": round(fear_score, 4),
            "Neutral / Inquiry": round(neutral_score, 4)
        }

        # 4. Multi-Label Thresholding (Binary Vector)
        # Threshold: 0.30 for specific emotions, with dynamic relaxation if contrast present
        threshold = 0.28 if has_contrast else 0.32
        multi_labels = {}
        for cls in EMOTION_CLASSES:
            if cls == "Neutral / Inquiry":
                multi_labels[cls] = 1 if neutral_score >= 0.50 and active_emotion_mass < 0.30 else 0
            else:
                multi_labels[cls] = 1 if emotion_scores[cls] >= threshold else 0

        # Ensure at least one label is active
        if sum(multi_labels.values()) == 0:
            best_cls = max(emotion_scores.items(), key=lambda x: x[1])[0]
            multi_labels[best_cls] = 1

        label_vector = [multi_labels[cls] for cls in EMOTION_CLASSES]

        # 5. Determine Primary & Secondary Emotions
        sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
        primary_emotion = sorted_emotions[0][0]
        
        # Secondary emotion qualifies if its score is substantial (>= 0.25 and >= 0.45 of primary)
        second_name, second_score = sorted_emotions[1]
        if second_score >= 0.25 and second_score >= (sorted_emotions[0][1] * 0.40) and second_name != primary_emotion:
            secondary_emotion = second_name
            is_mixed = True
        else:
            secondary_emotion = "None"
            is_mixed = False

        return {
            "sentiment": sentiment,
            "compound": compound,
            "pos": pos,
            "neg": neg,
            "neu": neu,
            "emotion": primary_emotion,
            "primary_emotion": primary_emotion,
            "secondary_emotion": secondary_emotion,
            "is_mixed": is_mixed,
            "emotion_scores": emotion_scores,
            "multi_labels": multi_labels,
            "label_vector": label_vector
        }

    def label_dataframe(self, df: pd.DataFrame, text_column: str = "clean_text") -> pd.DataFrame:
        """
        Applies negation-aware multi-label emotion tagging to a pandas DataFrame.
        """
        results = [self.analyze_tweet(text) for text in df[text_column]]
        
        df_out = df.copy()
        df_out['sentiment'] = [r['sentiment'] for r in results]
        df_out['emotion'] = [r['primary_emotion'] for r in results]
        df_out['primary_emotion'] = [r['primary_emotion'] for r in results]
        df_out['secondary_emotion'] = [r['secondary_emotion'] for r in results]
        df_out['is_mixed'] = [r['is_mixed'] for r in results]
        df_out['vader_compound'] = [r['compound'] for r in results]
        df_out['vader_pos'] = [r['pos'] for r in results]
        df_out['vader_neg'] = [r['neg'] for r in results]
        df_out['vader_neu'] = [r['neu'] for r in results]
        
        # Multi-label binary indicator columns
        for i, cls in enumerate(EMOTION_CLASSES):
            col_name = "is_" + cls.split("/")[0].strip().lower().replace(" ", "_")
            df_out[col_name] = [r['label_vector'][i] for r in results]
            
        df_out['multi_labels'] = [
            [cls for cls, val in r['multi_labels'].items() if val == 1]
            for r in results
        ]
        
        return df_out


if __name__ == '__main__':
    labeler = EmotionLabeler()
    test_cases = [
        "I was not disappointed with the outcome, in fact I am very happy!",
        "I am not angry at all, just asking for a status update.",
        "I found myself unexpectedly proud of the milestone, although I remained anxious about the release.",
        "I'm excited but nervous about tomorrow's flight.",
        "Worst customer service ever! Nobody answers and I am furious!",
        "Flight IB7449 was cancelled suddenly. I lost my connection, so sad.",
        "Great, another delay. Exactly what I needed today."
    ]
    
    print("=" * 75)
    print(" NEGATION-AWARE MULTI-LABEL EMOTION LABELER DEMO ")
    print("=" * 75)
    for txt in test_cases:
        res = labeler.analyze_tweet(txt)
        print(f"Input: \"{txt}\"")
        print(f"  -> Primary  : {res['primary_emotion']} ({res['emotion_scores'][res['primary_emotion']]*100:.1f}%)")
        print(f"  -> Secondary: {res['secondary_emotion']}")
        print(f"  -> Multi-Labels: {res['multi_labels']}")
        print(f"  -> Scores   : {res['emotion_scores']}\n")
