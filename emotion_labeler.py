import re
import pandas as pd
from typing import Dict, Any, Tuple
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class EmotionLabeler:
    """
    Rule and lexicon-enhanced sentiment and emotion labeling engine
    tailored for customer support interactions and social tweets.
    """
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        
        # Keyword & pattern lexicons for emotions
        self.anger_keywords = re.compile(
            r'\b(angry|furious|terrible|worst|horrible|frustrated|frustrating|unacceptable|garbage|'
            r'scam|ridiculous|useless|hate|broken|fail|fails|failed|sucks|annoyed|annoying|never again|'
            r'ridiculous|disgusting|pathetic|scammed|appalling|trash|shameful|infuriating)\b',
            re.IGNORECASE
        )
        
        self.sadness_keywords = re.compile(
            r'\b(sad|disappointed|disappointing|disappointment|bummer|upset|regret|missed|missing|'
            r'lost|ruined|refund|cancel|cancelled|cancellation|heartbroken|pity|unfortunate|'
            r'unhappy|loss|losing|hopeless|depressing)\b',
            re.IGNORECASE
        )
        
        self.fear_keywords = re.compile(
            r'\b(worried|worry|scared|afraid|anxious|anxiety|hacked|security|breach|compromised|'
            r'stolen|danger|urgent|panic|leak|fraud|warning|threat|locked out|identity theft|'
            r'suspicious|unsafe|vulnerability)\b',
            re.IGNORECASE
        )
        
        self.joy_keywords = re.compile(
            r'\b(thank|thanks|thankyou|awesome|great|amazing|excellent|wonderful|love|loved|'
            r'perfect|happy|kudos|solved|helpful|best|bless|blessed|appreciated|appreciate|'
            r'fantastic|brilliant|rockstar|applause)\b',
            re.IGNORECASE
        )

    def analyze_tweet(self, text: str) -> Dict[str, Any]:
        """
        Analyzes a single tweet text and returns VADER scores, overall sentiment,
        and primary emotion label.
        """
        if not isinstance(text, str) or not text.strip():
            return {
                "sentiment": "Neutral",
                "emotion": "Neutral / Inquiry",
                "compound": 0.0,
                "pos": 0.0,
                "neg": 0.0,
                "neu": 1.0
            }

        # Calculate VADER sentiment
        vader_scores = self.vader.polarity_scores(text)
        compound = vader_scores['compound']
        neg = vader_scores['neg']
        pos = vader_scores['pos']

        # Determine Valence Sentiment
        if compound >= 0.05:
            sentiment = "Positive"
        elif compound <= -0.05:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        # Count lexicon matches
        anger_matches = len(self.anger_keywords.findall(text))
        sadness_matches = len(self.sadness_keywords.findall(text))
        fear_matches = len(self.fear_keywords.findall(text))
        joy_matches = len(self.joy_keywords.findall(text))

        # Emotion assignment logic
        # 1. Joy / Gratitude
        if joy_matches > 0 and pos > neg and compound > 0.1:
            emotion = "Joy / Gratitude"
        # 2. Fear / Anxiety
        elif fear_matches > 0:
            emotion = "Fear / Anxiety"
        # 3. Anger / Frustration vs Disappointment / Sadness
        elif anger_matches > 0 or (neg > 0.25 and compound <= -0.5 and sadness_matches == 0):
            emotion = "Anger / Frustration"
        elif sadness_matches > 0 or (neg > 0.2 and compound < -0.2):
            # Distinguish strong negative without anger keywords as Disappointment
            if anger_matches > sadness_matches:
                emotion = "Anger / Frustration"
            else:
                emotion = "Disappointment / Sadness"
        # 4. Fallback for negative sentiment without specific keywords
        elif sentiment == "Negative":
            if neg > 0.3:
                emotion = "Anger / Frustration"
            else:
                emotion = "Disappointment / Sadness"
        # 5. Fallback for positive sentiment without joy keywords
        elif sentiment == "Positive" and pos > 0.2:
            emotion = "Joy / Gratitude"
        # 6. Default Neutral / Inquiry
        else:
            emotion = "Neutral / Inquiry"

        return {
            "sentiment": sentiment,
            "emotion": emotion,
            "compound": compound,
            "pos": pos,
            "neg": neg,
            "neu": vader_scores['neu']
        }

    def label_dataframe(self, df, text_column: str = "clean_text"):
        """
        Applies emotion and sentiment labeling to a pandas DataFrame.
        """
        results = [self.analyze_tweet(text) for text in df[text_column]]
        res_df = pd.DataFrame(results)
        
        df_out = df.copy()
        df_out['sentiment'] = res_df['sentiment']
        df_out['emotion'] = res_df['emotion']
        df_out['vader_compound'] = res_df['compound']
        df_out['vader_pos'] = res_df['pos']
        df_out['vader_neg'] = res_df['neg']
        df_out['vader_neu'] = res_df['neu']
        
        return df_out

if __name__ == '__main__':
    import pandas as pd
    labeler = EmotionLabeler()
    sample_texts = [
        "My PlayStation device gives error (ws-37338-4). What is the reason for this warning? please help me",
        "I have tried. For years now poor service. I hate this company!",
        "Hi the flight IB7449 is cancelled? I am losing all my connections to BA! Sad news.",
        "Is my personal data compromised after this security leak? Worried about my password!",
        "Thank you so much for fixing my internet connection so quickly! Amazing support team!"
    ]
    for txt in sample_texts:
        res = labeler.analyze_tweet(txt)
        print(f"Text: {txt}\n  --> Emotion: {res['emotion']} | Sentiment: {res['sentiment']} (Compound: {res['compound']:.3f})\n")
