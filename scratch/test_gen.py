import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score

random.seed(42)
np.random.seed(42)

subjects_neutral = ['flight booking', 'baggage policy', 'account profile', 'billing cycle', 'shipping tracking', 'password reset', 'invoice receipt', 'refund status', 'appointment schedule', 'store location', 'order update', 'seat selection', 'membership points', 'login credentials', 'cancellation fee']
subjects_joy = ['customer care executive', 'support specialist', 'technical team', 'chat agent', 'service representative', 'call center agent', 'flight crew', 'helpdesk staff']
adjectives_joy = ['amazing', 'fantastic', 'wonderful', 'super helpful', 'courteous', 'brilliant', 'prompt', 'delightful', 'outstanding', 'incredibly patient', 'so polite', 'first-class']
issues_anger = ['flight cancellation', 'wrong billing', 'broken item', 'unauthorized charge', 'account suspension', 'app crash', 'lost delivery', 'hidden fee', 'delayed refund', 'unresponsive chat']
adjectives_anger = ['terrible', 'worst ever', 'horrible', 'ridiculous', 'useless', 'unacceptable', 'pathetic', 'trash', 'disgraceful', 'disgusting', 'appalling', 'completely broken']
situations_sad = ['lost luggage', 'delayed delivery', 'ruined birthday gift', 'cancelled holiday trip', 'missing package', 'defective product', 'spoiled vacation', 'damaged goods']
feelings_sad = ['sad', 'disappointed', 'heartbroken', 'unhappy', 'let down', 'depressed', 'regretful', 'so upset', 'dismayed']
threats_fear = ['unauthorized transaction', 'security breach', 'hacked profile', 'identity theft', 'stolen card details', 'suspicious login', 'compromised password', 'fraudulent charge']
emotions_fear = ['panicking', 'scared', 'worried sick', 'afraid', 'terrified', 'anxious', 'alarmed', 'in extreme panic']

def generate_sentence(emotion):
    rand_num = random.randint(100, 999999)
    prefix = random.choice(['', 'Hello, ', 'Hi team, ', 'Customer support, ', 'Quick question: ', 'Attention: ', 'Please help: '])
    suffix = random.choice(['', f' Ticket #{rand_num}.', f' Order ID {rand_num}.', ' Please assist.', ' Awaiting reply.', ' Thanks.', ' Seriously?'])
    
    if emotion == 'Neutral / Inquiry':
        s = random.choice(subjects_neutral)
        templates = [
            f"Can you please check the status of my {s}?",
            f"What are the standard hours for handling {s} issues?",
            f"How do I update my {s} information on the mobile app?",
            f"Could you provide more details regarding your policy on {s}?",
            f"Where can I find the official documentation for {s}?",
            f"Is there any scheduled downtime affecting {s} this week?",
            f"Kindly clarify the next steps required for {s}.",
            f"I have a general inquiry about the terms and conditions for {s}.",
            f"Please let me know once the update on my {s} has been processed.",
            f"Could someone assist me with my {s} verification?"
        ]
    elif emotion == 'Joy / Gratitude':
        s = random.choice(subjects_joy)
        adj = random.choice(adjectives_joy)
        templates = [
            f"Thank you so much! The {s} was {adj} and solved my problem in minutes.",
            f"Huge shoutout to your {s}! They were {adj} and handled everything so smoothly. Loved it!",
            f"Really appreciate the {adj} assistance from customer service today. Exceptional experience!",
            f"Kudos to the team! Best support ever, the agent was {adj} and solved it quickly.",
            f"So grateful for the fast resolution. Your {s} provided {adj} service. Thank you!",
            f"Extremely satisfied with the prompt reply. The service was genuinely {adj}.",
            f"A big thank you to the {s} who assisted me today. Truly {adj} and wonderful!",
            f"Five stars! The representative went above and beyond, {adj} experience all around."
        ]
    elif emotion == 'Anger / Frustration':
        iss = random.choice(issues_anger)
        adj = random.choice(adjectives_anger)
        templates = [
            f"This is the {adj} experience ever! You completely ruined my {iss} and nobody responds!",
            f"I am furious! My {iss} has been completely ignored and customer support is {adj}. Fix this now!",
            f"Worst company ever! You caused a major {iss} and your staff is completely {adj}. Scam!",
            f"Your system is {adj} and nobody cares about resolving my {iss}! Absolutely disgraceful!",
            f"Unacceptable treatment! I have been waiting for hours regarding my {iss}. Beyond frustrated!",
            f"Horrible service. You messed up my {iss} and refuse to take any responsibility. Pathetic!",
            f"Still no response after days! Complete {adj} service regarding my {iss}. I demand a manager!",
            f"Ridiculous! Your agent was rude and hung up on me regarding my {iss}. Disgusting!"
        ]
    elif emotion == 'Disappointment / Sadness':
        sit = random.choice(situations_sad)
        flg = random.choice(feelings_sad)
        templates = [
            f"Really {flg} with the outcome. My {sit} was completely ruined and support could not help.",
            f"Feeling {flg} by the poor service. Been a loyal customer for years but this {sit} hurts.",
            f"So {flg} to hear my {sit} was lost. This was deeply important to our family.",
            f"Such a bummer. Was really looking forward to this, but the {sit} was an unfortunate failure.",
            f"Deeply {flg} that my request regarding {sit} was rejected without explanation. Regret this.",
            f"Sad to see how much quality has declined. My {sit} issue remains unresolved.",
            f"Heartbroken that my {sit} ruined our anniversary. Very disappointing service from your brand.",
            f"A very unfortunate experience with my {sit}. Really expected better care."
        ]
    else: # Fear / Anxiety
        thr = random.choice(threats_fear)
        emo = random.choice(emotions_fear)
        templates = [
            f"URGENT: I see an {thr} on my profile! I am {emo}, please lock my account immediately!",
            f"Help urgently! Received an alert about an {thr} and I am {emo}! Please contact me ASAP!",
            f"I am {emo} that someone compromised my details. Serious {thr} detected right now!",
            f"Terrified my personal information was stolen in the recent {thr}. My account is locked!",
            f"Please help, I am {emo}! Suspicious {thr} on my bank card linked to this account!",
            f"Emergency: An unauthorized {thr} was noticed! Please freeze everything immediately!",
            f"Extremely anxious about my security. A dangerous {thr} was reported, please verify!"
        ]
    
    return prefix + random.choice(templates) + suffix

# Generate 10,000 samples with realistic class balance
emotions = ['Neutral / Inquiry', 'Joy / Gratitude', 'Anger / Frustration', 'Disappointment / Sadness', 'Fear / Anxiety']
weights = [0.625, 0.194, 0.098, 0.071, 0.012]

sampled_emotions = np.random.choice(emotions, size=10000, p=weights)
texts = [generate_sentence(e) for e in sampled_emotions]

df_gen = pd.DataFrame({'clean_text': texts, 'emotion': sampled_emotions})
print(f"Generated {len(df_gen)} samples, {df_gen['clean_text'].nunique()} unique texts.")

# Test benchmark
X_tr, X_te, y_tr, y_te = train_test_split(df_gen['clean_text'], df_gen['emotion'], test_size=0.2, random_state=42, stratify=df_gen['emotion'])
vec = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
X_tr_vec = vec.fit_transform(X_tr)
X_te_vec = vec.transform(X_te)

models = [
    ('Multinomial Naive Bayes', MultinomialNB()),
    ('Logistic Regression', LogisticRegression(max_iter=500, class_weight='balanced')),
    ('Linear SVM', LinearSVC(class_weight='balanced', max_iter=1000)),
    ('Random Forest', RandomForestClassifier(n_estimators=50, max_depth=15, random_state=42)),
    ('LightGBM', LGBMClassifier(n_estimators=100, random_state=42, verbose=-1))
]

for name, clf in models:
    clf.fit(X_tr_vec, y_tr)
    preds = clf.predict(X_te_vec)
    acc = accuracy_score(y_te, preds) * 100
    f1 = f1_score(y_te, preds, average='macro')
    print(f"{name:32s} | Acc: {acc:5.2f}% | Macro F1: {f1:.4f}")
