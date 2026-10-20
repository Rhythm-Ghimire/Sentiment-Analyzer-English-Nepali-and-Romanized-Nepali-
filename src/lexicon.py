"""A small hand-built sentiment lexicon (English + Nepali + Romanized Nepali).

Used as extra features next to TF-IDF so the model has prior knowledge of
sentiment words even when the training set is tiny. Extend these lists to
improve accuracy for your own domain.
"""
import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

POSITIVE = {
    # English
    "good", "great", "amazing", "excellent", "love", "loved", "loving", "best", "nice", "happy",
    "wonderful", "fantastic", "friendly", "fresh", "fast", "quick", "clean", "beautiful", "helpful",
    "kind", "smooth", "satisfied", "recommend", "recommended", "brilliant", "comfortable", "polite",
    "delicious", "awesome", "perfect", "worth", "thank", "thanks", "enjoyed", "superb",
    # Romanized Nepali
    "ramro", "raamro", "mitho", "dami", "jhakkas", "khusi", "sajilo", "sasto", "santusta",
    "sahayogi", "maja", "dhanyabad", "chito", "sundar", "uttam", "babal",
    # Nepali (Devanagari)
    "राम्रो", "मिठो", "खुसी", "सफा", "सुन्दर", "मिलनसार", "उत्कृष्ट", "सिफारिस", "सजिलो", "सस्तो",
    "सहयोगी", "शानदार", "सन्तुष्ट", "धन्यवाद", "छिटो", "रमाइलो", "मज्जा",
}

NEGATIVE = {
    # English
    "bad", "worst", "terrible", "horrible", "poor", "rude", "dirty", "cold", "late", "slow",
    "broke", "broken", "damaged", "waste", "fake", "crash", "crashing", "crashes", "overpriced",
    "disappointed", "boring", "careless", "never", "tasteless", "unhelpful", "ugh", "awful",
    "hate", "expensive", "scam", "useless",
    # Romanized Nepali
    "naramro", "bekar", "dhilo", "chiso", "fohor", "nakkali", "mahango", "kamjor", "badtamij",
    "niras", "jhyau", "thagi", "bigriyo", "laparwah", "jhanjhat",
    # Nepali (Devanagari)
    "नराम्रो", "बेकार", "ढिला", "ढिलो", "चिसो", "फोहोर", "नक्कली", "महँगो", "कमजोर", "असभ्य",
    "निराश", "झन्झट", "ठगी", "बिग्रियो", "लापरबाह", "स्वादहीन", "गन्हाउने", "भीड",
}

TOKEN_RE = re.compile(r"[\w\u0900-\u097F]+")
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")


def _matches(token: str, words: set) -> bool:
    if token in words:
        return True
    # Nepali attaches suffixes (राम्रोसँग, सेवाले), so match Devanagari by prefix
    if DEVANAGARI_RE.match(token):
        return any(token.startswith(w) for w in words if DEVANAGARI_RE.match(w))
    return False


def lexicon_score(text: str) -> tuple[int, int]:
    pos = neg = 0
    for tok in TOKEN_RE.findall(str(text).lower()):
        if _matches(tok, NEGATIVE):  # check negative first: नराम्रो contains राम्रो
            neg += 1
        elif _matches(tok, POSITIVE):
            pos += 1
    return pos, neg


class LexiconFeatures(BaseEstimator, TransformerMixin):
    """Turns each text into [positive hits, negative hits, net score, has-any-hit]."""

    def __init__(self, weight: float = 1.0):
        self.weight = weight

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for text in X:
            pos, neg = lexicon_score(text)
            rows.append([pos, neg, pos - neg, float(pos + neg > 0)])
        return np.array(rows, dtype=float) * self.weight
