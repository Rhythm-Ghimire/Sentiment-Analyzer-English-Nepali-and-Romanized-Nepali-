"""Sentiment model: TF-IDF (word + character n-grams) + sentiment lexicon + Logistic Regression.

Character n-grams are the key trick here: they work on any script, so the same
model handles Devanagari, Romanized Nepali and English without separate tokenizers,
and they cope with spelling variations like "ramro"/"raamro" or "cha"/"chha".
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from src.lexicon import LexiconFeatures
from src.preprocess import clean_text, detect_language

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "sentiment_model.joblib"
TRAIN_PATH = ROOT / "data" / "train.csv"


def build_pipeline() -> Pipeline:
    features = FeatureUnion([
        ("words", TfidfVectorizer(preprocessor=clean_text, ngram_range=(1, 2),
                                  token_pattern=r"(?u)[\w\u0900-\u097F]+", sublinear_tf=True)),
        ("chars", TfidfVectorizer(preprocessor=clean_text, analyzer="char_wb",
                                  ngram_range=(2, 5), sublinear_tf=True)),
        ("lexicon", LexiconFeatures(weight=1.0)),
    ])
    clf = LogisticRegression(max_iter=2000, C=5.0, class_weight="balanced")
    return Pipeline([("features", features), ("clf", clf)])


def train(train_path: Path = TRAIN_PATH) -> Pipeline:
    df = pd.read_csv(train_path).dropna(subset=["text", "label"])
    model = build_pipeline()
    model.fit(df["text"], df["label"])
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return model


def load_model() -> Pipeline:
    """Load the saved model, training it first if it doesn't exist yet."""
    if not MODEL_PATH.exists():
        return train()
    return joblib.load(MODEL_PATH)


def predict(model: Pipeline, texts) -> pd.DataFrame:
    """Return a DataFrame with text, language, sentiment and confidence."""
    texts = pd.Series(list(texts), dtype=str)
    probs = model.predict_proba(texts)
    classes = model.classes_
    best = probs.argmax(axis=1)
    return pd.DataFrame({
        "text": texts,
        "language": texts.map(detect_language),
        "sentiment": classes[best],
        "confidence": probs.max(axis=1).round(3),
    })
