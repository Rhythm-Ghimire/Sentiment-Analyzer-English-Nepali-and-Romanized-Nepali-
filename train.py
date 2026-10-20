"""Train the model, print cross-validated metrics, and save it to models/."""
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.model import MODEL_PATH, TRAIN_PATH, build_pipeline, train
from src.preprocess import detect_language

df = pd.read_csv(TRAIN_PATH)
print(f"Training examples: {len(df)}")
print(df["label"].value_counts().to_string(), "\n")

# 5-fold cross-validation gives an honest estimate on such a small dataset
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
preds = cross_val_predict(build_pipeline(), df["text"], df["label"], cv=cv)
print("Cross-validated results:")
print(classification_report(df["label"], preds, digits=3))

df["lang"] = df["text"].map(detect_language)
df["correct"] = preds == df["label"]
print("Accuracy by language:")
print(df.groupby("lang")["correct"].mean().round(3).to_string(), "\n")

train()
print(f"Model saved to {MODEL_PATH}")
