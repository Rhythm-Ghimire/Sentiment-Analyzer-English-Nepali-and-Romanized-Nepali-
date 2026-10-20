"""Quick command-line prediction.

    python predict.py "khana ekdam mitho thiyo"
    python predict.py            # interactive mode
"""
import sys

from src.model import load_model, predict

model = load_model()


def show(text):
    row = predict(model, [text]).iloc[0]
    print(f"[{row.language}] {row.sentiment.upper()} ({row.confidence:.0%})  <- {text}")


if len(sys.argv) > 1:
    show(" ".join(sys.argv[1:]))
else:
    print("Type a review or tweet (empty line to quit).")
    while (line := input("> ").strip()):
        show(line)
