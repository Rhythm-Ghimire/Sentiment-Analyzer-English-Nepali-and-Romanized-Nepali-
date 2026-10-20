# Nepali + English Sentiment Analysis

A small AI project that classifies reviews and tweets as **positive**, **neutral** or **negative**. It handles three kinds of text, including code-mixed ones:

| Language | Example |
|---|---|
| Nepali (Devanagari) | खाना मिठो थियो, सेवा पनि राम्रो |
| Romanized Nepali | delivery dhilo bhayo, khana chiso |
| English | Customer support was rude and unhelpful |

It comes with a Streamlit dashboard for exploring results.

## Quick start

```bash
pip install -r requirements.txt
python train.py          # trains the model and prints evaluation metrics
streamlit run app.py     # opens the dashboard in your browser
```

Command-line predictions:

```bash
python predict.py "yo restaurant ko khana dherai mitho cha"
python predict.py        # interactive mode
```

## Project structure

```
nepali-sentiment/
├── app.py               Streamlit dashboard
├── train.py             Train + cross-validate + save the model
├── predict.py           Command-line predictions
├── src/
│   ├── preprocess.py    Text cleaning and language detection
│   ├── lexicon.py       Bilingual sentiment word lists (feature extractor)
│   └── model.py         Model pipeline, training, prediction
├── data/
│   ├── train.csv        135 labeled examples (45 per class, 3 languages)
│   └── sample_tweets.csv  28 unlabeled dated tweets for the dashboard demo
└── models/              Saved model (created by train.py)
```

## How it works

1. **Cleaning**: lowercases text and removes URLs, @mentions and the `#` from hashtags.
2. **Language detection**: if more than 30% of letters are Devanagari it's Nepali; otherwise, if it contains common Romanized Nepali words (*cha, thiyo, ramro, dherai…*), it's Romanized Nepali; otherwise English.
3. **Features** (combined with scikit-learn's `FeatureUnion`):
   - **Word TF-IDF** (1–2 word n-grams).
   - **Character TF-IDF** (2–5 character n-grams). This is what makes one model work across scripts, and it tolerates spelling variation such as *ramro / raamro* or *cha / chha*.
   - **Lexicon features**: counts of positive and negative words from a hand-built English + Nepali + Romanized Nepali word list. Devanagari words are matched by prefix because Nepali attaches suffixes (राम्रोसँग, सेवाले).
4. **Classifier**: Logistic Regression, which outputs a probability per class that's shown as the confidence.

## Results

5-fold cross-validation on `train.csv`:

| Setup | Accuracy |
|---|---|
| TF-IDF only | ~57% |
| TF-IDF + lexicon | ~98% |

**Be careful interpreting the 98%.** The dataset is tiny and the lexicon was written with these same kinds of sentences in mind, so the real-world accuracy will be lower, especially on sarcasm, negation ("not bad"), slang and topics outside reviews. The jump does show why a lexicon matters so much for a low-resource language with little training data.

## Dashboard

- A text box to test any single sentence
- Sample tweets or your own CSV upload (pick the text column; a `date` column enables the trend chart)
- Summary percentages, a sentiment donut, sentiment by language, sentiment over time
- Most common words per sentiment
- A filterable results table with confidence bars and a CSV download

## Ideas to extend it

- **More data**: add your own labeled rows to `data/train.csv` and rerun `python train.py`. A few hundred examples per class will help a lot.
- **Grow the lexicon** in `src/lexicon.py` with domain words (e.g. for banking, telecom, politics).
- **Handle negation**: flip the sentiment of a lexicon word that follows *not / never / छैन / chaina*.
- **Use a transformer**: fine-tune a multilingual model such as XLM-RoBERTa or MuRIL with Hugging Face `transformers`. These models already know some Nepali and usually beat TF-IDF once you have ~1,000+ labeled examples.
- **Real data**: collect tweets via the X API, or reviews from Google Maps, Daraz or app stores, then label a sample.
