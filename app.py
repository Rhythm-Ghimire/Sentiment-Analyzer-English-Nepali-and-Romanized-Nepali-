"""Sentiment dashboard for Nepali / Romanized Nepali / English reviews and tweets.

Run:  streamlit run app.py
"""
import re
from collections import Counter

import pandas as pd
import plotly.express as px
import streamlit as st

from src.model import ROOT, load_model, predict

COLORS = {"positive": "#2E8B57", "neutral": "#8A94A6", "negative": "#DC143C"}
ORDER = ["positive", "neutral", "negative"]
STOPWORDS = {
    "the", "a", "an", "and", "is", "was", "in", "at", "to", "of", "for", "with", "my", "this",
    "it", "i", "on", "me", "very", "so", "be", "are", "will", "again", "new", "from",
    "yo", "ma", "ko", "ho", "ra", "cha", "chha", "thiyo", "bhayo", "ni", "pani", "le", "lai",
    "र", "छ", "हो", "थियो", "भयो", "यो", "म", "पनि", "मा", "को", "धेरै", "dherai", "ekdam", "एकदम",
}

st.set_page_config(page_title="Nepali + English sentiment", page_icon="🇳🇵", layout="wide")


@st.cache_resource
def get_model():
    return load_model()


@st.cache_data
def analyze(texts: tuple) -> pd.DataFrame:
    return predict(get_model(), texts)


def top_words(texts, n=8):
    counts = Counter()
    for t in texts:
        for w in re.findall(r"[\w\u0900-\u097F]+", str(t).lower()):
            if w not in STOPWORDS and not w.isdigit() and len(w) > 1:
                counts[w] += 1
    return counts.most_common(n)


# ---------- Header ----------
st.title("How do people feel?")
st.caption("Sentiment analysis for reviews and tweets in Nepali (देवनागरी), Romanized Nepali and English.")

# ---------- Single-text tester ----------
with st.container(border=True):
    text = st.text_input("Try a sentence", placeholder="e.g. khana ekdam mitho thiyo / सेवा एकदम बेकार")
    if text.strip():
        r = analyze((text,)).iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Sentiment", r.sentiment.capitalize())
        c2.metric("Confidence", f"{r.confidence:.0%}")
        c3.metric("Language", r.language)

# ---------- Data source ----------
st.sidebar.header("Data")
source = st.sidebar.radio("Analyze", ["Sample tweets", "Upload a CSV"])
if source == "Upload a CSV":
    file = st.sidebar.file_uploader("CSV with a text column (optional date column)", type="csv")
    if file is None:
        st.info("Upload a CSV in the sidebar to see its dashboard, or switch to the sample tweets.")
        st.stop()
    raw = pd.read_csv(file)
else:
    raw = pd.read_csv(ROOT / "data" / "sample_tweets.csv")

text_col = st.sidebar.selectbox(
    "Text column", raw.columns, index=list(raw.columns).index("text") if "text" in raw.columns else 0
)
raw = raw.dropna(subset=[text_col]).reset_index(drop=True)
results = analyze(tuple(raw[text_col].astype(str)))
for col in raw.columns:
    if col != text_col:
        results[col] = raw[col].values

languages = st.sidebar.multiselect("Languages", sorted(results["language"].unique()),
                                   default=sorted(results["language"].unique()))
df = results[results["language"].isin(languages)]
if df.empty:
    st.warning("No rows match the selected languages. Pick at least one in the sidebar.")
    st.stop()

# ---------- Summary numbers ----------
share = df["sentiment"].value_counts(normalize=True)
m = st.columns(4)
m[0].metric("Texts analyzed", len(df))
for col, label in zip(m[1:], ORDER):
    col.metric(label.capitalize(), f"{share.get(label, 0):.0%}")

# ---------- Charts ----------
left, right = st.columns(2)
with left:
    st.subheader("Overall sentiment")
    counts = df["sentiment"].value_counts().reindex(ORDER, fill_value=0).reset_index()
    counts.columns = ["sentiment", "count"]
    fig = px.pie(counts, names="sentiment", values="count", hole=0.55,
                 color="sentiment", color_discrete_map=COLORS)
    fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), legend_title_text="")
    st.plotly_chart(fig, width="stretch")

with right:
    st.subheader("By language")
    by_lang = df.groupby(["language", "sentiment"]).size().reset_index(name="count")
    fig = px.bar(by_lang, x="language", y="count", color="sentiment",
                 color_discrete_map=COLORS, category_orders={"sentiment": ORDER})
    fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), xaxis_title="", legend_title_text="")
    st.plotly_chart(fig, width="stretch")

if "date" in df.columns:
    st.subheader("Over time")
    trend = df.assign(date=pd.to_datetime(df["date"], errors="coerce")).dropna(subset=["date"])
    trend = trend.groupby([trend["date"].dt.date, "sentiment"]).size().reset_index(name="count")
    fig = px.line(trend, x="date", y="count", color="sentiment", markers=True,
                  color_discrete_map=COLORS, category_orders={"sentiment": ORDER})
    fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), xaxis_title="", legend_title_text="")
    st.plotly_chart(fig, width="stretch")

st.subheader("Most common words")
word_cols = st.columns(3)
for col, label in zip(word_cols, ORDER):
    with col:
        st.markdown(f"**{label.capitalize()}**")
        words = top_words(df.loc[df["sentiment"] == label, "text"])
        if words:
            st.dataframe(pd.DataFrame(words, columns=["word", "count"]),
                         hide_index=True, width="stretch")
        else:
            st.caption("No texts in this group.")

# ---------- Table ----------
st.subheader("All results")
pick = st.multiselect("Show sentiment", ORDER, default=ORDER)
table = df[df["sentiment"].isin(pick)]
st.dataframe(table, hide_index=True, width="stretch",
             column_config={"confidence": st.column_config.ProgressColumn(
                 "confidence", min_value=0, max_value=1, format="%.2f")})
st.download_button("Download results as CSV", table.to_csv(index=False).encode("utf-8-sig"),
                   "sentiment_results.csv", "text/csv")
