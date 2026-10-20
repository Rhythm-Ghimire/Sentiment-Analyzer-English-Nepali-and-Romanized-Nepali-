"""Text cleaning and simple language detection for Nepali / Romanized Nepali / English."""
import re

# Common Romanized-Nepali words. If a non-Devanagari text contains any of these,
# we treat it as Romanized Nepali (often code-mixed with English).
ROMAN_NEPALI_WORDS = {
    "cha", "chha", "chu", "chau", "thiyo", "thiye", "bhayo", "bhaye", "huncha", "hunchha",
    "ramro", "naramro", "dherai", "ekdam", "mitho", "khana", "maile", "yo", "ma", "ko",
    "ho", "ra", "ni", "pani", "feri", "garchu", "garna", "gare", "lagyo", "lagcha",
    "aaja", "bholi", "hijo", "dami", "bekar", "sasto", "mahango", "chito", "dhilo",
    "kina", "kasto", "sabai", "lai", "bata", "sanga", "haru", "khusi", "aauchu", "kya",
}

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")


def clean_text(text: str) -> str:
    """Lowercase, strip URLs/mentions, keep the word part of hashtags."""
    text = str(text)
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = text.replace("#", " ")
    text = text.lower()
    return re.sub(r"\s+", " ", text).strip()


def detect_language(text: str) -> str:
    """Return 'Nepali', 'Romanized Nepali' or 'English'."""
    text = str(text)
    letters = [c for c in text if c.isalpha()]
    if letters:
        devanagari_share = sum(bool(DEVANAGARI_RE.match(c)) for c in letters) / len(letters)
        if devanagari_share > 0.3:
            return "Nepali"
    words = set(re.findall(r"[a-z]+", text.lower()))
    if words & ROMAN_NEPALI_WORDS:
        return "Romanized Nepali"
    return "English"
