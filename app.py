import re
from collections import Counter

import pandas as pd
import streamlit as st

# ----------------------------- Config & Lexicons -----------------------------

MAX_REVIEWS = 500
MAX_LINE_LENGTH = 2000  # longer lines are truncated to keep processing fast
CSV_DANGEROUS_PREFIXES = ("=", "+", "-", "@", "\t", "\r")  # Excel treats these as formulas

POSITIVE = {
    "great", "good", "awesome", "love", "loved", "fast", "excellent", "amazing",
    "best", "perfect", "fantastic", "wonderful", "happy", "reliable", "smooth",
    "recommend", "easy", "helpful", "friendly", "quality", "worth", "nice",
    "impressive", "satisfied", "durable", "cheap", "affordable", "beautiful",
}
NEGATIVE = {
    "terrible", "bad", "poor", "slow", "late", "worst", "hate", "hated", "broken",
    "awful", "horrible", "disappointed", "disappointing", "useless", "waste",
    "defective", "rude", "expensive", "overpriced", "laggy", "buggy", "cheaply",
    "refund", "damaged", "scam", "annoying", "unreliable", "faulty", "missing",
}
NEGATIONS = {
    "not", "no", "never", "hardly", "without", "dont", "don't", "didn't",
    "doesn't", "isn't", "wasn't", "aren't", "weren't", "won't", "can't",
    "couldn't", "wouldn't", "cannot", "barely",
}
BOOSTERS = {"very", "really", "extremely", "super", "absolutely", "incredibly", "totally"}

TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
REVIEW_COLUMNS = {"review", "reviews", "text", "comment", "comments", "content"}


# ------------------------------- Analysis logic ------------------------------

def analyze_sentiment(review: str) -> dict:
    """Score a review with negation + intensifier handling."""
    tokens = TOKEN_RE.findall(review.lower())
    score = 0.0
    pos_hits, neg_hits = [], []

    for i, tok in enumerate(tokens):
        if tok in POSITIVE:
            polarity = 1
        elif tok in NEGATIVE:
            polarity = -1
        else:
            continue

        window = tokens[max(0, i - 3):i]
        weight = 1.5 if any(w in BOOSTERS for w in window) else 1.0
        negated = any(w in NEGATIONS for w in window)
        if negated:
            polarity = -polarity

        score += polarity * weight
        label = f"not {tok}" if negated else tok
        (pos_hits if polarity > 0 else neg_hits).append(label)

    sentiment = "Positive" if score > 0 else "Negative" if score < 0 else "Neutral"
    return {
        "Review": review,
        "Sentiment": sentiment,
        "Score": round(score, 2),
        "Positive Words": ", ".join(pos_hits),
        "Negative Words": ", ".join(neg_hits),
    }


@st.cache_data(show_spinner=False)
def analyze_batch(reviews: tuple) -> pd.DataFrame:
    return pd.DataFrame([analyze_sentiment(r) for r in reviews])


def top_keywords(df: pd.DataFrame, column: str, n: int = 8) -> pd.DataFrame:
    counter = Counter(
        w.strip() for cell in df[column] for w in cell.split(",") if w.strip()
    )
    return pd.DataFrame(counter.most_common(n), columns=["Keyword", "Count"])


def load_uploaded(file) -> list:
    if file.name.lower().endswith(".csv"):
        df = pd.read_csv(file)
        col = next((c for c in df.columns if str(c).lower() in REVIEW_COLUMNS), df.columns[0])
        return df[col].dropna().astype(str).tolist()
    return file.read().decode("utf-8", errors="ignore").splitlines()


def clean_lines(lines) -> tuple:
    """Strip, truncate overly long lines, drop blanks and duplicates.

    Returns (cleaned_lines, number_of_truncated_lines).
    """
    seen, cleaned, truncated = set(), [], 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if len(line) > MAX_LINE_LENGTH:
            line = line[:MAX_LINE_LENGTH]
            truncated += 1
        key = line.lower()
        if key not in seen:
            seen.add(key)
            cleaned.append(line)
    return cleaned, truncated


def _sanitize_csv_cell(value):
    """Prefix risky text cells with ' so Excel won't execute them as formulas (CSV injection)."""
    if isinstance(value, str) and value.startswith(CSV_DANGEROUS_PREFIXES):
        return "'" + value
    return value


def to_safe_csv(df: pd.DataFrame) -> bytes:
    """Export a DataFrame to CSV bytes with every text cell sanitized.

    Only text cells are touched, so numeric columns (e.g. negative scores) stay numeric.
    """
    safe = df.copy()
    for col in safe.select_dtypes(include="object").columns:
        safe[col] = safe[col].map(_sanitize_csv_cell)
    return safe.to_csv(index=False).encode("utf-8-sig")


# ----------------------------------- UI --------------------------------------

st.set_page_config(page_title="Competitor Review Analyzer", page_icon="📊", layout="wide")
st.title("📊 Competitor Review Analyzer")
st.caption("Paste reviews or upload a file, then analyze sentiment, keywords, and export results.")

left, right = st.columns([3, 2])
with left:
    reviews_text = st.text_area(
        "Customer Reviews (one review per line)",
        height=220,
        placeholder="Great battery life, fast delivery...\nTerrible customer support and late shipping...",
    )
with right:
    uploaded = st.file_uploader("...or upload a .txt / .csv file", type=["txt", "csv"])
    st.caption(f"Max {MAX_REVIEWS} unique reviews per batch. CSV: uses a 'review' / 'text' column, or the first column.")

if st.button("Analyze Reviews", type="primary"):
    raw_lines = reviews_text.splitlines()
    if uploaded is not None:
        try:
            raw_lines += load_uploaded(uploaded)
        except Exception as exc:
            st.error(f"Could not read the file: {exc}")
            raw_lines = []

    lines, truncated = clean_lines(raw_lines)
    if truncated:
        st.warning(f"{truncated} review(s) were longer than {MAX_LINE_LENGTH} characters and were truncated.")

    if not lines:
        st.warning("Please enter or upload at least one review.")
    elif len(lines) > MAX_REVIEWS:
        st.error(f"Too many reviews ({len(lines)}). Limit is {MAX_REVIEWS} per batch.")
    else:
        st.session_state["df"] = analyze_batch(tuple(lines))  # persists across reruns

# Results live OUTSIDE the button block, so changing the filter doesn't wipe them.
df = st.session_state.get("df")
if df is not None and not df.empty:
    total = len(df)
    counts = df["Sentiment"].value_counts().reindex(["Positive", "Negative", "Neutral"], fill_value=0)

    st.success(f"Analyzed {total} unique review(s).")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Positive", int(counts["Positive"]), f"{counts['Positive'] / total:.0%}")
    m2.metric("Negative", int(counts["Negative"]), f"{counts['Negative'] / total:.0%}", delta_color="inverse")
    m3.metric("Neutral", int(counts["Neutral"]), f"{counts['Neutral'] / total:.0%}", delta_color="off")
    m4.metric("Avg. Score", f"{df['Score'].mean():.2f}")

    chart_col, kw_col = st.columns([2, 3])
    with chart_col:
        st.subheader("Distribution")
        st.bar_chart(counts)
    with kw_col:
        st.subheader("Top Keywords")
        k1, k2 = st.columns(2)
        k1.caption("👍 Praised for")
        k1.dataframe(top_keywords(df, "Positive Words"), hide_index=True, use_container_width=True)
        k2.caption("👎 Criticized for")
        k2.dataframe(top_keywords(df, "Negative Words"), hide_index=True, use_container_width=True)

    st.subheader("Reviews")
    selected = st.radio("Filter by Sentiment", ["All", "Positive", "Negative", "Neutral"], horizontal=True)
    filtered = df if selected == "All" else df[df["Sentiment"] == selected]
    st.dataframe(filtered, use_container_width=True, hide_index=True)

    st.download_button(
        label=f"Download {selected} results as CSV",
        data=to_safe_csv(filtered),  # sanitized against formula injection, utf-8-sig for Excel
        file_name=f"sentiment_{selected.lower()}.csv",
        mime="text/csv",
    )