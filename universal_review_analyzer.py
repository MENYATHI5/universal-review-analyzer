"""
Universal Review Analyzer
A Streamlit dashboard for analyzing text reviews from any dataset.

Run:
    pip install streamlit pandas nltk matplotlib wordcloud plotly
    streamlit run universal_review_analyzer.py
"""
import io
import re
from datetime import datetime, timedelta

import nltk
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

# ---------- NLTK setup ----------
for pkg in ["vader_lexicon"]:
    try:
        nltk.data.find(f"sentiment/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ---------- Page config & dark theme ----------
st.set_page_config(
    page_title="Universal Review Analyzer",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .stApp { background: linear-gradient(180deg,#0b1220 0%, #0f172a 100%); color:#e2e8f0; }
    section[data-testid="stSidebar"] { background-color:#0a0f1c; border-right:1px solid #1e293b; }
    h1,h2,h3,h4 { color:#f1f5f9; font-family: 'Inter', sans-serif; }
    .accent { color:#14b8a6; }
    .accent-orange { color:#f97316; }
    .stat-card {
        background:#111827; border:1px solid #1f2937; border-radius:14px;
        padding:18px 20px; box-shadow:0 4px 20px rgba(20,184,166,0.05);
    }
    .stat-value { font-size:2rem; font-weight:700; color:#14b8a6; }
    .stat-label { font-size:0.85rem; color:#94a3b8; text-transform:uppercase; letter-spacing:1px; }
    .review-pos { border-left:4px solid #14b8a6; padding:12px 16px; background:#0f1f1d; border-radius:8px; margin-bottom:10px; }
    .review-neg { border-left:4px solid #f97316; padding:12px 16px; background:#1f1410; border-radius:8px; margin-bottom:10px; }
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        background:#111827; border-radius:10px 10px 0 0; padding: 8px 18px; color:#94a3b8;
    }
    .stTabs [aria-selected="true"] { background:#14b8a6 !important; color:#0b1220 !important; font-weight:600; }
    .stButton>button { background:#14b8a6; color:#0b1220; border:none; font-weight:600; border-radius:8px; }
    .stTextArea textarea, .stTextInput input { background:#111827 !important; color:#e2e8f0 !important; border:1px solid #1f2937 !important; }
    .explain { background:#0f1729; border-left:3px solid #14b8a6; padding:10px 14px; border-radius:6px; color:#cbd5e1; font-size:0.9rem; margin:8px 0 18px; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------- Helpers ----------
@st.cache_resource
def get_analyzer():
    return SentimentIntensityAnalyzer()

sia = get_analyzer()

def classify(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return "Neutral"
    s = sia.polarity_scores(text)["compound"]
    if s >= 0.05: return "Positive"
    if s <= -0.05: return "Negative"
    return "Neutral"

def compound(text: str) -> float:
    if not isinstance(text, str) or not text.strip():
        return 0.0
    return sia.polarity_scores(text)["compound"]

def smart_read_csv(uploaded) -> pd.DataFrame:
    raw = uploaded.read()
    for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"]:
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc, on_bad_lines="skip")
        except Exception:
            continue
    return pd.read_csv(io.BytesIO(raw), encoding="latin-1", errors="ignore", on_bad_lines="skip")

def detect_review_col(df: pd.DataFrame) -> str:
    candidates = ["review", "reviews", "text", "comment", "content", "feedback", "body"]
    for c in candidates:
        for col in df.columns:
            if c == col.lower().strip():
                return col
    # fallback: longest avg string column
    obj_cols = df.select_dtypes(include="object").columns
    if len(obj_cols) == 0: return df.columns[0]
    return max(obj_cols, key=lambda c: df[c].astype(str).str.len().mean())

def detect_rating_col(df):
    for col in df.columns:
        if "rating" in col.lower() or "stars" in col.lower() or "score" in col.lower():
            return col
    return None

def detect_time_col(df):
    for col in df.columns:
        if any(k in col.lower() for k in ["time", "date", "timestamp", "created"]):
            return col
    return None

def parse_rating(v):
    if pd.isna(v): return None
    if isinstance(v, (int, float)): return float(v)
    m = re.search(r"(\d+(\.\d+)?)", str(v))
    return float(m.group(1)) if m else None

def parse_relative_time(v):
    """Handle '3 months ago', '5 days ago', or real dates."""
    if pd.isna(v): return None
    s = str(v).strip().lower()
    now = datetime.now()
    m = re.match(r"(?:a|an|\d+)\s+(second|minute|hour|day|week|month|year)s?\s+ago", s)
    if m:
        num_match = re.match(r"(\d+)", s)
        n = int(num_match.group(1)) if num_match else 1
        unit = m.group(1)
        mult = {"second":1/86400,"minute":1/1440,"hour":1/24,"day":1,"week":7,"month":30,"year":365}[unit]
        return now - timedelta(days=n*mult)
    try:
        return pd.to_datetime(v, errors="coerce")
    except Exception:
        return None

# ---------- Sidebar ----------
st.sidebar.markdown("## 💬 Universal Review Analyzer")
st.sidebar.markdown("Upload any CSV of text reviews and explore sentiment, themes, and trends.")
uploaded = st.sidebar.file_uploader("Upload reviews CSV", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚡ Quick sentiment check")
quick = st.sidebar.text_area("Type a review:", placeholder="e.g. The food was amazing and staff was friendly!", height=100)
if quick.strip():
    label = classify(quick)
    score = compound(quick)
    color = {"Positive":"#14b8a6","Negative":"#f97316","Neutral":"#64748b"}[label]
    st.sidebar.markdown(
        f"<div style='padding:10px;border-radius:8px;background:{color}20;border:1px solid {color};'>"
        f"<b style='color:{color}'>{label}</b><br><small>VADER compound: {score:+.3f}</small></div>",
        unsafe_allow_html=True
    )

# ---------- Header ----------
st.markdown("<h1>💬 Universal Review <span class='accent'>Analyzer</span></h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#94a3b8;'>NLTK VADER sentiment intelligence for any review dataset.</p>", unsafe_allow_html=True)

if uploaded is None:
    st.info("👈 Upload a CSV in the sidebar to begin. Works with McDonald's reviews, Amazon, Yelp, app store reviews — any text column.")
    st.stop()

with st.spinner("Loading dataset..."):
    df = smart_read_csv(uploaded)

review_col = detect_review_col(df)
rating_col = detect_rating_col(df)
time_col = detect_time_col(df)

col_a, col_b, col_c = st.columns(3)
review_col = col_a.selectbox("Review text column", df.columns.tolist(), index=df.columns.tolist().index(review_col))
rating_col = col_b.selectbox("Rating column (optional)", ["(none)"] + df.columns.tolist(),
                             index=(df.columns.tolist().index(rating_col)+1) if rating_col else 0)
time_col = col_c.selectbox("Timestamp column (optional)", ["(none)"] + df.columns.tolist(),
                           index=(df.columns.tolist().index(time_col)+1) if time_col else 0)
rating_col = None if rating_col == "(none)" else rating_col
time_col = None if time_col == "(none)" else time_col

df = df.dropna(subset=[review_col]).copy()
df[review_col] = df[review_col].astype(str)

@st.cache_data(show_spinner=False)
def enrich(df, review_col, rating_col, time_col):
    out = df.copy()
    out["_compound"] = out[review_col].map(compound)
    out["_sentiment"] = out["_compound"].apply(lambda s: "Positive" if s>=0.05 else ("Negative" if s<=-0.05 else "Neutral"))
    if rating_col:
        out["_rating"] = out[rating_col].map(parse_rating)
    if time_col:
        out["_time"] = out[time_col].map(parse_relative_time)
    return out

with st.spinner("Scoring sentiment..."):
    df = enrich(df, review_col, rating_col, time_col)

tabs = st.tabs(["📊 Overview", "🧭 Sentiment", "☁️ Word Cloud", "📈 Trends", "🏆 Top Reviews"])

# ---------- Overview ----------
with tabs[0]:
    st.subheader("Dataset Overview")
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"<div class='stat-card'><div class='stat-label'>Total reviews</div><div class='stat-value'>{len(df):,}</div></div>", unsafe_allow_html=True)
    avg_rating = df["_rating"].mean() if rating_col else None
    c2.markdown(f"<div class='stat-card'><div class='stat-label'>Average rating</div><div class='stat-value'>{avg_rating:.2f}</div></div>" if avg_rating==avg_rating and avg_rating is not None else "<div class='stat-card'><div class='stat-label'>Average rating</div><div class='stat-value'>—</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='stat-card'><div class='stat-label'>Avg. sentiment</div><div class='stat-value'>{df['_compound'].mean():+.2f}</div></div>", unsafe_allow_html=True)
    st.markdown("<div class='explain'>📌 <b>What this means:</b> Total reviews shows your sample size — larger samples make the analysis more reliable. Average rating (if available) is the mean numeric star value. Average sentiment is the mean VADER compound score: <b>+1 very positive, -1 very negative, 0 neutral</b>.</div>", unsafe_allow_html=True)

    st.markdown("#### Sample reviews")
    st.dataframe(df[[review_col] + ([rating_col] if rating_col else []) + ["_sentiment"]].head(10), use_container_width=True)

# ---------- Sentiment ----------
with tabs[1]:
    st.subheader("Sentiment Distribution")
    counts = df["_sentiment"].value_counts().reindex(["Positive","Neutral","Negative"]).fillna(0).reset_index()
    counts.columns = ["Sentiment","Count"]
    fig = px.bar(counts, x="Sentiment", y="Count", color="Sentiment",
                 color_discrete_map={"Positive":"#14b8a6","Neutral":"#64748b","Negative":"#f97316"},
                 text="Count")
    fig.update_layout(plot_bgcolor="#0f172a", paper_bgcolor="#0f172a", font_color="#e2e8f0", showlegend=False, height=420)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        "<div class='explain'>📌 <b>How to read this:</b> Each bar shows how many reviews VADER classified into each bucket. "
        "A taller <span class='accent'>Positive</span> bar means customers are largely happy. A taller "
        "<span class='accent-orange'>Negative</span> bar signals widespread issues to investigate. Neutral reviews are factual or mixed.</div>",
        unsafe_allow_html=True
    )

# ---------- Word Cloud ----------
with tabs[2]:
    st.subheader("Word Cloud of Frequent Terms")
    text = " ".join(df[review_col].astype(str).tolist())
    text = re.sub(r"[^A-Za-z\s]", " ", text).lower()
    stop = set(STOPWORDS) | {"food","place","one","get","got","go","went","said","really","also","would","u","im"}
    if text.strip():
        wc = WordCloud(width=1200, height=500, background_color="#0f172a",
                       colormap="cool", stopwords=stop, collocations=False).generate(text)
        fig, ax = plt.subplots(figsize=(12,5), facecolor="#0f172a")
        ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
        st.pyplot(fig)
    st.markdown("<div class='explain'>📌 <b>What this shows:</b> Larger words appear more often across all reviews. Use it to spot recurring themes — menu items, service issues, locations, or emotions customers mention repeatedly.</div>", unsafe_allow_html=True)

# ---------- Trends ----------
with tabs[3]:
    st.subheader("Sentiment Trend Over Time")
    if not time_col or df["_time"].isna().all():
        st.warning("No usable timestamp column detected. Pick one above to enable trends.")
    else:
        ts = df.dropna(subset=["_time"]).copy()
        ts["_period"] = pd.to_datetime(ts["_time"]).dt.to_period("M").dt.to_timestamp()
        agg = ts.groupby("_period")["_compound"].mean().reset_index()
        fig = px.line(agg, x="_period", y="_compound", markers=True,
                      labels={"_period":"Month","_compound":"Avg sentiment"})
        fig.update_traces(line_color="#14b8a6")
        fig.add_hline(y=0, line_dash="dot", line_color="#64748b")
        fig.update_layout(plot_bgcolor="#0f172a", paper_bgcolor="#0f172a", font_color="#e2e8f0", height=420)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("<div class='explain'>📌 <b>How to read this:</b> The line is the average VADER sentiment per month. Trending up = customer perception is improving. Trending down = growing dissatisfaction. The dotted line at 0 is the neutral baseline.</div>", unsafe_allow_html=True)

# ---------- Top Reviews ----------
with tabs[4]:
    st.subheader("Most Positive & Most Negative")
    pos = df.nlargest(5, "_compound")
    neg = df.nsmallest(5, "_compound")
    cL, cR = st.columns(2)
    with cL:
        st.markdown("### <span class='accent'>🌟 Most Positive</span>", unsafe_allow_html=True)
        for _, r in pos.iterrows():
            st.markdown(f"<div class='review-pos'><small>VADER score: <b>{r['_compound']:+.3f}</b></small><br>{r[review_col][:400]}{'...' if len(r[review_col])>400 else ''}</div>", unsafe_allow_html=True)
    with cR:
        st.markdown("### <span class='accent-orange'>⚠️ Most Negative</span>", unsafe_allow_html=True)
        for _, r in neg.iterrows():
            st.markdown(f"<div class='review-neg'><small>VADER score: <b>{r['_compound']:+.3f}</b></small><br>{r[review_col][:400]}{'...' if len(r[review_col])>400 else ''}</div>", unsafe_allow_html=True)
    st.markdown("<div class='explain'>📌 <b>Why these were chosen:</b> VADER assigns each review a compound score from -1 to +1 based on the strength and combination of sentiment-bearing words (great, terrible, love, awful), emphasis (CAPS, !!!), and negations (not good). The highest scores rise to Most Positive; the lowest sink to Most Negative.</div>", unsafe_allow_html=True)
