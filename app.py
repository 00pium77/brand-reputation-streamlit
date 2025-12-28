# app.py
# Brand Reputation Monitor (2023) — Streamlit + Scraping + Hugging Face Sentiment
# Loads pre-scraped CSVs from ./data and runs sentiment analysis on demand (Render-friendly).

from __future__ import annotations

import calendar
import os
import subprocess
from datetime import date, datetime
from typing import Iterable, List, Tuple

import pandas as pd
import streamlit as st

# Optional but recommended deps (in your requirements.txt)
try:
    import altair as alt
except Exception:
    alt = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from wordcloud import WordCloud
except Exception:
    WordCloud = None

from transformers import pipeline


# ----------------------------
# Config
# ----------------------------
st.set_page_config(
    page_title="Brand Reputation Monitor (2023)",
    page_icon="📈",
    layout="wide",
)

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
PRODUCTS_CSV = os.path.join(DATA_DIR, "products.csv")
TESTIMONIALS_CSV = os.path.join(DATA_DIR, "testimonials.csv")
REVIEWS_CSV = os.path.join(DATA_DIR, "reviews.csv")

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"


# ----------------------------
# Helpers
# ----------------------------
def ensure_pandas(obj):
    """Convert common dataframe-like objects into pandas.DataFrame safely."""
    if obj is None:
        return pd.DataFrame()
    if isinstance(obj, pd.DataFrame):
        return obj

    # Polars
    if hasattr(obj, "to_pandas"):
        try:
            return obj.to_pandas()
        except Exception:
            pass

    # Streamlit sometimes hints about df.to_native()
    if hasattr(obj, "to_native"):
        try:
            native = obj.to_native()
            if isinstance(native, pd.DataFrame):
                return native
        except Exception:
            pass

    # Fallback: try constructor
    try:
        return pd.DataFrame(obj)
    except Exception:
        return pd.DataFrame()


def month_label(d: date) -> str:
    return f"{calendar.month_abbr[d.month]} {d.year}"


def month_start_end(year: int, month: int) -> Tuple[date, date]:
    start = date(year, month, 1)
    last_day = calendar.monthrange(year, month)[1]
    end = date(year, month, last_day)
    return start, end


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    return ensure_pandas(df)


@st.cache_data(show_spinner=False)
def load_all_data():
    products = load_csv(PRODUCTS_CSV)
    testimonials = load_csv(TESTIMONIALS_CSV)
    reviews = load_csv(REVIEWS_CSV)

    # Normalize reviews date
    if not reviews.empty and "date" in reviews.columns:
        reviews["date"] = pd.to_datetime(reviews["date"], errors="coerce")
        reviews = reviews.dropna(subset=["date"])
        reviews["date"] = reviews["date"].dt.date

    # Ensure expected columns exist (prevents KeyErrors)
    if not products.empty:
        for c in ["id", "title", "price", "category", "description", "url"]:
            if c not in products.columns:
                products[c] = None

    if not testimonials.empty:
        for c in ["page", "text", "rating"]:
            if c not in testimonials.columns:
                testimonials[c] = None

    if not reviews.empty:
        for c in ["product_id", "id", "text", "rating", "date"]:
            if c not in reviews.columns:
                reviews[c] = None

    return products, testimonials, reviews


@st.cache_resource(show_spinner=False)
def get_sentiment_pipe():
    # Render-friendly: default backend will be torch if installed
    # (pipeline will download model on first run)
    return pipeline("sentiment-analysis", model=MODEL_NAME)


@st.cache_data(show_spinner=False)
def run_sentiment_cached(texts: Tuple[str, ...]) -> List[dict]:
    """
    Cached sentiment run. Keyed by the exact text tuple.
    Keep texts as a tuple for hashability.
    """
    sa = get_sentiment_pipe()
    # Batch to reduce overhead (and reduce RAM spikes)
    # (small batches are safer on free-tier)
    batch_size = 16
    out: List[dict] = []
    for i in range(0, len(texts), batch_size):
        chunk = list(texts[i : i + batch_size])
        out.extend(sa(chunk))
    return out


def summarize_predictions(preds: List[dict]) -> pd.DataFrame:
    """
    Returns dataframe with columns:
    label (Positive/Negative), count, avg_conf
    """
    if not preds:
        return pd.DataFrame(columns=["label", "count", "avg_conf"])

    rows = []
    for p in preds:
        label_raw = (p.get("label") or "").upper()
        score = float(p.get("score") or 0.0)
        label = "Positive" if "POS" in label_raw else "Negative"
        rows.append({"label": label, "score": score})

    df = pd.DataFrame(rows)
    g = (
        df.groupby("label", as_index=False)
        .agg(count=("label", "size"), avg_conf=("score", "mean"))
        .sort_values("label")
    )
    return g


def make_bar_chart(summary_df: pd.DataFrame):
    if summary_df.empty:
        st.info("No predictions to plot for this month.")
        return

    # Ensure both classes appear (nice chart)
    for lab in ["Negative", "Positive"]:
        if lab not in set(summary_df["label"]):
            summary_df = pd.concat(
                [summary_df, pd.DataFrame([{"label": lab, "count": 0, "avg_conf": 0.0}])],
                ignore_index=True,
            )

    summary_df["avg_conf"] = summary_df["avg_conf"].fillna(0.0)

    if alt is not None:
        chart = (
            alt.Chart(summary_df)
            .mark_bar()
            .encode(
                x=alt.X("label:N", title="Sentiment"),
                y=alt.Y("count:Q", title="Number of reviews"),
                tooltip=[
                    alt.Tooltip("label:N", title="Sentiment"),
                    alt.Tooltip("count:Q", title="Count"),
                    alt.Tooltip("avg_conf:Q", title="Avg confidence", format=".3f"),
                ],
            )
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        # Fallback to Streamlit's bar_chart (no tooltip)
        tmp = summary_df.set_index("label")["count"]
        st.bar_chart(tmp)


def make_wordcloud(texts: Iterable[str]):
    if WordCloud is None or plt is None:
        st.info("WordCloud requires `wordcloud` and `matplotlib` installed.")
        return
    combined = " ".join([t for t in texts if isinstance(t, str) and t.strip()])
    if not combined.strip():
        st.info("Not enough text to generate a word cloud.")
        return
    wc = WordCloud(width=1100, height=450, background_color="black").generate(combined)
    fig = plt.figure(figsize=(12, 4))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(fig, clear_figure=True)


def local_scrape_button():
    """
    Runs scrape_data.py locally (works on your laptop).
    On Render, outbound scraping may be blocked / slow; we warn.
    """
    st.markdown("### Data refresh (optional)")
    st.caption("Recommended workflow: scrape locally → commit CSVs → deploy fast.")
    if st.button("Scrape / Refresh data now (local)"):
        if not os.path.exists(os.path.join(BASE_DIR, "scrape_data.py")):
            st.error("scrape_data.py not found in project root.")
            return

        with st.spinner("Running scraper..."):
            try:
                # Use current python environment
                proc = subprocess.run(
                    ["python", "scrape_data.py"],
                    cwd=BASE_DIR,
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
                if proc.returncode != 0:
                    st.error("Scraper failed.")
                    st.code(proc.stderr[-4000:] if proc.stderr else "No stderr")
                else:
                    st.success("Scrape complete. Reloading data...")
                    st.code(proc.stdout[-4000:] if proc.stdout else "No stdout")
                    # Clear caches so app reloads latest CSVs
                    st.cache_data.clear()
                    st.rerun()
            except Exception as e:
                st.error(f"Could not run scraper: {e}")


# ----------------------------
# UI
# ----------------------------
st.title("Brand Reputation Monitor (2023)")
st.caption(
    "Scrapes data from web-scraping.dev (Products, Testimonials, Reviews), filters Reviews by month in 2023, "
    "and runs Transformer-based sentiment analysis."
)

products, testimonials, reviews = load_all_data()

with st.sidebar:
    st.header("Navigation")
    section = st.radio("Go to", ["Reviews", "Products", "Testimonials"], index=0)

    st.divider()
    st.caption("Data files expected in ./data/")
    ok = all(os.path.exists(p) for p in [PRODUCTS_CSV, TESTIMONIALS_CSV, REVIEWS_CSV])
    st.write("Status:", "✅ found" if ok else "⚠️ missing one or more CSVs")

    local_scrape_button()


# ----------------------------
# Products / Testimonials
# ----------------------------
if section == "Products":
    st.subheader("Products")
    if products.empty:
        st.warning("No products loaded. Make sure data/products.csv exists.")
    else:
        st.dataframe(products, use_container_width=True)

elif section == "Testimonials":
    st.subheader("Testimonials")
    if testimonials.empty:
        st.warning("No testimonials loaded. Make sure data/testimonials.csv exists.")
    else:
        st.dataframe(testimonials, use_container_width=True)

# ----------------------------
# Reviews (core)
# ----------------------------
else:
    st.subheader("Reviews — Sentiment Analysis (Core Feature)")

    if reviews.empty:
        st.warning("No reviews loaded. Make sure data/reviews.csv exists.")
        st.stop()

    # Build month choices for 2023
    months = [date(2023, m, 1) for m in range(1, 13)]
    month_choice = st.select_slider(
        "Select a month in 2023",
        options=months,
        format_func=month_label,
        value=date(2023, 1, 1),
    )

    start, end = month_start_end(2023, month_choice.month)

    # Filter by month
    filtered = reviews.copy()
    filtered = filtered[(filtered["date"] >= start) & (filtered["date"] <= end)].copy()

    colA, colB = st.columns([1, 2])
    with colA:
        st.metric("Total reviews (selected month)", int(len(filtered)))
    with colB:
        st.markdown(f"**Filtering window:** {start} to {end}")

    if filtered.empty:
        st.info("No reviews in this month. Choose another month.")
        st.stop()

    # ---- Render-friendly: run sentiment ONLY on button click ----
    st.markdown("### Sentiment analysis")
    run_now = st.button("Run sentiment analysis for selected month")

    if not run_now:
        st.info(
            "Click the button to run sentiment analysis. "
            "This avoids re-running the model on every slider change (important for Render stability)."
        )
        st.markdown("### Reviews (filtered)")
        show_cols = [c for c in ["date", "product_id", "rating", "text"] if c in filtered.columns]
        st.dataframe(
            filtered[show_cols].sort_values("date", ascending=False),
            use_container_width=True,
        )
        # Optional word cloud preview (cheap, no model)
        st.markdown("### Word Cloud (bonus)")
        make_wordcloud(filtered["text"].astype(str).tolist())
        st.stop()

    with st.spinner("Running sentiment analysis on selected reviews..."):
        # Limit to keep Render safe (you can raise locally)
        MAX_REVIEWS = 250
        texts_list = filtered["text"].astype(str).fillna("").tolist()
        if len(texts_list) > MAX_REVIEWS:
            st.warning(f"Too many reviews for one run ({len(texts_list)}). Using first {MAX_REVIEWS} for stability.")
            texts_list = texts_list[:MAX_REVIEWS]

        texts = tuple(texts_list)
        preds = run_sentiment_cached(texts)

    # Metrics
    scores = [float(p.get("score") or 0.0) for p in preds]
    avg_all = sum(scores) / len(scores) if scores else 0.0

    pos_scores = [float(p.get("score") or 0.0) for p in preds if "POS" in str(p.get("label", "")).upper()]
    neg_scores = [float(p.get("score") or 0.0) for p in preds if "NEG" in str(p.get("label", "")).upper()]
    avg_pos = sum(pos_scores) / len(pos_scores) if pos_scores else 0.0
    avg_neg = sum(neg_scores) / len(neg_scores) if neg_scores else 0.0

    m1, m2, m3 = st.columns(3)
    m1.metric("Average confidence (all)", f"{avg_all:.3f}")
    m2.metric("Avg confidence — Positive", f"{avg_pos:.3f}")
    m3.metric("Avg confidence — Negative", f"{avg_neg:.3f}")

    st.markdown("### Sentiment summary")
    summary_df = summarize_predictions(preds)
    make_bar_chart(summary_df)

    # Attach predictions to dataframe (optional view)
    labels = []
    confs = []
    for p in preds:
        lab_raw = (p.get("label") or "").upper()
        labels.append("Positive" if "POS" in lab_raw else "Negative")
        confs.append(float(p.get("score") or 0.0))

    filtered_out = filtered.iloc[: len(labels)].copy()
    filtered_out["sentiment"] = labels
    filtered_out["confidence"] = confs

    st.markdown("### Reviews (filtered)")
    show_cols = [c for c in ["date", "product_id", "rating", "sentiment", "confidence", "text"] if c in filtered_out.columns]
    st.dataframe(
        filtered_out[show_cols].sort_values("date", ascending=False),
        use_container_width=True,
    )

    # Bonus word cloud
    st.markdown("### Word Cloud (bonus)")
    make_wordcloud(filtered_out["text"].astype(str).tolist())
