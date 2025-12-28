# app.py
"""
Brand Reputation Monitor (2023) — Streamlit App
- Loads pre-scraped CSVs from ./data (products, testimonials, reviews)
- Reviews view: month selector (Jan–Dec 2023), filters by month, runs HF sentiment model,
  shows Positive/Negative counts + confidence metrics.
- Optional: "Scrape/Refresh data now" button runs scrape_data.py to regenerate CSVs.

Run locally:
  source .venv/bin/activate
  streamlit run app.py
"""

from __future__ import annotations

import os
import subprocess
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
from transformers import pipeline


# -----------------------------
# Paths
# -----------------------------
APP_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(APP_DIR, "data")

PRODUCTS_CSV = os.path.join(DATA_DIR, "products.csv")
TESTIMONIALS_CSV = os.path.join(DATA_DIR, "testimonials.csv")
REVIEWS_CSV = os.path.join(DATA_DIR, "reviews.csv")

SCRAPER_PY = os.path.join(APP_DIR, "scrape_data.py")


# -----------------------------
# Helpers
# -----------------------------
def ensure_pandas(obj):
    """Convert dataframe-like objects (Arrow/Polars/dict/list) to pandas for Streamlit."""
    if isinstance(obj, pd.DataFrame):
        return obj
    if hasattr(obj, "to_pandas"):
        return obj.to_pandas()
    if hasattr(obj, "to_native"):
        native = obj.to_native()
        if isinstance(native, pd.DataFrame):
            return native
        if hasattr(native, "to_pandas"):
            return native.to_pandas()
        return pd.DataFrame(native)
    try:
        return pd.DataFrame(obj)
    except Exception:
        return pd.DataFrame({"value": [str(obj)]})


def month_label(dt: pd.Timestamp) -> str:
    return dt.strftime("%b %Y")  # Jan 2023


def month_range_2023() -> List[pd.Timestamp]:
    return [pd.Timestamp(2023, m, 1) for m in range(1, 13)]


def month_start_end(selected_month_start: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(selected_month_start.date())
    end = start + pd.offsets.MonthEnd(0)
    return start, end


# -----------------------------
# Data Loading
# -----------------------------
@st.cache_data(show_spinner=False)
def load_csv(path: str, *, columns: List[str], parse_dates: List[str] | None = None) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=columns)

    df = pd.read_csv(path)
    # Ensure columns exist
    for c in columns:
        if c not in df.columns:
            df[c] = None

    if parse_dates:
        for c in parse_dates:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")

    return df[columns]


@st.cache_resource(show_spinner=False)
def get_sentiment_pipe():
    # Default backend is usually torch; keep it simple/stable for deployment.
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")


@st.cache_data(show_spinner=False)
def run_sentiment(texts_tuple: Tuple[str, ...]) -> pd.DataFrame:
    """
    Cached sentiment results for a given tuple of texts.
    Returns DataFrame with columns: label, score
    """
    if not texts_tuple:
        return pd.DataFrame(columns=["label", "score"])

    sa = get_sentiment_pipe()
    preds = sa(list(texts_tuple), truncation=True, batch_size=16)
    pred_df = pd.DataFrame(preds)
    if not pred_df.empty and "label" in pred_df.columns:
        pred_df["label"] = pred_df["label"].astype(str).str.capitalize()  # Positive/Negative
    return pred_df


def scrape_refresh_data() -> Tuple[int, str]:
    """
    Runs scrape_data.py to regenerate ./data/*.csv
    Returns (exit_code, combined_output).
    """
    if not os.path.exists(SCRAPER_PY):
        return 1, f"Missing scraper script: {SCRAPER_PY}"

    cmd = ["python", SCRAPER_PY]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    return proc.returncode, out


# -----------------------------
# UI
# -----------------------------
st.set_page_config(
    page_title="Brand Reputation Monitor (2023)",
    page_icon="📊",
    layout="wide",
)

st.title("Brand Reputation Monitor (2023)")
st.caption(
    "Scrapes data from web-scraping.dev (Products, Testimonials, Reviews), "
    "filters Reviews by month in 2023, and runs Transformer-based sentiment analysis."
)

with st.sidebar:
    st.header("Navigation")
    section = st.radio("Go to", ["Products", "Testimonials", "Reviews"], index=2)

    st.divider()
    st.subheader("Data")
    st.caption("Data files expected in ./data/")
    if st.button("Scrape / Refresh data now", type="primary"):
        with st.spinner("Running scraper (scrape_data.py)…"):
            code, output = scrape_refresh_data()
        if code == 0:
            st.success("Scrape complete. Reloading app data…")
            # Clear cached data so re-reads happen
            load_csv.clear()
            st.rerun()
        else:
            st.error("Scrape failed. See output below.")
            st.code(output)

    st.caption("Tip: Commit ./data/*.csv to GitHub so the deployed app loads instantly.")


# Load data (cached)
products = load_csv(
    PRODUCTS_CSV,
    columns=["id", "title", "price", "category", "description", "url"],
    parse_dates=None,
)
testimonials = load_csv(
    TESTIMONIALS_CSV,
    columns=["page", "text", "rating"],
    parse_dates=None,
)
reviews = load_csv(
    REVIEWS_CSV,
    columns=["product_id", "id", "text", "rating", "date"],
    parse_dates=["date"],
)


# -----------------------------
# Section: Products
# -----------------------------
if section == "Products":
    st.subheader("Products")
    if products.empty:
        st.warning("No products data found. Click 'Scrape / Refresh data now' in the sidebar.")
    else:
        st.dataframe(ensure_pandas(products), use_container_width=True)
        st.download_button(
            "Download products.csv",
            data=products.to_csv(index=False).encode("utf-8"),
            file_name="products.csv",
            mime="text/csv",
        )


# -----------------------------
# Section: Testimonials
# -----------------------------
elif section == "Testimonials":
    st.subheader("Testimonials")
    if testimonials.empty:
        st.warning("No testimonials data found. Click 'Scrape / Refresh data now' in the sidebar.")
    else:
        st.dataframe(ensure_pandas(testimonials), use_container_width=True)
        st.download_button(
            "Download testimonials.csv",
            data=testimonials.to_csv(index=False).encode("utf-8"),
            file_name="testimonials.csv",
            mime="text/csv",
        )


# -----------------------------
# Section: Reviews (Core Feature)
# -----------------------------
else:
    st.subheader("Reviews — Sentiment Analysis (Core Feature)")

    if reviews.empty or reviews["date"].isna().all():
        st.warning("No reviews data found (or dates could not be parsed). Click 'Scrape / Refresh data now'.")
        st.stop()

    # Month selector for 2023
    months = month_range_2023()
    month_labels = [month_label(m) for m in months]

    st.markdown("**Select a month in 2023**")
    selected_label = st.select_slider(
        label="Month",
        options=month_labels,
        value=month_labels[0],
    )
    selected_month_start = months[month_labels.index(selected_label)]
    start_ts, end_ts = month_start_end(selected_month_start)

    # Filter reviews by selected month
    filtered = reviews.copy()
    filtered = filtered.dropna(subset=["date"])
    filtered = filtered[(filtered["date"] >= start_ts) & (filtered["date"] <= end_ts)].copy()

    # Summary
    top_left, top_right = st.columns([1, 2])
    with top_left:
        st.metric("Total reviews (selected month)", int(len(filtered)))
    with top_right:
        st.write(f"Filtering window: **{start_ts.date()}** to **{end_ts.date()}**")

    if filtered.empty:
        st.info("No reviews found for this month.")
        st.stop()

    # Run sentiment analysis
    texts = filtered["text"].fillna("").astype(str).tolist()
    texts_tuple = tuple(texts)

    with st.spinner("Running sentiment analysis on selected reviews…"):
        pred_df = run_sentiment(texts_tuple)

    # Attach predictions back to filtered
    if not pred_df.empty and len(pred_df) == len(filtered):
        filtered["sentiment"] = pred_df["label"].values
        filtered["confidence"] = pred_df["score"].values
    else:
        # Safety fallback
        filtered["sentiment"] = None
        filtered["confidence"] = None

    # Confidence metrics (rubric)
    if "confidence" in filtered.columns and filtered["confidence"].notna().any():
        avg_conf = float(filtered["confidence"].mean())
        avg_by_label: Dict[str, float] = (
            filtered.groupby("sentiment")["confidence"].mean().dropna().to_dict()
        )
    else:
        avg_conf = 0.0
        avg_by_label = {}

    c1, c2, c3 = st.columns(3)
    c1.metric("Average confidence (all)", f"{avg_conf:.3f}")
    c2.metric("Avg confidence — Positive", f"{avg_by_label.get('Positive', 0.0):.3f}")
    c3.metric("Avg confidence — Negative", f"{avg_by_label.get('Negative', 0.0):.3f}")

    # Sentiment summary chart (with tooltip if Altair is available)
    st.markdown("### Sentiment summary")

    counts = (
        filtered["sentiment"]
        .value_counts()
        .reindex(["Positive", "Negative"], fill_value=0)
        .reset_index()
    )
    counts.columns = ["sentiment", "count"]

    # Add avg confidence per sentiment to support "Advanced" rubric
    conf_by_sent = (
        filtered.groupby("sentiment")["confidence"].mean().reindex(["Positive", "Negative"])
    )
    counts["avg_confidence"] = counts["sentiment"].map(conf_by_sent).fillna(0.0)

    # Try Altair tooltips; fallback to st.bar_chart + table
    try:
        import altair as alt  # noqa: F401

        chart = (
            alt.Chart(counts)
            .mark_bar()
            .encode(
                x=alt.X("sentiment:N", title="Sentiment"),
                y=alt.Y("count:Q", title="Number of reviews"),
                tooltip=[
                    alt.Tooltip("sentiment:N", title="Sentiment"),
                    alt.Tooltip("count:Q", title="Count"),
                    alt.Tooltip("avg_confidence:Q", title="Avg confidence", format=".3f"),
                ],
            )
            .properties(height=320)
        )
        st.altair_chart(chart, use_container_width=True)
    except Exception:
        st.bar_chart(counts.set_index("sentiment")["count"])
        st.caption(
            f"Avg confidence (all): {avg_conf:.3f} | "
            f"Positive: {avg_by_label.get('Positive', 0.0):.3f} | "
            f"Negative: {avg_by_label.get('Negative', 0.0):.3f}"
        )

    # Show reviews table
    st.markdown("### Reviews (filtered)")
    show_cols = ["date", "product_id", "rating", "sentiment", "confidence", "text"]
    show_cols = [c for c in show_cols if c in filtered.columns]

    # Sort newest first
    filtered_view = filtered[show_cols].sort_values("date", ascending=False)

    st.dataframe(ensure_pandas(filtered_view), use_container_width=True)

    st.download_button(
        "Download filtered reviews (CSV)",
        data=filtered_view.to_csv(index=False).encode("utf-8"),
        file_name=f"reviews_{start_ts.strftime('%Y_%m')}.csv",
        mime="text/csv",
    )
