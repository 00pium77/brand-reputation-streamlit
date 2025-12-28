# Brand Reputation Monitor (2023) - Streamlit + Transformers

A Python Streamlit app that:
- Scrapes products, testimonials, and product reviews from https://web-scraping.dev
- Filters reviews by month (Jan-Dec 2023)
- Runs deep-learning sentiment analysis using Hugging Face Transformers
- Visualizes Positive vs Negative counts + average confidence
- (Bonus) Generates a word cloud for the selected month

## Local Run

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Scrape once (recommended)
python scrape_data.py

# Run app
streamlit run app.py
```

## Render Deployment

**Build command**
```bash
pip install -r requirements.txt
```

**Start command**
```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

> Tip: Commit the `data/` CSVs so the app starts instantly on Render.

## Files
- `app.py` - Streamlit UI
- `scrape_data.py` - Scraper that writes `data/products.csv`, `data/testimonials.csv`, `data/reviews.csv`
- `requirements.txt` - Python deps
- `data/` - generated data files
