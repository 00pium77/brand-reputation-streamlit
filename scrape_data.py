"""
scrape_data.py
Scrapes products, testimonials, and reviews from https://web-scraping.dev

Recommended workflow:
1) Run: python scrape_data.py
2) Commit ./data/*.csv to GitHub so Streamlit runs fast in production.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE = "https://web-scraping.dev"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

PRODUCTS_CSV = os.path.join(DATA_DIR, "products.csv")
TESTIMONIALS_CSV = os.path.join(DATA_DIR, "testimonials.csv")
REVIEWS_CSV = os.path.join(DATA_DIR, "reviews.csv")

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; BrandReputationMonitor/1.0)",
    "Accept": "text/html,application/json;q=0.9,*/*;q=0.8",
}

# Keep cookies/session across requests (important on this playground)
SESSION = requests.Session()


def get(url: str, *, params=None, headers=None, timeout=30) -> requests.Response:
    h = dict(DEFAULT_HEADERS)
    if headers:
        h.update(headers)
    resp = SESSION.get(url, params=params, headers=h, timeout=timeout)
    return resp


def get_json(url: str, *, params=None, headers=None, timeout=30) -> dict:
    resp = get(url, params=params, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def save_csv_with_headers(df: pd.DataFrame, path: str, columns: List[str]) -> None:
    """Write CSV even if df is empty so it still has headers."""
    if df is None or df.empty:
        pd.DataFrame(columns=columns).to_csv(path, index=False)
        return

    # Ensure all columns exist
    for c in columns:
        if c not in df.columns:
            df[c] = None
    df[columns].to_csv(path, index=False)


# -------------------- PRODUCTS --------------------

def scrape_products_via_api(max_pages: int = 50) -> pd.DataFrame:
    results = []
    for page in range(1, max_pages + 1):
        try:
            data = get_json(f"{BASE}/api/products", params={"page": page})
        except Exception:
            break

        page_items = data.get("results") or data.get("items") or data.get("data") or []
        if not page_items:
            break

        for item in page_items:
            pid = item.get("id")
            results.append(
                {
                    "id": pid,
                    "title": item.get("title") or item.get("name"),
                    "price": item.get("price"),
                    "category": item.get("category"),
                    "description": item.get("description"),
                    "url": item.get("url")
                    or item.get("href")
                    or (f"{BASE}/product/{pid}" if pid else None),
                }
            )
        time.sleep(0.15)

    df = pd.DataFrame(results)
    if df.empty or "id" not in df.columns:
        return pd.DataFrame()

    df = df.dropna(subset=["id"])
    df["id"] = df["id"].astype(int)
    df = df.drop_duplicates(subset=["id"]).sort_values("id")
    return df


def scrape_products_via_html(max_pages: int = 50) -> pd.DataFrame:
    results = []
    for page in range(1, max_pages + 1):
        resp = get(f"{BASE}/products", params={"page": page})
        if resp.status_code != 200:
            break

        soup = BeautifulSoup(resp.text, "html.parser")
        links = soup.select('a[href*="/product/"]')

        ids_seen = set()
        for a in links:
            href = a.get("href") or ""
            m = re.search(r"/product/(\d+)", href)
            if not m:
                continue
            pid = int(m.group(1))
            if pid in ids_seen:
                continue
            ids_seen.add(pid)
            results.append(
                {
                    "id": pid,
                    "title": a.get_text(strip=True) or None,
                    "url": BASE + href if href.startswith("/") else href,
                }
            )

        if not ids_seen:
            break

        time.sleep(0.15)

    df = pd.DataFrame(results)
    if df.empty or "id" not in df.columns:
        return pd.DataFrame()
    df = df.dropna(subset=["id"]).drop_duplicates(subset=["id"]).sort_values("id")
    return df


def scrape_products() -> pd.DataFrame:
    df = scrape_products_via_api()
    if df.empty:
        df = scrape_products_via_html()
    return df


# -------------------- TESTIMONIALS --------------------

def find_secret_token_from_testimonials_html(html: str) -> Optional[str]:
    m = re.search(r"(secret\d{1,})", html)
    return m.group(1) if m else None


def scrape_testimonials(max_pages: int = 50) -> pd.DataFrame:
    landing = get(f"{BASE}/testimonials")
    landing.raise_for_status()
    token = find_secret_token_from_testimonials_html(landing.text) or "secret123"

    results = []
    for page in range(1, max_pages + 1):
        resp = get(
            f"{BASE}/api/testimonials",
            params={"page": page},
            headers={"Referer": f"{BASE}/testimonials", "X-Secret-Token": token},
        )
        if resp.status_code != 200:
            break

        soup = BeautifulSoup(resp.text, "html.parser")
        cards = soup.select(".testimonial")
        if not cards:
            break

        for c in cards:
            text_el = c.select_one(".text")
            text = text_el.get_text(" ", strip=True) if text_el else c.get_text(" ", strip=True)
            rating = len(c.select(".rating svg")) or None
            results.append({"page": page, "text": text, "rating": rating})

        time.sleep(0.15)

    return pd.DataFrame(results)


# -------------------- REVIEWS --------------------

def get_csrf_token_for_product(product_id: int) -> Optional[str]:
    resp = get(f"{BASE}/product/{product_id}")
    if resp.status_code != 200:
        return None
    soup = BeautifulSoup(resp.text, "html.parser")
    inp = soup.select_one('input[name="csrf-token"]')
    if inp and inp.get("value"):
        return inp["value"]
    return None


def _collect_review_like_dicts(obj: Any, acc: List[Dict[str, Any]]) -> None:
    """
    Recursively traverse JSON-like object and collect dicts that look like reviews.
    We accept various key names because playground can change schema.
    """
    if isinstance(obj, dict):
        keys = set(obj.keys())
        # Common review-ish signatures
        has_text = any(k in keys for k in ("text", "content", "body", "review"))
        has_rating = any(k in keys for k in ("rating", "stars", "score"))
        has_date = any(k in keys for k in ("date", "created", "created_at", "createdAt"))

        if has_text and has_rating:
            acc.append(obj)

        for v in obj.values():
            _collect_review_like_dicts(v, acc)

    elif isinstance(obj, list):
        for it in obj:
            _collect_review_like_dicts(it, acc)


def _try_parse_json(text: str) -> Optional[Any]:
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def scrape_reviews_from_product_html(product_id: int) -> List[dict]:
    """
    Fallback strategy:
    Many web-scraping.dev pages hide reviews JSON in <script> tags.
    We parse scripts and search for review-like objects.
    """
    resp = get(f"{BASE}/product/{product_id}")
    if resp.status_code != 200:
        return []

    soup = BeautifulSoup(resp.text, "html.parser")

    candidates: List[Dict[str, Any]] = []

    # 1) Script tags that are pure JSON
    for s in soup.select('script[type="application/json"]'):
        parsed = _try_parse_json(s.get_text())
        if parsed is not None:
            _collect_review_like_dicts(parsed, candidates)

    # 2) Common Next.js style: <script id="__NEXT_DATA__" type="application/json">...</script>
    next_data = soup.select_one('script#__NEXT_DATA__')
    if next_data:
        parsed = _try_parse_json(next_data.get_text())
        if parsed is not None:
            _collect_review_like_dicts(parsed, candidates)

    # 3) Last resort: scan all scripts and try to extract a JSON array that contains "rating" and "date"
    if not candidates:
        for s in soup.find_all("script"):
            txt = s.get_text(" ", strip=True)
            if not txt:
                continue
            if "rating" not in txt or "date" not in txt:
                continue

            # very loose: find first [...] that might be JSON
            m = re.search(r"(\[\s*\{.*?\}\s*\])", txt, flags=re.DOTALL)
            if m:
                parsed = _try_parse_json(m.group(1))
                if parsed is not None:
                    _collect_review_like_dicts(parsed, candidates)

    # Normalize collected dicts into our schema
    out: List[dict] = []
    seen = set()

    for c in candidates:
        text = c.get("text") or c.get("content") or c.get("body") or c.get("review")
        rating = c.get("rating") or c.get("stars") or c.get("score")
        date = c.get("date") or c.get("created") or c.get("created_at") or c.get("createdAt")
        rid = c.get("id") or c.get("review_id") or c.get("reviewId")

        if not text or rating is None or not date:
            continue

        key = (product_id, str(rid), str(date), str(text)[:60])
        if key in seen:
            continue
        seen.add(key)

        out.append(
            {
                "product_id": product_id,
                "id": rid,
                "text": str(text),
                "rating": rating,
                "date": str(date),
            }
        )

    return out


def scrape_reviews_via_api(product_id: int, max_pages: int = 50) -> List[dict]:
    """
    Try API with multiple possible param names (playground can change).
    If all attempts return 0, we fallback to HTML hidden JSON.
    """
    csrf = get_csrf_token_for_product(product_id) or "secret-csrf-token-123"

    param_variants = [
        {"product_id": product_id},
        {"product": product_id},
        {"productId": product_id},
        {"productID": product_id},
    ]

    headers = {
        "X-Csrf-Token": csrf,
        "x-csrf-token": csrf,
        "Referer": f"{BASE}/product/{product_id}",
        "Accept": "application/json",
    }

    for variant in param_variants:
        out: List[dict] = []
        for page in range(1, max_pages + 1):
            params = {"page": page, **variant}
            resp = get(f"{BASE}/api/reviews", params=params, headers=headers)
            if resp.status_code != 200:
                out = []
                break

            try:
                data = resp.json()
            except Exception:
                out = []
                break

            items = data.get("results") or data.get("items") or data.get("data") or []
            if not items:
                break

            for it in items:
                out.append(
                    {
                        "product_id": product_id,
                        "id": it.get("id"),
                        "text": it.get("text") or it.get("content") or it.get("body"),
                        "rating": it.get("rating") or it.get("stars") or it.get("score"),
                        "date": it.get("date") or it.get("created_at") or it.get("createdAt"),
                    }
                )
            time.sleep(0.1)

        # If this variant produced something, use it
        out = [r for r in out if r.get("text") and r.get("date")]
        if out:
            return out

    return []


def scrape_reviews_for_product(product_id: int) -> List[dict]:
    # 1) API tries
    out = scrape_reviews_via_api(product_id)
    if out:
        return out

    # 2) HTML hidden JSON fallback
    return scrape_reviews_from_product_html(product_id)


def scrape_all_reviews(products_df: pd.DataFrame, limit_products: Optional[int] = None) -> pd.DataFrame:
    product_ids = products_df["id"].dropna().astype(int).tolist()
    if limit_products:
        product_ids = product_ids[:limit_products]

    all_reviews: List[dict] = []
    for i, pid in enumerate(product_ids, start=1):
        print(f"[reviews] product {pid} ({i}/{len(product_ids)})")
        try:
            all_reviews.extend(scrape_reviews_for_product(pid))
        except Exception as e:
            print(f"  - failed product {pid}: {e}")

    df = pd.DataFrame(all_reviews)
    if df.empty:
        return df

    # Parse date → keep as ISO string for CSV
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["date"] = df["date"].dt.date.astype(str)
    return df


# -------------------- MAIN --------------------

def main():
    print("Scraping products...")
    products = scrape_products()
    if products.empty:
        raise RuntimeError("Could not scrape products (API + HTML fallbacks failed).")

    save_csv_with_headers(
        products,
        PRODUCTS_CSV,
        columns=["id", "title", "price", "category", "description", "url"],
    )
    print(f"Saved {len(products)} products -> {PRODUCTS_CSV}")

    print("Scraping testimonials...")
    testimonials = scrape_testimonials()
    save_csv_with_headers(testimonials, TESTIMONIALS_CSV, columns=["page", "text", "rating"])
    print(f"Saved {len(testimonials)} testimonials -> {TESTIMONIALS_CSV}")

    print("Scraping reviews (all products)...")
    reviews = scrape_all_reviews(products)
    save_csv_with_headers(reviews, REVIEWS_CSV, columns=["product_id", "id", "text", "rating", "date"])
    print(f"Saved {len(reviews)} reviews -> {REVIEWS_CSV}")

    print("Done.")


if __name__ == "__main__":
    main()
