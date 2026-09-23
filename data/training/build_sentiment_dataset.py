import pandas as pd
import requests
import time
import re
from pathlib import Path
from datetime import timedelta
from dotenv import load_dotenv
from transformers import pipeline

# ============================================================
# CONFIG
# ============================================================

load_dotenv()

INPUT_FILE = Path("data/training/layoff_events.csv")
OUTPUT_FILE = Path("data/training/historical_sentiment.csv")

# Increase this gradually if your News/API limits allow it
MAX_ARTICLES = 3000

REQUEST_TIMEOUT = 10
SLEEP_SECONDS = 0.2

# ============================================================
# LOAD FINBERT
# ============================================================

print("\nLoading FinBERT...")

sentiment_model = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert"
)

print("FinBERT loaded successfully.")

# ============================================================
# KEYWORDS
# ============================================================

KEYWORDS = {
    "layoff": [
        "layoff",
        "layoffs",
        "laid off",
        "job cuts",
        "workforce reduction",
        "workforce reductions"
    ],

    "restructuring": [
        "restructuring",
        "reorganization",
        "reorganizing",
        "restructure"
    ],

    "cost_cutting": [
        "cost cutting",
        "cost-cutting",
        "cost reduction",
        "cost reductions",
        "reduce costs",
        "cut costs"
    ],

    "hiring_freeze": [
        "hiring freeze",
        "hiring slowdown",
        "freeze hiring",
        "paused hiring"
    ]
}

# ============================================================
# HELPERS
# ============================================================

def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = re.sub(r"\s+", " ", text)
    return text.strip()


def count_keywords(text, keywords):
    text = text.lower()

    return sum(
        len(re.findall(r"\b" + re.escape(keyword) + r"\b", text))
        for keyword in keywords
    )


def analyze_text(text):
    """
    Run FinBERT on article text.
    """

    text = clean_text(text)

    if not text:
        return {
            "label": "neutral",
            "score": 0.0,
            "layoff_keyword_count": 0,
            "restructuring_keyword_count": 0,
            "cost_cutting_keyword_count": 0,
            "hiring_freeze_keyword_count": 0
        }

    # Keep input reasonably small for FinBERT
    text_for_model = text[:2000]

    try:
        result = sentiment_model(
            text_for_model,
            truncation=True
        )[0]

        label = result["label"].lower()
        score = float(result["score"])

    except Exception as e:
        print(f"FinBERT error: {e}")

        label = "neutral"
        score = 0.0

    return {
        "label": label,
        "score": score,

        "layoff_keyword_count": count_keywords(
            text,
            KEYWORDS["layoff"]
        ),

        "restructuring_keyword_count": count_keywords(
            text,
            KEYWORDS["restructuring"]
        ),

        "cost_cutting_keyword_count": count_keywords(
            text,
            KEYWORDS["cost_cutting"]
        ),

        "hiring_freeze_keyword_count": count_keywords(
            text,
            KEYWORDS["hiring_freeze"]
        )
    }


def fetch_article(url):
    """
    Fetch article page and extract visible text.

    This is intentionally lightweight.
    Some websites may block requests.
    """

    if not isinstance(url, str) or not url.startswith("http"):
        return ""

    try:

        headers = {
            "User-Agent": (
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/120 Safari/537.36"
            )
        }

        response = requests.get(
            url,
            headers=headers,
            timeout=REQUEST_TIMEOUT
        )

        if response.status_code != 200:
            return ""

        html = response.text

        # Remove scripts/styles
        html = re.sub(
            r"<script.*?</script>",
            " ",
            html,
            flags=re.DOTALL | re.IGNORECASE
        )

        html = re.sub(
            r"<style.*?</style>",
            " ",
            html,
            flags=re.DOTALL | re.IGNORECASE
        )

        # Remove HTML tags
        text = re.sub(
            r"<[^>]+>",
            " ",
            html
        )

        text = clean_text(text)

        return text[:10000]

    except Exception:
        return ""


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading historical layoff events...")

events = pd.read_csv(INPUT_FILE)

print(f"Total events available: {len(events):,}")

# ============================================================
# CLEAN DATA
# ============================================================

events["Company"] = (
    events["Company"]
    .fillna("")
    .astype(str)
    .str.strip()
)

events["Date"] = pd.to_datetime(
    events["Date"],
    errors="coerce"
)

events["Source"] = (
    events["Source"]
    .fillna("")
    .astype(str)
    .str.strip()
)

events = events.dropna(
    subset=["Date"]
)

events = events[
    (events["Company"] != "") &
    (events["Source"] != "")
]

# Remove duplicate company/date/source combinations

events = events.drop_duplicates(
    subset=["Company", "Date", "Source"]
)

# ============================================================
# LIMIT ARTICLES
# ============================================================

if len(events) > MAX_ARTICLES:

    print(
        f"\nLimiting processing to "
        f"{MAX_ARTICLES:,} articles."
    )

    # Spread selection across the historical period
    events = (
        events
        .sort_values("Date")
        .iloc[
            ::max(1, len(events) // MAX_ARTICLES)
        ]
        .head(MAX_ARTICLES)
    )

print(
    f"Articles selected: {len(events):,}"
)

# ============================================================
# PROCESS ARTICLES
# ============================================================

results = []

successful = 0
failed = 0

print("\nProcessing articles...\n")

for index, row in events.iterrows():

    company = row["Company"]
    event_date = row["Date"]
    source_url = row["Source"]

    print(
        f"[{len(results) + 1}/{len(events)}] "
        f"{company} | {event_date.date()}"
    )

    article_text = fetch_article(source_url)

    if not article_text:

        failed += 1

        print("  -> Could not fetch article")

        continue

    sentiment = analyze_text(article_text)

    # --------------------------------------------------------
    # IMPORTANT:
    # We do NOT use the exact layoff event date as a future
    # prediction snapshot.
    #
    # Instead, sentiment is assigned to a snapshot slightly
    # BEFORE the layoff event.
    #
    # This reduces direct target leakage.
    # --------------------------------------------------------

    snapshot_date = (
        event_date - timedelta(days=7)
    )

    results.append({
        "company": company,
        "snapshot_date": snapshot_date,

        "news_count": 1,

        "negative_news_ratio": (
            1 if sentiment["label"] == "negative"
            else 0
        ),

        "positive_news_ratio": (
            1 if sentiment["label"] == "positive"
            else 0
        ),

        "neutral_news_ratio": (
            1 if sentiment["label"] == "neutral"
            else 0
        ),

        "avg_sentiment": (
            sentiment["score"]
            if sentiment["label"] == "positive"
            else -sentiment["score"]
            if sentiment["label"] == "negative"
            else 0
        ),

        "layoff_keyword_count":
            sentiment["layoff_keyword_count"],

        "restructuring_keyword_count":
            sentiment["restructuring_keyword_count"],

        "cost_cutting_keyword_count":
            sentiment["cost_cutting_keyword_count"],

        "hiring_freeze_keyword_count":
            sentiment["hiring_freeze_keyword_count"]
    })

    successful += 1

    if len(results) % 25 == 0:

        print(
            f"  -> Successful: {successful}, "
            f"Failed: {failed}"
        )

    time.sleep(SLEEP_SECONDS)

# ============================================================
# CHECK RESULTS
# ============================================================

if not results:

    print(
        "\nERROR: No articles were successfully processed."
    )

    raise SystemExit(1)

sentiment_df = pd.DataFrame(results)

sentiment_df["snapshot_date"] = pd.to_datetime(
    sentiment_df["snapshot_date"]
)

# ============================================================
# AGGREGATE COMPANY + DATE
# ============================================================

print("\nAggregating company/date sentiment...")

sentiment_df = (
    sentiment_df
    .groupby(
        ["company", "snapshot_date"],
        as_index=False
    )
    .agg({
        "news_count": "sum",

        "negative_news_ratio": "mean",
        "positive_news_ratio": "mean",
        "neutral_news_ratio": "mean",

        "avg_sentiment": "mean",

        "layoff_keyword_count": "sum",
        "restructuring_keyword_count": "sum",
        "cost_cutting_keyword_count": "sum",
        "hiring_freeze_keyword_count": "sum"
    })
)

# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

sentiment_df.to_csv(
    OUTPUT_FILE,
    index=False
)

# ============================================================
# SUMMARY
# ============================================================

print("\n========================================")
print("HISTORICAL SENTIMENT DATASET CREATED")
print("========================================")

print(
    f"Articles processed : {successful:,}"
)

print(
    f"Articles failed    : {failed:,}"
)

print(
    f"Final rows         : {len(sentiment_df):,}"
)

print(
    f"Unique companies   : "
    f"{sentiment_df['company'].nunique():,}"
)

print(
    f"Date range         : "
    f"{sentiment_df['snapshot_date'].min().date()} "
    f"to "
    f"{sentiment_df['snapshot_date'].max().date()}"
)

print(
    f"Saved to           : {OUTPUT_FILE}"
)

print("========================================\n")

