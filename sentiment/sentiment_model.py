import re

from transformers import pipeline


# Load FinBERT once.
# Loading it inside every function would be very slow.

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert"
)


# ---------------------------------------
# Layoff-related keywords
# ---------------------------------------

LAYOFF_KEYWORDS = [
    "layoff",
    "layoffs",
    "laid off",
    "job cuts",
    "job cut",
    "workforce reduction",
    "workforce reductions",
    "staff reduction",
    "staff reductions",
    "employee reduction",
    "employee reductions"
]


RESTRUCTURING_KEYWORDS = [
    "restructuring",
    "restructure",
    "reorganization",
    "reorganize",
    "organizational changes"
]


COST_CUTTING_KEYWORDS = [
    "cost cutting",
    "cost-cutting",
    "cost reduction",
    "cost reductions",
    "reduce costs",
    "expense reduction",
    "efficiency measures"
]


HIRING_FREEZE_KEYWORDS = [
    "hiring freeze",
    "hiring slowdown",
    "freeze hiring",
    "paused hiring",
    "pause hiring",
    "slowing hiring"
]


def count_keywords(
    text,
    keywords
):
    """
    Count occurrences of a group of keywords.
    """

    text = text.lower()

    count = 0

    for keyword in keywords:

        count += text.count(
            keyword.lower()
        )

    return count


def analyze_article(article):
    """
    Run FinBERT on one article.
    """

    text = article.get(
        "text",
        ""
    )

    if not text.strip():
        return {
            "label": "neutral",
            "score": 0.0,
            "layoff_keywords": 0,
            "restructuring_keywords": 0,
            "cost_cutting_keywords": 0,
            "hiring_freeze_keywords": 0
        }

    # FinBERT
    result = sentiment_pipeline(
        text[:2000],
        truncation=True
    )[0]

    label = result["label"].lower()
    score = float(
        result["score"]
    )

    return {

        "label": label,

        "score": score,

        "layoff_keywords":
            count_keywords(
                text,
                LAYOFF_KEYWORDS
            ),

        "restructuring_keywords":
            count_keywords(
                text,
                RESTRUCTURING_KEYWORDS
            ),

        "cost_cutting_keywords":
            count_keywords(
                text,
                COST_CUTTING_KEYWORDS
            ),

        "hiring_freeze_keywords":
            count_keywords(
                text,
                HIRING_FREEZE_KEYWORDS
            )
    }


def analyze_articles(articles):
    """
    Analyze all articles in a fast batch instead of a slow loop.
    """
    if not articles:
        return []

    print(f"Running FinBERT batch inference on {len(articles)} articles...")

    texts = [
        (a.get("text") or "").strip()[:2000]
        for a in articles
    ]

    # Batch run on valid non-empty texts
    valid_indices = [i for i, t in enumerate(texts) if t]
    valid_texts = [texts[i] for i in valid_indices]

    batch_results = {}
    if valid_texts:
        preds = sentiment_pipeline(valid_texts, truncation=True, batch_size=16)
        for idx, pred in zip(valid_indices, preds):
            batch_results[idx] = pred

    analyzed = []
    for i, article in enumerate(articles):
        text = texts[i]
        pred = batch_results.get(i, {"label": "neutral", "score": 0.0})

        sentiment = {
            "label": pred["label"].lower(),
            "score": float(pred["score"]),
            "layoff_keywords": count_keywords(text, LAYOFF_KEYWORDS),
            "restructuring_keywords": count_keywords(text, RESTRUCTURING_KEYWORDS),
            "cost_cutting_keywords": count_keywords(text, COST_CUTTING_KEYWORDS),
            "hiring_freeze_keywords": count_keywords(text, HIRING_FREEZE_KEYWORDS),
        }
        analyzed.append({**article, **sentiment})

    print(f"Processed all {len(articles)} articles in batch!")
    return analyzed
def create_sentiment_features(
    analyzed_articles
):
    """
    Convert article-level sentiment into
    company-level ML features.
    """

    total = len(
        analyzed_articles
    )

    if total == 0:

        return {
            "news_count": 0,
            "negative_news_ratio": 0,
            "positive_news_ratio": 0,
            "neutral_news_ratio": 0,
            "avg_sentiment": 0,
            "negative_sentiment_score": 0,
            "sentiment_volatility": 0,
            "layoff_keyword_count": 0,
            "restructuring_keyword_count": 0,
            "cost_cutting_keyword_count": 0,
            "hiring_freeze_keyword_count": 0
        }

    labels = [
        article["label"]
        for article in analyzed_articles
    ]

    scores = [
        article["score"]
        for article in analyzed_articles
    ]

    # --------------------------------
    # Sentiment ratios
    # --------------------------------

    negative_count = labels.count(
        "negative"
    )

    positive_count = labels.count(
        "positive"
    )

    neutral_count = labels.count(
        "neutral"
    )

    negative_ratio = (
        negative_count / total
    )

    positive_ratio = (
        positive_count / total
    )

    neutral_ratio = (
        neutral_count / total
    )

    # --------------------------------
    # Average sentiment
    # --------------------------------

    sentiment_values = []

    for article in analyzed_articles:

        label = article["label"]
        score = article["score"]

        if label == "positive":
            value = score

        elif label == "negative":
            value = -score

        else:
            value = 0

        sentiment_values.append(
            value
        )

    avg_sentiment = (
        sum(sentiment_values)
        / len(sentiment_values)
    )

    # --------------------------------
    # Negative sentiment strength
    # --------------------------------

    negative_scores = [
        article["score"]
        for article in analyzed_articles
        if article["label"] == "negative"
    ]

    if negative_scores:

        negative_sentiment_score = (
            sum(negative_scores)
            / len(negative_scores)
        )

    else:

        negative_sentiment_score = 0

    # --------------------------------
    # Sentiment volatility
    # --------------------------------

    if len(sentiment_values) > 1:

        import statistics

        sentiment_volatility = (
            statistics.stdev(
                sentiment_values
            )
        )

    else:

        sentiment_volatility = 0

    # --------------------------------
    # Keyword signals
    # --------------------------------

    layoff_keyword_count = sum(
        article[
            "layoff_keywords"
        ]
        for article in analyzed_articles
    )

    restructuring_keyword_count = sum(
        article[
            "restructuring_keywords"
        ]
        for article in analyzed_articles
    )

    cost_cutting_keyword_count = sum(
        article[
            "cost_cutting_keywords"
        ]
        for article in analyzed_articles
    )

    hiring_freeze_keyword_count = sum(
        article[
            "hiring_freeze_keywords"
        ]
        for article in analyzed_articles
    )

    return {

        "news_count": total,

        "negative_news_ratio":
            negative_ratio,

        "positive_news_ratio":
            positive_ratio,

        "neutral_news_ratio":
            neutral_ratio,

        "avg_sentiment":
            avg_sentiment,

        "negative_sentiment_score":
            negative_sentiment_score,

        "sentiment_volatility":
            sentiment_volatility,

        "layoff_keyword_count":
            layoff_keyword_count,

        "restructuring_keyword_count":
            restructuring_keyword_count,

        "cost_cutting_keyword_count":
            cost_cutting_keyword_count,

        "hiring_freeze_keyword_count":
            hiring_freeze_keyword_count
    }