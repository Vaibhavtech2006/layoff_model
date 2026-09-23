import os
import json
import re

import requests

from datetime import datetime, timedelta


NEWS_API_URL = "https://newsapi.org/v2/everything"


def get_news(
    company_name,
    days=30,
    to_date=None,
    page_size=100
):
    """Collect company-related news from NewsAPI."""

    api_key = os.getenv("NEWS_API_KEY")

    if not api_key:
        raise ValueError(
            "NEWS_API_KEY not found in .env"
        )

    if to_date is None:
        end_date = datetime.utcnow()
    else:
        end_date = datetime.strptime(
            to_date,
            "%Y-%m-%d"
        )

    start_date = end_date - timedelta(days=days)

    params = {
        "q": f'"{company_name}"',
        "from": start_date.strftime("%Y-%m-%d"),
        "to": end_date.strftime("%Y-%m-%d"),
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "page": 1
    }

    headers = {
        "X-Api-Key": api_key
    }

    print(
        f"Collecting news for {company_name}..."
    )

    response = requests.get(
        NEWS_API_URL,
        params=params,
        headers=headers,
        timeout=30
    )

    if response.status_code != 200:

        try:
            error_data = response.json()
            message = error_data.get(
                "message",
                "Unknown NewsAPI error"
            )
        except Exception:
            message = response.text

        raise RuntimeError(
            f"NewsAPI error "
            f"{response.status_code}: {message}"
        )

    data = response.json()

    if data.get("status") != "ok":
        raise RuntimeError(
            data.get(
                "message",
                "NewsAPI request failed."
            )
        )

    articles = data.get(
        "articles",
        []
    )

    print(
        f"Articles collected: {len(articles)}"
    )

    return articles


def is_relevant_article(
    article,
    company_name
):
    """
    Check whether the company is actually
    relevant to the article.
    """

    title = article.get("title") or ""
    description = article.get("description") or ""

    text = (
        title + " " + description
    ).lower()

    company = company_name.lower().strip()

    # Direct company-name match
    if company in text:
        return True

    # Individual words for multi-word companies
    words = [
        word
        for word in re.findall(
            r"[a-zA-Z0-9]+",
            company
        )
        if len(word) > 2
    ]

    if not words:
        return False

    matches = sum(
        word in text
        for word in words
    )

    # Require at least half the company
    # name's meaningful words.
    return matches >= max(
        1,
        len(words) // 2
    )


def clean_articles(
    articles,
    company_name
):
    """
    Filter irrelevant articles and keep
    fields needed for NLP.
    """

    cleaned = []

    for article in articles:

        if not is_relevant_article(
            article,
            company_name
        ):
            continue

        title = article.get(
            "title"
        )

        description = article.get(
            "description"
        )

        content = article.get(
            "content"
        )

        # Prefer title + description.
        # NewsAPI content may be truncated.
        text_parts = [
            title,
            description
        ]

        text_parts = [
            text.strip()
            for text in text_parts
            if text
        ]

        text = " ".join(
            text_parts
        )

        cleaned_article = {

            "source": (
                article
                .get("source", {})
                .get("name")
            ),

            "title": title,

            "description": description,

            "content": content,

            "text": text,

            "published_at": article.get(
                "publishedAt"
            ),

            "url": article.get(
                "url"
            )
        }

        cleaned.append(
            cleaned_article
        )

    print(
        f"Relevant articles: "
        f"{len(cleaned)}"
    )

    return cleaned


def save_raw_news(
    articles,
    company_name
):
    """Save raw NewsAPI response."""

    os.makedirs(
        "data/raw",
        exist_ok=True
    )

    safe_name = re.sub(
        r"[^a-zA-Z0-9_-]",
        "_",
        company_name.lower()
    )

    file_path = (
        f"data/raw/news_{safe_name}.json"
    )

    with open(
        file_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            articles,
            file,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"Raw news saved to: {file_path}"
    )


def get_company_news(
    company_name,
    days=30,
    to_date=None
):
    """Complete NewsAPI collection pipeline."""

    articles = get_news(
        company_name=company_name,
        days=days,
        to_date=to_date
    )

    save_raw_news(
        articles,
        company_name
    )

    cleaned_articles = clean_articles(
        articles,
        company_name
    )

    return cleaned_articles