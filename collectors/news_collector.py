import os
import json
import re
import urllib.parse
import requests
import xml.etree.ElementTree as ET
from datetime import datetime


def get_news(company_name, days=30, max_results=40):
    """
    100% Free Global News Collector using Google News RSS.
    Works for any company in the world without any API key!
    """
    print(f"Collecting live global news for {company_name} (last {days}d)...")

    # Query specifically targets company + business/workforce context
    # Line 19 in collectors/news_collector.py:
    query = f'"{company_name}" (stock OR earnings OR layoffs OR workforce OR restructuring OR revenue) when:{days}d'
    encoded_query = urllib.parse.quote(query)
    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    response = requests.get(rss_url, headers=headers, timeout=20)
    if response.status_code != 200:
        raise RuntimeError(f"Google News RSS error: {response.status_code}")

    root = ET.fromstring(response.content)
    articles = []

    for item in root.findall(".//item")[:max_results]:
        title = item.findtext("title") or ""
        description = item.findtext("description") or ""
        # Strip HTML tags from RSS description
        clean_desc = re.sub(r"<[^>]+>", " ", description).strip()
        pub_date = item.findtext("pubDate") or ""
        link = item.findtext("link") or ""
        source_elem = item.find("source")
        source_name = source_elem.text if source_elem is not None else "Google News"

        articles.append({
            "source": {"name": source_name},
            "title": title,
            "description": clean_desc,
            "content": clean_desc,
            "publishedAt": pub_date,
            "url": link
        })

    print(f"Articles collected: {len(articles)}")
    return articles


def clean_articles(articles, company_name):
    """Prepare articles for FinBERT sentiment analysis."""
    cleaned = []
    for article in articles:
        title = article.get("title") or ""
        description = article.get("description") or ""
        text = f"{title}. {description}".strip()

        if not text:
            continue

        cleaned.append({
            "source": article.get("source", {}).get("name"),
            "title": title,
            "description": description,
            "content": article.get("content"),
            "text": text,
            "published_at": article.get("publishedAt"),
            "url": article.get("url")
        })

    print(f"Relevant articles ready for FinBERT: {len(cleaned)}")
    return cleaned


def save_raw_news(articles, company_name):
    os.makedirs("data/raw", exist_ok=True)
    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", company_name.lower())
    file_path = f"data/raw/news_{safe_name}.json"
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(articles, file, indent=2, ensure_ascii=False)


def get_company_news(company_name, days=30, to_date=None):
    """Complete pipeline called by predict.py"""
    articles = get_news(company_name=company_name, days=days)
    save_raw_news(articles, company_name)
    return clean_articles(articles, company_name)