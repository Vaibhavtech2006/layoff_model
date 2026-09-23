import json

from dotenv import load_dotenv

from collectors.news_collector import (
    get_company_news
)


load_dotenv()


articles = get_company_news(
    company_name="Microsoft",
    days=30
)


print(
    "\n========== NEWS RESULTS ==========\n"
)


print(
    json.dumps(
        articles[:5],
        indent=4,
        ensure_ascii=False
    )
)