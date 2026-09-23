import os
import json
import re
from datetime import datetime

from apify_client import ApifyClient


ACTOR_ID = "data-slayer/linkedin-company-scraper"


def run_apify_actor(company_url):
    """
    Run Apify LinkedIn Company Scraper and return raw data.
    """

    token = os.getenv("APIFY_API_TOKEN")

    if not token:
        raise ValueError("APIFY_API_TOKEN not found in .env")

    client = ApifyClient(token)

    run_input = {
        "linkedin_urls": [company_url]
    }

    print("Starting Apify Actor...")

    run = client.actor(ACTOR_ID).call(
        run_input=run_input
    )

    if run is None:
        raise RuntimeError("Apify Actor run failed.")

    print("Apify Actor completed.")
    print(f"Run ID: {run.id}")

    dataset_id = run.default_dataset_id

    dataset = client.dataset(dataset_id)

    items = dataset.list_items().items

    return items


def save_raw_data(data, company_name="company"):
    """
    Save raw Apify response for debugging/reproducibility.
    """

    os.makedirs("data/raw", exist_ok=True)

    safe_name = re.sub(
        r"[^a-zA-Z0-9_-]",
        "_",
        company_name.lower()
    )

    file_path = f"data/raw/apify_{safe_name}.json"

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
            default=str
        )

    print(f"Raw data saved to: {file_path}")


def extract_company_features(raw_data):
    """
    Convert raw Apify data into clean ML features.
    """

    if not raw_data:
        raise ValueError("No data received from Apify.")

    company = raw_data[0]

    # -----------------------------
    # Basic company information
    # -----------------------------

    company_name = company.get("company_name")

    industry = company.get("industry")

    company_type = company.get("company_type")

    employee_count = company.get("employee_count")

    follower_count = company.get("follower_count")

    founded_year = company.get("founded_year")

    # -----------------------------
    # Location information
    # -----------------------------

    locations = company.get("locations") or []

    location_count = len(locations)

    # -----------------------------
    # Affiliated pages
    # -----------------------------

    affiliated_pages = company.get("affiliated_pages") or []

    affiliated_page_count = len(affiliated_pages)

    # -----------------------------
    # Recent LinkedIn activity
    # -----------------------------

    recent_posts = company.get("recent_posts") or []

    recent_post_count = len(recent_posts)

    total_reactions = 0
    total_comments = 0

    for post in recent_posts:

        reactions = post.get("reaction_count") or 0
        comments = post.get("comment_count") or 0

        total_reactions += reactions
        total_comments += comments

    if recent_post_count > 0:

        avg_post_reactions = (
            total_reactions / recent_post_count
        )

        avg_post_comments = (
            total_comments / recent_post_count
        )

        post_engagement = (
            total_reactions + total_comments
        ) / recent_post_count

    else:

        avg_post_reactions = 0
        avg_post_comments = 0
        post_engagement = 0

    # -----------------------------
    # Hiring signal
    # -----------------------------

    jobs = company.get("jobs") or []

    # IMPORTANT:
    # Do NOT treat jobs.open_jobs_count as
    # actual company vacancies.
    #
    # The Actor output may contain LinkedIn
    # search-result counts rather than actual
    # Microsoft job openings.

    if jobs:
        hiring_signal_available = 1
    else:
        hiring_signal_available = 0

    # -----------------------------
    # Final feature dictionary
    # -----------------------------

    features = {

        # Company
        "company_name": company_name,
        "industry": industry,
        "company_type": company_type,

        # Workforce
        "employee_count": employee_count,
        "follower_count": follower_count,

        # Company age
        "founded_year": founded_year,

        # Geographic footprint
        "location_count": location_count,

        # Company ecosystem
        "affiliated_page_count": affiliated_page_count,

        # LinkedIn activity
        "recent_post_count": recent_post_count,
        "avg_post_reactions": avg_post_reactions,
        "avg_post_comments": avg_post_comments,
        "post_engagement": post_engagement,

        # Hiring
        "hiring_signal_available": hiring_signal_available,

        # Metadata
        "collected_at": datetime.utcnow().isoformat()
    }

    return features


def get_company_data(company_url):
    """
    Main function:
    Run Apify → save raw data → return clean features.
    """

    raw_data = run_apify_actor(company_url)

    company_name = "company"

    if raw_data:
        company_name = raw_data[0].get(
            "company_name",
            "company"
        )

    # Save raw response
    save_raw_data(
        raw_data,
        company_name
    )

    # Extract ML features
    features = extract_company_features(
        raw_data
    )

    return features