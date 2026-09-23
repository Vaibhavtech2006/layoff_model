import sys
from pathlib import Path

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent 
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv 
load_dotenv(PROJECT_ROOT / ".env")
# ============================================================
# IMPORTS
# ============================================================

import joblib
import numpy as np
import pandas as pd
import shap

from datetime import datetime

from collectors.yfinance_collector import get_market_features
from collectors.news_collector import get_company_news
from sentiment.sentiment_model import analyze_articles, create_sentiment_features


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "training"
    / "models"
    / "xgboost_v2_calibrated.pkl"
)

EVENTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "training"
    / "layoff_events.csv"
)

TICKER_PATH = (
    PROJECT_ROOT
    / "data"
    / "training"
    / "company_tickers.csv"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading final calibrated model...")

package = joblib.load(MODEL_PATH)

model = package["model"]
imputer = package["imputer"]
calibrator = package["calibrator"]

FEATURES = package["features"]

HIGH_THRESHOLD = package["high_threshold"]
EXTREME_THRESHOLD = package["extreme_threshold"]

print("Model loaded successfully.")


# ============================================================
# LOAD HISTORICAL DATA
# ============================================================

events = pd.read_csv(EVENTS_PATH)
tickers = pd.read_csv(TICKER_PATH)

events["Date"] = pd.to_datetime(
    events["Date"],
    errors="coerce"
)

# Normalize company names
events["company_key"] = (
    events["Company"]
    .astype(str)
    .str.lower()
    .str.strip()
)

tickers["company_key"] = (
    tickers["company"]
    .astype(str)
    .str.lower()
    .str.strip()
)


# ============================================================
# COMPANY MATCHING
# ============================================================

def find_company(company_name):

    key = company_name.lower().strip()

    # Exact event match
    exact = events[
        events["company_key"] == key
    ]

    if len(exact) > 0:
        return exact.iloc[0]

    # Exact ticker mapping match
    exact_ticker = tickers[
        tickers["company_key"] == key
    ]

    if len(exact_ticker) > 0:

        row = exact_ticker.iloc[0]

        matching_events = events[
            events["company_key"] == row["company_key"]
        ]

        if len(matching_events) > 0:
            return matching_events.iloc[0]

    # Partial match
    partial = events[
        events["company_key"].str.contains(
            key,
            regex=False,
            na=False
        )
    ]

    if len(partial) > 0:
        return partial.iloc[0]

    return None


# ============================================================
# GET COMPANY INFO
# ============================================================

def get_company_info(company_name):

    row = find_company(company_name)

    if row is None:
        return {
            "company": company_name,
            "industry": "Unknown",
            "ticker": None
        }

    company = row["Company"]
    industry = row.get("Industry", "Unknown")

    ticker_row = tickers[
        tickers["company_key"]
        == str(company).lower().strip()
    ]

    ticker = None

    if len(ticker_row) > 0:
        ticker = ticker_row.iloc[0]["ticker"]

    return {
        "company": company,
        "industry": industry,
        "ticker": ticker
    }


# ============================================================
# HISTORICAL LAYOFF FEATURES
# ============================================================

def get_layoff_features(company, industry, snapshot_date):

    company_events = events[
        events["company_key"]
        == company.lower().strip()
    ].copy()

    company_events = company_events[
        company_events["Date"] <= snapshot_date
    ]

    industry_events = events[
        events["Industry"].fillna("Unknown")
        == industry
    ].copy()

    industry_events = industry_events[
        industry_events["Date"] <= snapshot_date
    ]

    # --------------------------------------------------------
    # Company history
    # --------------------------------------------------------

    company_previous_layoffs = len(
        company_events
    )

    company_previous_laid_off = pd.to_numeric(
        company_events["Laid_Off_Count"],
        errors="coerce"
    ).fillna(0).sum()

    if len(company_events) > 0:

        last_layoff = company_events["Date"].max()

        days_since_last_layoff = (
            snapshot_date - last_layoff
        ).days

    else:

        days_since_last_layoff = 9999

    # --------------------------------------------------------
    # Rolling company counts
    # --------------------------------------------------------

    def count_recent_days(days):

        cutoff = snapshot_date - pd.Timedelta(
            days=days
        )

        return int(
            (
                company_events["Date"]
                >= cutoff
            ).sum()
        )

    company_layoffs_30d = count_recent_days(30)
    company_layoffs_90d = count_recent_days(90)
    company_layoffs_365d = count_recent_days(365)

    # --------------------------------------------------------
    # Rolling industry counts
    # --------------------------------------------------------

    def industry_recent_days(days):

        cutoff = snapshot_date - pd.Timedelta(
            days=days
        )

        return int(
            (
                industry_events["Date"]
                >= cutoff
            ).sum()
        )

    industry_layoffs_30d = industry_recent_days(30)
    industry_layoffs_90d = industry_recent_days(90)
    industry_layoffs_365d = industry_recent_days(365)

    return {

        "company_previous_layoffs":
            company_previous_layoffs,

        "company_previous_laid_off":
            company_previous_laid_off,

        "days_since_last_layoff":
            days_since_last_layoff,

        "company_layoffs_30d":
            company_layoffs_30d,

        "company_layoffs_90d":
            company_layoffs_90d,

        "company_layoffs_365d":
            company_layoffs_365d,

        "industry_layoffs_30d":
            industry_layoffs_30d,

        "industry_layoffs_90d":
            industry_layoffs_90d,

        "industry_layoffs_365d":
            industry_layoffs_365d
    }


# ============================================================
# INDUSTRY FEATURES
# ============================================================

def add_industry_features(features, industry):

    industry_columns = [
        feature
        for feature in FEATURES
        if feature.startswith("industry_")
    ]

    for column in industry_columns:
        features[column] = 0

    column_name = f"industry_{industry}"

    if column_name in features:
        features[column_name] = 1

    return features


# ============================================================
# MARKET FEATURES
# ============================================================

def add_market_features(features, ticker):

    market_defaults = [
        "return_30d",
        "return_90d",
        "volatility_30d",
        "volatility_90d",
        "drawdown_30d",
        "drawdown_90d",
        "volume_change_30d"
    ]

    for feature in market_defaults:
        features[feature] = np.nan

    for feature in market_defaults:
        features[f"{feature}_available"] = 0

    if not ticker:
        return features

    try:

        market = get_market_features(ticker)

        if not market:
            return features

        for feature in market_defaults:

            value = market.get(feature)

            if value is not None:

                features[feature] = value
                features[f"{feature}_available"] = 1

    except Exception as e:

        print(
            f"Market data warning: {e}"
        )

    return features


# ============================================================
# LIVE SENTIMENT
# ============================================================

def add_sentiment_features(
    features,
    company_name
):

    sentiment_names = [
        "news_count",
        "negative_news_ratio",
        "positive_news_ratio",
        "neutral_news_ratio",
        "avg_sentiment",
        "layoff_keyword_count",
        "restructuring_keyword_count",
        "cost_cutting_keyword_count",
        "hiring_freeze_keyword_count"
    ]

    # Default = unavailable
    for feature in sentiment_names:
        features[feature] = np.nan

    features["sentiment_available"] = 0
    features["sentiment_age_days"] = np.nan

    try:

        print("Collecting recent news...")

        articles = get_company_news(
            company_name=company_name,
            days=30
        )

        if not articles:

            print("No recent news found.")
            return features

        analyzed = analyze_articles(
            articles
        )

        sentiment = create_sentiment_features(
            analyzed
        )

        for feature in sentiment_names:

            if feature in sentiment:

                features[feature] = sentiment[
                    feature
                ]

        features["sentiment_available"] = 1

        # Current news is approximately 0 days old
        features["sentiment_age_days"] = 0

        print(
            f"News articles analyzed: {len(articles)}"
        )

    except Exception as e:

        print(
            f"Sentiment warning: {e}"
        )

    return features


# ============================================================
# RISK CATEGORY
# ============================================================

def get_risk(probability):

    if probability >= EXTREME_THRESHOLD:
        return "EXTREME HIGH"

    elif probability >= HIGH_THRESHOLD:
        return "HIGH"

    else:
        return "LOW"


# ============================================================
# SHAP EXPLANATION
# ============================================================

def explain_prediction(
    X_model,
    top_n=5
):

    try:

        explainer = shap.TreeExplainer(
            model
        )

        shap_values = explainer.shap_values(
            X_model
        )

        values = np.array(
            shap_values
        ).reshape(-1)

        feature_values = X_model.iloc[0]

        results = []

        for feature, shap_value in zip(
            FEATURES,
            values
        ):

            results.append({
                "feature": feature,
                "shap": float(shap_value),
                "value": feature_values[feature]
            })

        results.sort(
            key=lambda x: abs(x["shap"]),
            reverse=True
        )

        positive = [
            item for item in results
            if item["shap"] > 0
        ][:top_n]

        negative = [
            item for item in results
            if item["shap"] < 0
        ][:top_n]

        return positive, negative

    except Exception as e:

        print(
            f"SHAP warning: {e}"
        )

        return [], []


# ============================================================
# MAIN PREDICTION
# ============================================================

def predict_company(company_name):

    print("\n" + "=" * 60)
    print("CORPORATE LAYOFF PREDICTOR")
    print("=" * 60)

    print(
        f"\nCompany requested: {company_name}"
    )

    # --------------------------------------------------------
    # Company info
    # --------------------------------------------------------

    info = get_company_info(
        company_name
    )

    company = info["company"]
    industry = info["industry"]
    ticker = info["ticker"]

    print(f"Matched company : {company}")
    print(f"Industry        : {industry}")
    print(f"Ticker          : {ticker}")

    # --------------------------------------------------------
    # Snapshot date
    # --------------------------------------------------------

    snapshot_date = pd.Timestamp(
        datetime.now().date()
    )

    # --------------------------------------------------------
    # Build features
    # --------------------------------------------------------

    features = {}

    features.update(
        get_layoff_features(
            company,
            industry,
            snapshot_date
        )
    )

    features = add_market_features(
        features,
        ticker
    )

    features = add_sentiment_features(
        features,
        company_name
    )

    features = add_industry_features(
        features,
        industry
    )

    # --------------------------------------------------------
    # DataFrame
    # --------------------------------------------------------

    X = pd.DataFrame(
        [features]
    )

    # Ensure exact model columns
    X = X.reindex(
        columns=FEATURES,
        fill_value=0
    )

    # Numeric conversion
    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # --------------------------------------------------------
    # Imputation
    # --------------------------------------------------------

    X_imp = imputer.transform(X)

    X_imp_df = pd.DataFrame(
        X_imp,
        columns=FEATURES
    )

    # --------------------------------------------------------
    # RAW PROBABILITY
    # --------------------------------------------------------

    raw_probability = float(
        model.predict_proba(
            X_imp
        )[0][1]
    )

    # --------------------------------------------------------
    # CALIBRATED PROBABILITY
    # --------------------------------------------------------

    probability = float(
        calibrator.predict(
            [raw_probability]
        )[0]
    )

    risk = get_risk(
        probability
    )

    # --------------------------------------------------------
    # SHAP
    # --------------------------------------------------------

    positive, negative = explain_prediction(
        X_imp_df
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    print("\n" + "=" * 60)
    print("FINAL PREDICTION")
    print("=" * 60)

    print(f"\nCompany       : {company}")
    print(f"Industry      : {industry}")
    print(f"Layoff Risk   : {risk}")
    print(
        f"Probability   : {probability * 100:.2f}%"
    )

    print("\nMain Factors Increasing Risk:")

    if positive:

        for item in positive:
            print(
                f"  + {item['feature']}: "
                f"{item['shap']:.4f}"
            )

    else:
        print("  None available")

    print("\nFactors Reducing Risk:")

    if negative:

        for item in negative:
            print(
                f"  - {item['feature']}: "
                f"{item['shap']:.4f}"
            )

    else:
        print("  None available")

    print("\n" + "=" * 60)

    return {
        "company": company,
        "industry": industry,
        "ticker": ticker,
        "risk": risk,
        "probability": probability,
        "raw_probability": raw_probability,
        "positive_factors": positive,
        "negative_factors": negative
    }


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    company_name = input(
        "\nEnter company name: "
    ).strip()

    if not company_name:

        print(
            "Please enter a company name."
        )

    else:

        predict_company(
            company_name
        )

