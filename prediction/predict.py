import warnings
warnings.filterwarnings("ignore")
import os
import sys
import requests
import yfinance as yf
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import joblib
import numpy as np
import pandas as pd
import shap

from collectors.yfinance_collector import get_market_features
from collectors.news_collector import get_company_news
from sentiment.sentiment_model import analyze_articles, create_sentiment_features

MODEL_PATH = PROJECT_ROOT / "data" / "training" / "models" / "xgboost_v2_calibrated.pkl"
EVENTS_PATH = PROJECT_ROOT / "data" / "training" / "layoff_events.csv"

print("\nLoading final calibrated model...")
package = joblib.load(MODEL_PATH)

model = package["model"]
imputer = package["imputer"]
calibrator = package["calibrator"]
FEATURES = package["features"]
HIGH_THRESHOLD = package["high_threshold"]
EXTREME_THRESHOLD = package["extreme_threshold"]
print("Model loaded successfully.")

events = pd.read_csv(EVENTS_PATH) if EVENTS_PATH.exists() else pd.DataFrame()
if not events.empty:
    events["Date"] = pd.to_datetime(events["Date"], errors="coerce")
    events["company_key"] = events["Company"].astype(str).str.lower().str.strip()

SECTOR_TO_MODEL_INDUSTRY = {
    "Technology": "Data",
    "Communication Services": "Media",
    "Consumer Cyclical": "Retail",
    "Consumer Defensive": "Food",
    "Financial Services": "Finance",
    "Healthcare": "Healthcare",
    "Industrials": "Manufacturing",
    "Energy": "Energy",
    "Real Estate": "Real Estate",
    "Basic Materials": "Manufacturing",
    "Utilities": "Energy"
}

# Technical/metadata columns that should NOT appear in user-facing SHAP explanations
METADATA_COLS = {
    "sentiment_age_days", "sentiment_available",
    "return_30d_available", "return_90d_available",
    "volatility_30d_available", "volatility_90d_available",
    "drawdown_30d_available", "drawdown_90d_available",
    "volume_change_30d_available"
}

def resolve_global_company(query_name):
    url = "https://query2.finance.yahoo.com/v1/finance/search"
    params = {"q": query_name, "quotesCount": 5, "newsCount": 0}
    headers = {"User-Agent": "Mozilla/5.0"}

    ticker_symbol = None
    official_name = query_name

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        if resp.status_code == 200:
            quotes = resp.json().get("quotes", [])
            equities = [q for q in quotes if q.get("quoteType") == "EQUITY"]
            if equities:
                best = equities[0]
                ticker_symbol = best.get("symbol")
                official_name = best.get("shortname") or best.get("longname") or query_name
    except Exception as e:
        print(f"Yahoo Search warning: {e}")

    industry = "Unknown"
    fundamentals = {}

    if ticker_symbol:
        try:
            tk = yf.Ticker(ticker_symbol)
            info = tk.info or {}
            sector = info.get("sector", "")
            raw_ind = info.get("industry", "")
            official_name = info.get("shortName") or official_name
            industry = SECTOR_TO_MODEL_INDUSTRY.get(sector, raw_ind or "Other")

            fundamentals = {
                "market_cap": info.get("marketCap"),
                "revenue_growth": info.get("revenueGrowth"),
                "profit_margins": info.get("profitMargins"),
                "debt_to_equity": info.get("debtToEquity"),
                "full_time_employees": info.get("fullTimeEmployees"),
                "sector": sector,
                "raw_industry": raw_ind
            }
        except Exception as e:
            print(f"yfinance info warning: {e}")

    return {
        "company": official_name,
        "search_name": query_name,
        "industry": industry,
        "ticker": ticker_symbol,
        "fundamentals": fundamentals
    }

def get_layoff_features(company, search_name, industry, snapshot_date, sentiment_features):
    if not events.empty:
        company_events = events[
            (events["company_key"] == company.lower().strip()) |
            (events["company_key"] == search_name.lower().strip())
        ].copy()
        company_events = company_events[company_events["Date"] <= snapshot_date]

        industry_events = events[
            events["Industry"].fillna("Unknown") == industry
        ].copy()
        industry_events = industry_events[industry_events["Date"] <= snapshot_date]
    else:
        company_events = pd.DataFrame()
        industry_events = pd.DataFrame()

    company_previous_layoffs = len(company_events)
    company_previous_laid_off = (
        pd.to_numeric(company_events["Laid_Off_Count"], errors="coerce").fillna(0).sum()
        if not company_events.empty else 0
    )

    if not company_events.empty:
        last_layoff = company_events["Date"].max()
        days_since_last_layoff = (snapshot_date - last_layoff).days
    else:
        days_since_last_layoff = 9999

    # Live news override: If current news strongly mentions layoffs/cuts, update recency
    layoff_kw = sentiment_features.get("layoff_keyword_count", 0) or 0
    restruct_kw = sentiment_features.get("restructuring_keyword_count", 0) or 0
    if layoff_kw >= 2 or (layoff_kw >= 1 and restruct_kw >= 2):
        days_since_last_layoff = min(days_since_last_layoff, 30)
        company_previous_layoffs = max(company_previous_layoffs, 1)

    def count_recent(df, days):
        if df.empty:
            return 0
        cutoff = snapshot_date - pd.Timedelta(days=days)
        return int((df["Date"] >= cutoff).sum())

    c_30d = count_recent(company_events, 30) + (1 if layoff_kw >= 3 else 0)
    c_90d = count_recent(company_events, 90) + (1 if layoff_kw >= 2 else 0)
    c_365d = count_recent(company_events, 365) + (1 if layoff_kw >= 1 else 0)

    return {
        "company_previous_layoffs": company_previous_layoffs,
        "company_previous_laid_off": company_previous_laid_off,
        "days_since_last_layoff": days_since_last_layoff,
        "company_layoffs_30d": c_30d,
        "company_layoffs_90d": c_90d,
        "company_layoffs_365d": c_365d,
        "industry_layoffs_30d": count_recent(industry_events, 30),
        "industry_layoffs_90d": count_recent(industry_events, 90),
        "industry_layoffs_365d": count_recent(industry_events, 365),
    }

def add_industry_features(features, industry):
    for col in [f for f in FEATURES if f.startswith("industry_")]:
        features[col] = 0
    col_name = f"industry_{industry}"
    if col_name in features:
        features[col_name] = 1
    return features

def add_market_features(features, ticker):
    market_defaults = [
        "return_30d", "return_90d", "volatility_30d", "volatility_90d",
        "drawdown_30d", "drawdown_90d", "volume_change_30d"
    ]
    for f in market_defaults:
        features[f] = np.nan
        features[f"{f}_available"] = 0

    if not ticker:
        return features

    try:
        market = get_market_features(ticker)
        if market:
            for f in market_defaults:
                val = market.get(f)
                if val is not None:
                    features[f] = val
                    features[f"{f}_available"] = 1
    except Exception as e:
        print(f"Market data warning: {e}")

    return features

def add_sentiment_features(features, company_name):
    sentiment_names = [
        "news_count", "negative_news_ratio", "positive_news_ratio",
        "neutral_news_ratio", "avg_sentiment", "layoff_keyword_count",
        "restructuring_keyword_count", "cost_cutting_keyword_count",
        "hiring_freeze_keyword_count"
    ]
    for f in sentiment_names:
        features[f] = np.nan

    features["sentiment_available"] = 0
    # Keep sentiment_age_days as NaN so SimpleImputer uses training median (prevents data leakage!)
    features["sentiment_age_days"] = np.nan
    articles = []

    try:
        articles = get_company_news(company_name=company_name, days=30)
        if not articles:
            return features, []

        analyzed = analyze_articles(articles)
        sentiment = create_sentiment_features(analyzed)

        for f in sentiment_names:
            if f in sentiment:
                features[f] = sentiment[f]

        features["sentiment_available"] = 1
    except Exception as e:
        print(f"Sentiment warning: {e}")

    return features, articles

def explain_prediction(X_model, top_n=5):
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_model)
        values = np.array(shap_values).reshape(-1)
        feature_values = X_model.iloc[0]

        results = []
        for feature, shap_value in zip(FEATURES, values):
            val = float(feature_values[feature])

            # 1. Hide internal metadata columns
            if feature in METADATA_COLS:
                continue

            # 2. Hide inactive one-hot industry columns
            if feature.startswith("industry_") and val == 0:
                continue

            # 3. Prevent 0-value count features from showing up as "increasing risk"
            if shap_value > 0 and val == 0 and (
                "keyword_count" in feature or "layoffs_" in feature
            ):
                continue

            results.append({
                "feature": feature,
                "shap": float(shap_value),
                "value": val
            })

        results.sort(key=lambda x: abs(x["shap"]), reverse=True)
        positive = [item for item in results if item["shap"] > 0][:top_n]
        negative = [item for item in results if item["shap"] < 0][:top_n]
        return positive, negative
    except Exception as e:
        print(f"SHAP warning: {e}")
        return [], []

def get_risk(probability):
    if probability >= EXTREME_THRESHOLD:
        return "EXTREME HIGH"
    elif probability >= HIGH_THRESHOLD:
        return "HIGH"
    return "LOW"

def generate_grounded_llm_report(company_info, features, risk_label, risk_score, articles):
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        return "Tip: Add GROQ_API_KEY in .env to enable AI Financial & News Verification."

    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import SystemMessage, HumanMessage

        # Lock in the exact model that worked on your Groq account
        active_model = "openai/gpt-oss-20b"

        llm = ChatGroq(model=active_model, temperature=0.0, api_key=groq_key)
        headlines = [f"- {a['title']} ({a['source']})" for a in articles[:12]]

        prompt = f"""
        Analyze the corporate layoff risk for {company_info['company']} (Ticker: {company_info['ticker']}).
        STRICT RULE: Do NOT hallucinate. Rely ONLY on the verified live metrics below.

        1. LIVE FINANCIAL FUNDAMENTALS (Yahoo Finance):
           - Sector/Industry: {company_info['fundamentals'].get('sector')} / {company_info['fundamentals'].get('raw_industry')}
           - Revenue Growth (YoY): {company_info['fundamentals'].get('revenue_growth')}
           - Profit Margins: {company_info['fundamentals'].get('profit_margins')}
           - Debt to Equity: {company_info['fundamentals'].get('debt_to_equity')}
           - 30d Return: {features.get('return_30d')} | 90d Return: {features.get('return_90d')}

        2. FINBERT NLP SIGNALS (Last 30 Days News):
           - Negative News Ratio: {features.get('negative_news_ratio')}
           - Layoff Keywords: {features.get('layoff_keyword_count')} | Restructuring Keywords: {features.get('restructuring_keyword_count')}
           - Cost Cutting Keywords: {features.get('cost_cutting_keyword_count')}

        3. MODEL VERDICT: {risk_label} RISK (Risk Index: {risk_score:.1f}/100)

        4. RECENT HEADLINES:
        {chr(10).join(headlines) if headlines else "No recent headlines."}

        Give a crisp 4-bullet executive breakdown covering: (1) Financial Health, (2) News & Layoff Signals, (3) Stock Momentum, and (4) Final Verdict.
        """
        resp = llm.invoke([
            SystemMessage(content="You are a quantitative financial risk analyst. Never hallucinate facts."),
            HumanMessage(content=prompt)
        ])
        return f"[Model: {active_model}]\n" + resp.content
    except Exception as e:
        return f"LLM Report skipped: {e}"

def predict_company(company_name):
    print("\n" + "=" * 60)
    print("GLOBAL CORPORATE LAYOFF PREDICTOR")
    print("=" * 60)

    info = resolve_global_company(company_name)
    company = info["company"]
    industry = info["industry"]
    ticker = info["ticker"]

    print(f"Requested       : {company_name}")
    print(f"Resolved Company: {company}")
    print(f"Mapped Industry : {industry} ({info['fundamentals'].get('sector', 'N/A')})")
    print(f"Global Ticker   : {ticker}")

    snapshot_date = pd.Timestamp(datetime.now().date())
    features = {}

    features, articles = add_sentiment_features(features, company_name)
    features.update(get_layoff_features(company, company_name, industry, snapshot_date, features))
    features = add_market_features(features, ticker)
    features = add_industry_features(features, industry)

    X = pd.DataFrame([features]).reindex(columns=FEATURES, fill_value=np.nan)
    # Fill industry one-hot columns with 0 where NaN
    for col in [f for f in FEATURES if f.startswith("industry_")]:
        X[col] = X[col].fillna(0)

    X = X.apply(pd.to_numeric, errors="coerce")
    X_imp = imputer.transform(X)
    X_imp_df = pd.DataFrame(X_imp, columns=FEATURES)

    raw_probability = float(model.predict_proba(X_imp)[0][1])
    probability = float(calibrator.predict([raw_probability])[0])

    # Fundamental & Live News Distress Adjustment
    rev_growth = info["fundamentals"].get("revenue_growth")
    profit_margin = info["fundamentals"].get("profit_margins")
    neg_news = features.get("negative_news_ratio") or 0
    layoff_kw = features.get("layoff_keyword_count") or 0

    if rev_growth is not None and rev_growth < -0.05 and profit_margin is not None and profit_margin < 0:
        probability = min(0.95, probability + 0.08)
    if layoff_kw >= 3 or neg_news >= 0.45:
        probability = min(0.95, probability + 0.06)

    risk = get_risk(probability)

    # Human-friendly 0-100 Risk Score scaled relative to calibrated thresholds
    if probability < HIGH_THRESHOLD:
        risk_score = (probability / max(HIGH_THRESHOLD, 0.01)) * 49.0
    elif probability < EXTREME_THRESHOLD:
        span = max(EXTREME_THRESHOLD - HIGH_THRESHOLD, 0.01)
        risk_score = 50.0 + ((probability - HIGH_THRESHOLD) / span) * 29.0
    else:
        risk_score = min(99.0, 80.0 + ((probability - EXTREME_THRESHOLD) / max(1.0 - EXTREME_THRESHOLD, 0.01)) * 20.0)

    positive, negative = explain_prediction(X_imp_df)
    llm_summary = generate_grounded_llm_report(info, features, risk, risk_score, articles)

    print("\n" + "=" * 60)
    print("FINAL PREDICTION")
    print("=" * 60)
    print(f"Company            : {company} ({ticker or 'Private/Unlisted'})")
    print(f"Industry           : {industry}")
    print(f"Layoff Risk        : {risk}")
    print(f"Risk Index Score   : {risk_score:.1f} / 100")
    print(f"Calibrated ML Prob : {probability * 100:.2f}% (Raw XGBoost: {raw_probability * 100:.2f}%)")

    print("\nMain Factors Increasing Risk (SHAP):")
    for item in positive:
        print(f"  + {item['feature']}: {item['shap']:.4f} (val: {item['value']:.2f})")

    print("\nFactors Reducing Risk (SHAP):")
    for item in negative:
        print(f"  - {item['feature']}: {item['shap']:.4f} (val: {item['value']:.2f})")

    print("\nGrounded AI Financial & News Verification:")
    print(llm_summary)
    print("\n" + "=" * 60)

    return {
        "company": company,
        "industry": industry,
        "ticker": ticker,
        "risk": risk,
        "risk_score": round(risk_score, 1),
        "probability": probability,
        "raw_probability": raw_probability,
        "fundamentals": info["fundamentals"],
        "positive_factors": positive,
        "negative_factors": negative,
        "llm_summary": llm_summary
    }

if __name__ == "__main__":
    company_name = input("\nEnter ANY global company name: ").strip()
    if company_name:
        predict_company(company_name)

     

     