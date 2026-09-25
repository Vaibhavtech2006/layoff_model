import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from prediction.predict import (
    predict_company,
    FEATURES,
    HIGH_THRESHOLD,
    EXTREME_THRESHOLD,
)

app = FastAPI(title="LayoffRadar AI Live Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).resolve().parent
EVENTS_PATH = PROJECT_ROOT / "data" / "training" / "layoff_events.csv"
FEAT_IMP_PATH = (
    PROJECT_ROOT
    / "data"
    / "training"
    / "models"
    / "xgboost_v2_feature_importance.csv"
)

WATCHLIST_TICKERS = [
    {"name": "Meta Platforms", "ticker": "META", "sector": "Technology"},
    {"name": "Intel Corp", "ticker": "INTC", "sector": "Semiconductors"},
    {"name": "Microsoft", "ticker": "MSFT", "sector": "Software"},
    {"name": "Infosys Ltd", "ticker": "INFY", "sector": "IT Services"},
    {"name": "NVIDIA Corp", "ticker": "NVDA", "sector": "Semiconductors"},
    {"name": "Tesla Inc", "ticker": "TSLA", "sector": "Automotive"},
    {"name": "Boeing Co", "ticker": "BA", "sector": "Aerospace"},
    {"name": "Amazon.com", "ticker": "AMZN", "sector": "Cloud / Retail"},
]


@app.get("/api/stock-history")
def get_stock_history(
    ticker: str = Query(..., min_length=1),
    range_key: str = Query("1Y")
):
    """
    Direct Yahoo Finance v8 Chart API — 100% reliable & instant.
    """
    range_map = {
        "1D": ("5d", "15m"),
        "5D": ("5d", "1h"),
        "1M": ("1mo", "1d"),
        "6M": ("6mo", "1d"),
        "1Y": ("1y", "1d"),
        "5Y": ("5y", "1wk"),
    }
    yf_range, yf_interval = range_map.get(range_key, ("1y", "1d"))

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(ticker)}"
    params = {"range": yf_range, "interval": yf_interval}
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        data = resp.json()
        result = data.get("chart", {}).get("result", [])
        if not result:
            return {"points": []}

        chart_res = result[0]
        timestamps = chart_res.get("timestamp", [])
        closes = (
            chart_res.get("indicators", {})
            .get("quote", [{}])[0]
            .get("close", [])
        )

        points = []
        for ts, close_val in zip(timestamps, closes):
            if close_val is not None:
                dt_str = datetime.fromtimestamp(ts).strftime("%b %d, %Y")
                points.append({
                    "date": dt_str,
                    "price": round(float(close_val), 2),
                })

        if len(points) < 2:
            return {"points": []}

        first_price = points[0]["price"]
        last_price = points[-1]["price"]
        abs_change = round(last_price - first_price, 2)
        pct_change = round(((last_price - first_price) / first_price) * 100, 2)

        return {
            "ticker": ticker.upper(),
            "range": range_key,
            "current_price": last_price,
            "abs_change": abs_change,
            "pct_change": pct_change,
            "points": points,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def fetch_live_global_layoff_alerts(limit=8):
    query = '("layoffs" OR "job cuts" OR "workforce reduction") when:7d'
    url = f"https://news.google.com/rss/search?q={urllib.parse.quote(query)}&hl=en-US&gl=US&ceid=US:en"
    headers = {"User-Agent": "Mozilla/5.0"}
    alerts = []
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            root = ET.fromstring(resp.content)
            for item in root.findall(".//item")[:limit]:
                source_el = item.find("source")
                alerts.append({
                    "title": item.findtext("title") or "",
                    "source": source_el.text if source_el is not None else "Global Wire",
                    "published_at": (item.findtext("pubDate") or "")[:22],
                    "url": item.findtext("link") or "",
                })
    except Exception:
        pass
    return alerts


def fetch_live_market_watchlist():
    market_list = []
    headers = {"User-Agent": "Mozilla/5.0"}
    for item in WATCHLIST_TICKERS:
        sym = item["ticker"]
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?range=3mo&interval=1d"
            r = requests.get(url, headers=headers, timeout=6)
            res = r.json().get("chart", {}).get("result", [])
            if not res:
                continue
            closes = [
                float(c)
                for c in res[0].get("indicators", {}).get("quote", [{}])[0].get("close", [])
                if c is not None
            ]
            if len(closes) < 5:
                continue
            latest = closes[-1]
            c_30d = closes[max(0, len(closes) - 22)]
            ret_30d = ((latest - c_30d) / c_30d) * 100.0
            peak = max(closes)
            drawdown = ((latest - peak) / peak) * 100.0
            s = pd.Series(closes).pct_change().dropna()
            vol = float(s.std() * np.sqrt(252) * 100.0)

            market_list.append({
                "name": item["name"],
                "ticker": sym,
                "sector": item["sector"],
                "price": round(latest, 2),
                "return_30d": round(ret_30d, 2),
                "drawdown_90d": round(drawdown, 2),
                "volatility": round(vol, 1),
                "distress_signal": "ELEVATED" if (ret_30d < -8 or drawdown < -18) else "STABLE",
            })
        except Exception:
            continue
    return market_list


@app.get("/")
def health_check():
    return {"status": "online", "service": "LayoffRadar AI Live Backend"}


@app.get("/api/dashboard")
def get_live_dashboard_data():
    total_events = 0
    total_laid_off = 0
    unique_companies = 0
    top_industries = []
    recent_recorded_events = []

    if EVENTS_PATH.exists():
        df = pd.read_csv(EVENTS_PATH)
        total_events = int(len(df))
        unique_companies = int(df["Company"].nunique())
        df["Laid_Off_Count_Num"] = pd.to_numeric(df.get("Laid_Off_Count"), errors="coerce").fillna(0)
        total_laid_off = int(df["Laid_Off_Count_Num"].sum())

        ind_grp = (
            df.groupby("Industry")["Laid_Off_Count_Num"]
            .agg(events="count", headcount="sum")
            .reset_index()
            .sort_values("headcount", ascending=False)
            .head(6)
        )
        for _, row in ind_grp.iterrows():
            top_industries.append({
                "industry": str(row["Industry"]),
                "events": int(row["events"]),
                "laid_off": int(row["headcount"]),
            })

        df["Date_Parsed"] = pd.to_datetime(df["Date"], errors="coerce")
        for _, row in df.sort_values("Date_Parsed", ascending=False).head(6).iterrows():
            recent_recorded_events.append({
                "company": str(row.get("Company", "Unknown")),
                "industry": str(row.get("Industry", "Other")),
                "laid_off": int(row.get("Laid_Off_Count_Num", 0)),
                "date": str(row.get("Date", ""))[:10],
            })

    top_features = []
    if FEAT_IMP_PATH.exists():
        f_df = pd.read_csv(FEAT_IMP_PATH).head(6)
        cols = f_df.columns.tolist()
        for _, row in f_df.iterrows():
            top_features.append({
                "feature": str(row[cols[0]]),
                "importance": round(float(row[cols[1] if len(cols) > 1 else cols[0]]), 4),
            })

    return {
        "status": "success",
        "kpis": {
            "total_verified_events": total_events,
            "total_employees_impacted": total_laid_off,
            "tracked_corporations": unique_companies,
            "active_ml_features": len(FEATURES),
            "high_risk_threshold_pct": round(HIGH_THRESHOLD * 100, 2),
            "extreme_risk_threshold_pct": round(EXTREME_THRESHOLD * 100, 2),
        },
        "top_industries": top_industries,
        "recent_recorded_events": recent_recorded_events,
        "top_model_features": top_features,
        "live_market_watchlist": fetch_live_market_watchlist(),
        "live_global_alerts": fetch_live_global_layoff_alerts(limit=8),
    }


@app.get("/api/search")
def search_companies(q: str = Query(..., min_length=1)):
    url = "https://query2.finance.yahoo.com/v1/finance/search"
    params = {"q": q, "quotesCount": 6, "newsCount": 0}
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=8)
        quotes = resp.json().get("quotes", [])
        return {
            "results": [
                {
                    "name": item.get("shortname") or item.get("longname") or item.get("symbol"),
                    "ticker": item.get("symbol"),
                    "exchange": item.get("exchDisp", ""),
                }
                for item in quotes
                if item.get("quoteType") == "EQUITY"
            ]
        }
    except Exception:
        return {"results": []}


@app.get("/api/predict")
def get_prediction(company: str = Query(..., min_length=1)):
    try:
        result = predict_company(company)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    print("\nStarting FastAPI server on http://127.0.0.1:8000 ...")
    uvicorn.run(app, host="127.0.0.1", port=8000)