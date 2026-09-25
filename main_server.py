import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from prediction.predict import predict_company

app = FastAPI(title="LayoffRadar AI API")

# Allow React Frontend (localhost:3000 or localhost:5173) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health_check():
    return {"status": "online", "service": "LayoffRadar AI Backend"}


@app.get("/api/search")
def search_companies(q: str = Query(..., min_length=1)):
    """
    Live global stock autocomplete for React search bar.
    """
    url = "https://query2.finance.yahoo.com/v1/finance/search"
    params = {"q": q, "quotesCount": 6, "newsCount": 0}
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=8)
        if resp.status_code != 200:
            return {"results": []}

        quotes = resp.json().get("quotes", [])
        results = [
            {
                "name": item.get("shortname") or item.get("longname") or item.get("symbol"),
                "ticker": item.get("symbol"),
                "exchange": item.get("exchDisp", ""),
            }
            for item in quotes
            if item.get("quoteType") == "EQUITY"
        ]
        return {"results": results}
    except Exception:
        return {"results": []}


@app.get("/api/predict")
def get_prediction(company: str = Query(..., min_length=1)):
    """
    Main endpoint called by React to get Layoff Risk + SHAP + FinBERT + LLM Report.
    """
    try:
        result = predict_company(company)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

        if __name__ == "__main__":
    import uvicorn
    print("\nStarting FastAPI server on http://127.0.0.1:8000 ...")
    uvicorn.run(app, host="127.0.0.1", port=8000)