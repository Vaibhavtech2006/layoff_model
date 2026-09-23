import pandas as pd
import yfinance as yf
import time
from pathlib import Path


INPUT_FILE = "data/training/layoff_events.csv"
OUTPUT_FILE = "data/training/company_tickers.csv"


df = pd.read_csv(INPUT_FILE)

companies = (
    df["Company"]
    .dropna()
    .astype(str)
    .str.strip()
    .drop_duplicates()
    .tolist()
)


# Load existing mappings
if Path(OUTPUT_FILE).exists():
    old = pd.read_csv(OUTPUT_FILE)
    mapping = dict(zip(old["company"], old["ticker"]))
else:
    mapping = {}


results = []

print(f"Total companies: {len(companies)}")

for i, company in enumerate(companies, 1):

    # Keep already-known ticker
    if company in mapping and pd.notna(mapping[company]) and mapping[company] != "":
        results.append({
            "company": company,
            "ticker": mapping[company]
        })
        continue

    ticker = ""

    try:
        search = yf.Search(
            company,
            max_results=5
        )

        quotes = search.quotes

        # Find an equity candidate
        for quote in quotes:

            symbol = quote.get("symbol", "")
            quote_type = quote.get("quoteType", "")

            if quote_type == "EQUITY" and symbol:
                ticker = symbol
                break

    except Exception as e:
        print(f"Error: {company} -> {e}")

    results.append({
        "company": company,
        "ticker": ticker
    })

    print(
        f"[{i}/{len(companies)}] "
        f"{company} -> {ticker or 'UNKNOWN'}"
    )

    time.sleep(0.2)


result_df = pd.DataFrame(results)

result_df.to_csv(
    OUTPUT_FILE,
    index=False
)


mapped = (result_df["ticker"] != "").sum()
unmapped = (result_df["ticker"] == "").sum()

print("\n==============================")
print("TICKER MAPPING COMPLETE")
print("==============================")

print(f"Total:    {len(result_df)}")
print(f"Mapped:   {mapped}")
print(f"Unmapped: {unmapped}")

print(f"\nSaved to: {OUTPUT_FILE}")