import pandas as pd
import numpy as np
import yfinance as yf

from pathlib import Path
from tqdm import tqdm
import time


# =========================================================
# PATHS
# =========================================================

TRAINING_DIR = Path("data/training")

LABEL_FILE = TRAINING_DIR / "layoff_labels.csv"
EVENT_FILE = TRAINING_DIR / "layoff_events.csv"
TICKER_FILE = TRAINING_DIR / "company_tickers.csv"

OUTPUT_FILE = TRAINING_DIR / "historical_features.csv"


# =========================================================
# LOAD DATA
# =========================================================

print("Loading training data...")

labels = pd.read_csv(LABEL_FILE)
events = pd.read_csv(EVENT_FILE)
tickers = pd.read_csv(TICKER_FILE)

# Force all project dates to the same datetime precision
labels["snapshot_date"] = (
    pd.to_datetime(
        labels["snapshot_date"],
        errors="coerce"
    )
    .dt.normalize()
    .astype("datetime64[ns]")
)

events["Date"] = (
    pd.to_datetime(
        events["Date"],
        errors="coerce"
    )
    .dt.normalize()
    .astype("datetime64[ns]")
)

events["Company"] = (
    events["Company"]
    .astype(str)
    .str.strip()
)

tickers["company"] = (
    tickers["company"]
    .astype(str)
    .str.strip()
)

tickers["ticker"] = (
    tickers["ticker"]
    .fillna("")
    .astype(str)
    .str.strip()
)

print(f"Labels: {len(labels):,}")
print(f"Events: {len(events):,}")
print(f"Ticker mappings: {len(tickers):,}")


# =========================================================
# COMPANY INDUSTRY
# =========================================================

company_info = (
    events[
        [
            "Company",
            "Industry"
        ]
    ]
    .drop_duplicates("Company")
    .rename(
        columns={
            "Company": "company"
        }
    )
)


# =========================================================
# MERGE TICKERS + INDUSTRY
# =========================================================

data = labels.merge(
    tickers,
    on="company",
    how="left"
)

data = data.merge(
    company_info,
    on="company",
    how="left"
)

data["ticker"] = (
    data["ticker"]
    .fillna("")
    .astype(str)
    .str.strip()
)

print(
    f"Companies with ticker: "
    f"{data.loc[data['ticker'] != '', 'company'].nunique():,}"
)

print(
    f"Companies without ticker: "
    f"{data.loc[data['ticker'] == '', 'company'].nunique():,}"
)


# =========================================================
# HISTORICAL LAYOFF FEATURES
# =========================================================

print("\nCreating historical layoff features...")


def calculate_layoff_features(row):

    company = row["company"]
    snapshot_date = row["snapshot_date"]
    industry = row["Industry"]

    # -----------------------------------------
    # Events known at snapshot date
    # -----------------------------------------

    previous_events = events[
        (events["Company"] == company)
        &
        (events["Date"] <= snapshot_date)
    ]

    # -----------------------------------------
    # Company history
    # -----------------------------------------

    company_previous_layoffs = len(
        previous_events
    )

    company_previous_laid_off = (
        pd.to_numeric(
            previous_events["Laid_Off_Count"],
            errors="coerce"
        )
        .fillna(0)
        .sum()
    )

    # -----------------------------------------
    # Days since previous layoff
    # -----------------------------------------

    if len(previous_events) > 0:

        last_layoff_date = (
            previous_events["Date"].max()
        )

        days_since_last_layoff = (
            snapshot_date - last_layoff_date
        ).days

    else:

        days_since_last_layoff = -1

    # -----------------------------------------
    # Company recent layoffs
    # -----------------------------------------

    company_layoffs_30d = len(
        previous_events[
            previous_events["Date"]
            >= snapshot_date - pd.Timedelta(days=30)
        ]
    )

    company_layoffs_90d = len(
        previous_events[
            previous_events["Date"]
            >= snapshot_date - pd.Timedelta(days=90)
        ]
    )

    company_layoffs_365d = len(
        previous_events[
            previous_events["Date"]
            >= snapshot_date - pd.Timedelta(days=365)
        ]
    )

    # -----------------------------------------
    # Industry history
    # -----------------------------------------

    if pd.isna(industry):

        industry_events = events.iloc[0:0]

    else:

        industry_events = events[
            (events["Industry"] == industry)
            &
            (events["Date"] <= snapshot_date)
        ]

    industry_layoffs_30d = len(
        industry_events[
            industry_events["Date"]
            >= snapshot_date - pd.Timedelta(days=30)
        ]
    )

    industry_layoffs_90d = len(
        industry_events[
            industry_events["Date"]
            >= snapshot_date - pd.Timedelta(days=90)
        ]
    )

    industry_layoffs_365d = len(
        industry_events[
            industry_events["Date"]
            >= snapshot_date - pd.Timedelta(days=365)
        ]
    )

    return pd.Series({

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
    })


# =========================================================
# CALCULATE LAYOFF FEATURES
# =========================================================

layoff_features = data.apply(
    calculate_layoff_features,
    axis=1
)

data = pd.concat(
    [
        data.reset_index(drop=True),
        layoff_features.reset_index(drop=True)
    ],
    axis=1
)


# =========================================================
# HISTORICAL MARKET FEATURES
# =========================================================

print("\n========================================")
print("DOWNLOADING HISTORICAL MARKET DATA")
print("========================================")

unique_tickers = sorted(
    data.loc[
        data["ticker"] != "",
        "ticker"
    ]
    .drop_duplicates()
    .tolist()
)

print(
    f"Tickers to process: {len(unique_tickers):,}"
)


# =========================================================
# MARKET FEATURE COLUMNS
# =========================================================

market_columns = [
    "return_30d",
    "return_90d",
    "volatility_30d",
    "volatility_90d",
    "drawdown_30d",
    "drawdown_90d",
    "volume_change_30d"
]


for column in market_columns:
    data[column] = np.nan


# =========================================================
# PROCESS EACH TICKER
# =========================================================

successful = 0
failed = 0

failed_tickers = []


for ticker in tqdm(
    unique_tickers,
    desc="Processing tickers"
):

    try:

        ticker_rows = data[
            data["ticker"] == ticker
        ].copy()

        min_snapshot = (
            ticker_rows["snapshot_date"].min()
        )

        max_snapshot = (
            ticker_rows["snapshot_date"].max()
        )

        # Extra history for 90-day calculations
        download_start = (
            min_snapshot
            - pd.Timedelta(days=180)
        )

        download_end = (
            max_snapshot
            + pd.Timedelta(days=5)
        )


        # =================================================
        # DOWNLOAD
        # =================================================

        stock = yf.download(
            ticker,
            start=download_start.strftime("%Y-%m-%d"),
            end=download_end.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
            threads=False
        )


        if stock is None or stock.empty:

            failed += 1

            failed_tickers.append(
                (ticker, "empty download")
            )

            continue


        # =================================================
        # HANDLE YFINANCE MULTIINDEX
        # =================================================

        if isinstance(
            stock.columns,
            pd.MultiIndex
        ):

            stock.columns = (
                stock.columns
                .get_level_values(0)
            )


        stock = stock.reset_index()


        # =================================================
        # NORMALIZE COLUMN NAMES
        # =================================================

        stock.columns = [
            str(column).strip()
            for column in stock.columns
        ]


        # =================================================
        # CHECK REQUIRED COLUMNS
        # =================================================

        required_columns = [
            "Date",
            "Close",
            "Volume"
        ]

        if not all(
            column in stock.columns
            for column in required_columns
        ):

            failed += 1

            failed_tickers.append(
                (
                    ticker,
                    f"missing columns: {list(stock.columns)}"
                )
            )

            continue


        # =================================================
        # NORMALIZE DATE
        # =================================================

        stock["Date"] = (
            pd.to_datetime(
                stock["Date"],
                errors="coerce"
            )
            .dt.normalize()
            .astype("datetime64[ns]")
        )


        # =================================================
        # NUMERIC VALUES
        # =================================================

        stock["Close"] = pd.to_numeric(
            stock["Close"],
            errors="coerce"
        )

        stock["Volume"] = pd.to_numeric(
            stock["Volume"],
            errors="coerce"
        )


        # =================================================
        # REMOVE INVALID ROWS
        # =================================================

        stock = stock.dropna(
            subset=[
                "Date",
                "Close"
            ]
        )


        if len(stock) < 100:

            failed += 1

            failed_tickers.append(
                (
                    ticker,
                    f"insufficient data: {len(stock)} rows"
                )
            )

            continue


        stock = stock.sort_values(
            "Date"
        ).reset_index(drop=True)


        # =================================================
        # DAILY RETURN
        # =================================================

        stock["daily_return"] = (
            stock["Close"].pct_change()
        )


        # =================================================
        # 30-DAY RETURN
        # =================================================

        stock["return_30d"] = (
            stock["Close"].pct_change(30)
        )


        # =================================================
        # 90-DAY RETURN
        # =================================================

        stock["return_90d"] = (
            stock["Close"].pct_change(90)
        )


        # =================================================
        # VOLATILITY
        # =================================================

        stock["volatility_30d"] = (
            stock["daily_return"]
            .rolling(30)
            .std()
            * np.sqrt(252)
        )

        stock["volatility_90d"] = (
            stock["daily_return"]
            .rolling(90)
            .std()
            * np.sqrt(252)
        )


        # =================================================
        # DRAWDOWN
        # =================================================

        rolling_max_30 = (
            stock["Close"]
            .rolling(30)
            .max()
        )

        rolling_max_90 = (
            stock["Close"]
            .rolling(90)
            .max()
        )

        stock["drawdown_30d"] = (
            stock["Close"]
            / rolling_max_30
            - 1
        )

        stock["drawdown_90d"] = (
            stock["Close"]
            / rolling_max_90
            - 1
        )


        # =================================================
        # VOLUME CHANGE
        # =================================================

        stock["volume_change_30d"] = (
            stock["Volume"].pct_change(30)
        )


        # =================================================
        # MARKET DATA
        # =================================================

        market_data = stock[
            [
                "Date",
                "return_30d",
                "return_90d",
                "volatility_30d",
                "volatility_90d",
                "drawdown_30d",
                "drawdown_90d",
                "volume_change_30d"
            ]
        ].copy()


        # Explicitly force merge key to datetime64[ns]
        market_data["Date"] = (
            pd.to_datetime(
                market_data["Date"],
                errors="coerce"
            )
            .dt.normalize()
            .astype("datetime64[ns]")
        )

        market_data = (
            market_data
            .dropna(subset=["Date"])
            .sort_values("Date")
            .reset_index(drop=True)
        )


        # =================================================
        # SNAPSHOT DATA
        # =================================================

        snapshot_data = ticker_rows[
            [
                "company",
                "snapshot_date"
            ]
        ].copy()

        snapshot_data["ticker"] = ticker


        # Explicitly force merge key to datetime64[ns]
        snapshot_data["snapshot_date"] = (
            pd.to_datetime(
                snapshot_data["snapshot_date"],
                errors="coerce"
            )
            .dt.normalize()
            .astype("datetime64[ns]")
        )

        snapshot_data = (
            snapshot_data
            .dropna(subset=["snapshot_date"])
            .sort_values("snapshot_date")
            .reset_index(drop=True)
        )


        # =================================================
        # FINAL DATETIME SAFETY CHECK
        # =================================================

        if (
            snapshot_data["snapshot_date"].dtype
            != market_data["Date"].dtype
        ):

            raise TypeError(
                f"Datetime mismatch: "
                f"{snapshot_data['snapshot_date'].dtype} "
                f"vs "
                f"{market_data['Date'].dtype}"
            )


        # =================================================
        # MERGE SNAPSHOTS WITH MARKET DATA
        # =================================================

        merged = pd.merge_asof(
            snapshot_data,
            market_data,
            left_on="snapshot_date",
            right_on="Date",
            direction="backward"
        )


        # =================================================
        # UPDATE ORIGINAL DATAFRAME
        # =================================================

        indices = ticker_rows.index

        for column in market_columns:

            values = merged[
                column
            ].to_numpy()

            data.loc[
                indices,
                column
            ] = values


        successful += 1


    except Exception as e:

        failed += 1

        failed_tickers.append(
            (
                ticker,
                str(e)
            )
        )

        print(
            f"\nFailed: {ticker}"
        )

        print(
            f"Reason: {e}"
        )


    # Small delay
    time.sleep(0.15)


# =========================================================
# CLEAN INVALID VALUES
# =========================================================

data = data.replace(
    [np.inf, -np.inf],
    np.nan
)


# =========================================================
# SORT
# =========================================================

data = data.sort_values(
    [
        "company",
        "snapshot_date"
    ]
).reset_index(drop=True)


# =========================================================
# SAVE
# =========================================================

data.to_csv(
    OUTPUT_FILE,
    index=False
)


# =========================================================
# FINAL REPORT
# =========================================================

print("\n========================================")
print("HISTORICAL FEATURE ENGINEERING COMPLETE")
print("========================================")

print(
    f"Rows: {len(data):,}"
)

print(
    f"Companies: "
    f"{data['company'].nunique():,}"
)

print(
    f"Columns: {len(data.columns):,}"
)

print(
    f"\nSuccessful tickers: {successful:,}"
)

print(
    f"Failed tickers: {failed:,}"
)

print(
    "\nOutput saved to:"
)

print(
    OUTPUT_FILE
)


# =========================================================
# MARKET FEATURE COVERAGE
# =========================================================

print("\n========================================")
print("MARKET FEATURE COVERAGE")
print("========================================")

for column in market_columns:

    available = data[column].notna().sum()

    percentage = (
        available
        / len(data)
        * 100
    )

    print(
        f"{column:25s}"
        f"{available:,} "
        f"({percentage:.2f}%)"
    )


# =========================================================
# MISSING VALUES
# =========================================================

print("\n========================================")
print("MISSING VALUES")
print("========================================")

print(
    data.isnull().sum()
)


# =========================================================
# SAMPLE
# =========================================================

print("\n========================================")
print("SAMPLE MARKET FEATURES")
print("========================================")

sample = (
    data[
        [
            "company",
            "snapshot_date",
            "ticker",
            "return_30d",
            "return_90d",
            "volatility_30d",
            "drawdown_90d"
        ]
    ]
    .dropna(
        subset=["return_30d"]
    )
    .head(10)
)

print(
    sample.to_string(index=False)
)


# =========================================================
# FAILED TICKER SAMPLE
# =========================================================

print("\n========================================")
print("FAILED TICKER SAMPLE")
print("========================================")

for ticker, reason in failed_tickers[:20]:

    print(
        f"{ticker:15s} -> {reason}"
    )


print("\n========================================")
print("DONE")
print("========================================")

