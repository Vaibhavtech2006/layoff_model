
import pandas as pd
from pathlib import Path

FEATURE_FILE = Path("data/training/historical_features.csv")
SENTIMENT_FILE = Path("data/training/historical_sentiment.csv")
OUTPUT_FILE = Path("data/training/historical_features_v2.csv")

MAX_DAYS = 30


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading historical features...")

features = pd.read_csv(FEATURE_FILE)

print(f"Historical features: {features.shape}")


print("\nLoading sentiment features...")

sentiment = pd.read_csv(SENTIMENT_FILE)

print(f"Sentiment features: {sentiment.shape}")


# ============================================================
# CLEAN DATA
# ============================================================

features["company"] = (
    features["company"]
    .fillna("")
    .astype(str)
    .str.strip()
)

sentiment["company"] = (
    sentiment["company"]
    .fillna("")
    .astype(str)
    .str.strip()
)

features["snapshot_date"] = pd.to_datetime(
    features["snapshot_date"],
    errors="coerce"
)

sentiment["snapshot_date"] = pd.to_datetime(
    sentiment["snapshot_date"],
    errors="coerce"
)

features = features.dropna(
    subset=["snapshot_date"]
)

sentiment = sentiment.dropna(
    subset=["snapshot_date"]
)


# ============================================================
# CHECK DUPLICATES
# ============================================================

duplicates = sentiment.duplicated(
    subset=["company", "snapshot_date"]
).sum()

print(
    f"\nDuplicate sentiment company/date rows: {duplicates}"
)


# ============================================================
# RENAME SENTIMENT DATE
# ============================================================

sentiment = sentiment.rename(
    columns={
        "snapshot_date": "sentiment_date"
    }
)


# ============================================================
# TIME-BASED MERGE BY COMPANY
# ============================================================

print(
    f"\nMerging previous sentiment "
    f"(maximum age = {MAX_DAYS} days)..."
)

merged_parts = []

companies = features["company"].unique()

print(
    f"Companies to process: {len(companies):,}"
)

for i, company in enumerate(companies, start=1):

    company_features = features[
        features["company"] == company
    ].copy()

    company_sentiment = sentiment[
        sentiment["company"] == company
    ].copy()

    # No sentiment available for this company
    if company_sentiment.empty:

        company_features["sentiment_date"] = pd.NaT

        for column in [
            "news_count",
            "negative_news_ratio",
            "positive_news_ratio",
            "neutral_news_ratio",
            "avg_sentiment",
            "layoff_keyword_count",
            "restructuring_keyword_count",
            "cost_cutting_keyword_count",
            "hiring_freeze_keyword_count"
        ]:
            company_features[column] = pd.NA

        merged_parts.append(company_features)

        continue

    # Sort ONLY by date inside each company
    company_features = company_features.sort_values(
        "snapshot_date"
    )

    company_sentiment = company_sentiment.sort_values(
        "sentiment_date"
    )

    # Time-based merge
    company_merged = pd.merge_asof(
        company_features,
        company_sentiment,
        left_on="snapshot_date",
        right_on="sentiment_date",
        direction="backward",
        tolerance=pd.Timedelta(
            days=MAX_DAYS
        ),
        suffixes=("", "_sentiment")
    )

    merged_parts.append(company_merged)

    # Progress
    if i % 250 == 0:

        print(
            f"Processed {i:,}/{len(companies):,} companies..."
        )


# ============================================================
# COMBINE RESULTS
# ============================================================

print("\nCombining company results...")

merged = pd.concat(
    merged_parts,
    ignore_index=True
)


# ============================================================
# RESTORE ORIGINAL ORDER
# ============================================================

merged = merged.sort_values(
    ["company", "snapshot_date"]
).reset_index(drop=True)


# ============================================================
# SENTIMENT AVAILABILITY
# ============================================================

merged["sentiment_available"] = (
    merged["news_count"]
    .notna()
    .astype(int)
)


# ============================================================
# SENTIMENT AGE
# ============================================================

merged["sentiment_age_days"] = (
    merged["snapshot_date"]
    - merged["sentiment_date"]
).dt.days

merged.loc[
    merged["sentiment_available"] == 0,
    "sentiment_age_days"
] = pd.NA


# ============================================================
# COVERAGE
# ============================================================

sentiment_columns = [
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

print("\nSentiment coverage:")

for column in sentiment_columns:

    if column in merged.columns:

        count = merged[column].notna().sum()

        percentage = (
            count / len(merged) * 100
        )

        print(
            f"{column:35s}"
            f"{count:6d} "
            f"({percentage:.2f}%)"
        )


# ============================================================
# SUMMARY
# ============================================================

available = merged[
    "sentiment_available"
].sum()

coverage = (
    available / len(merged) * 100
)

print("\n========================================")
print("SENTIMENT MERGE COMPLETED")
print("========================================")

print(
    f"Total rows       : {len(merged):,}"
)

print(
    f"Sentiment rows   : {available:,}"
)

print(
    f"Coverage         : {coverage:.2f}%"
)

if available > 0:

    average_age = merged.loc[
        merged["sentiment_available"] == 1,
        "sentiment_age_days"
    ].mean()

    print(
        f"Average age      : {average_age:.1f} days"
    )

print(
    f"Columns          : {len(merged.columns):,}"
)

print(
    f"Saved            : {OUTPUT_FILE}"
)

print("========================================\n")


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

merged.to_csv(
    OUTPUT_FILE,
    index=False
)

