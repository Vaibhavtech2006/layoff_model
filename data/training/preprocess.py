
import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path(
    "data/training/historical_features_v2.csv"
)

OUTPUT_FILE = Path(
    "data/training/ml_dataset_v2.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Input shape: {df.shape}")


# ============================================================
# DATE PROCESSING
# ============================================================

df["snapshot_date"] = pd.to_datetime(
    df["snapshot_date"],
    errors="coerce"
)

df["snapshot_year"] = (
    df["snapshot_date"].dt.year
)

df["snapshot_month"] = (
    df["snapshot_date"].dt.month
)

df = df.drop(
    columns=["snapshot_date"],
    errors="ignore"
)


# ============================================================
# TARGET CHECK
# ============================================================

if "layoff_next_90d" not in df.columns:
    raise ValueError(
        "Target column 'layoff_next_90d' not found."
    )

df["layoff_next_90d"] = pd.to_numeric(
    df["layoff_next_90d"],
    errors="coerce"
)

df = df.dropna(
    subset=["layoff_next_90d"]
)

df["layoff_next_90d"] = (
    df["layoff_next_90d"]
    .astype(int)
)

print("\nTarget distribution:")

print(
    df["layoff_next_90d"]
    .value_counts()
    .sort_index()
)


# ============================================================
# REMOVE IDENTIFIERS
# ============================================================

df = df.drop(
    columns=[
        "company",
        "ticker",
        "sentiment_date"
    ],
    errors="ignore"
)


# ============================================================
# INDUSTRY
# ============================================================

if "Industry" in df.columns:

    df["Industry"] = (
        df["Industry"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
    )

    industry_dummies = pd.get_dummies(
        df["Industry"],
        prefix="industry",
        dtype=int
    )

    df = pd.concat(
        [
            df.drop(columns=["Industry"]),
            industry_dummies
        ],
        axis=1
    )


# ============================================================
# NUMERIC COLUMNS
# ============================================================

target = "layoff_next_90d"

feature_columns = [
    column
    for column in df.columns
    if column != target
]

for column in feature_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ============================================================
# HANDLE INFINITE VALUES
# ============================================================

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)


# ============================================================
# SENTIMENT FEATURES
# ============================================================

sentiment_features = [
    "news_count",
    "negative_news_ratio",
    "positive_news_ratio",
    "neutral_news_ratio",
    "avg_sentiment",
    "layoff_keyword_count",
    "restructuring_keyword_count",
    "cost_cutting_keyword_count",
    "hiring_freeze_keyword_count",
    "sentiment_available",
    "sentiment_age_days"
]

print("\nSentiment features:")

for column in sentiment_features:

    if column in df.columns:

        available = df[column].notna().sum()

        print(
            f"{column:35s}"
            f"{available:6d} available"
        )


# ============================================================
# IMPORTANT:
# DO NOT IMPUTE HERE.
#
# Missing values will be handled by the training pipeline
# using training-only statistics.
# ============================================================

# Keep NaN values.


# ============================================================
# VALIDATION
# ============================================================

if df[target].isna().any():

    raise ValueError(
        "Target contains missing values."
    )

if not set(
    df[target].unique()
).issubset({0, 1}):

    raise ValueError(
        "Target must contain only 0 and 1."
    )


# ============================================================
# REMOVE SNAPSHOT YEAR/MONTH FROM MODEL FEATURES
# ============================================================
#
# Keep them temporarily for time-based splitting.
# train.py will remove them before training.
#

# ============================================================
# FINAL CHECK
# ============================================================

print("\nFinal dataset information:")

print(
    f"Rows    : {len(df):,}"
)

print(
    f"Columns : {len(df.columns):,}"
)

print(
    f"Features: {len(df.columns) - 1:,}"
)

print(
    f"Missing values: "
    f"{df.drop(columns=[target]).isna().sum().sum():,}"
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n========================================")
print("PREPROCESSING COMPLETED")
print("========================================")

print(
    f"Saved to: {OUTPUT_FILE}"
)

print(
    f"Shape   : {df.shape}"
)

print("========================================\n")

