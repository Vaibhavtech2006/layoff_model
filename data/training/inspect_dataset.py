import pandas as pd
import numpy as np
from pathlib import Path


# ========================================
# CONFIG
# ========================================

INPUT_FILE = Path("data/training/historical_features.csv")


# ========================================
# LOAD DATA
# ========================================

print("=" * 50)
print("LOADING HISTORICAL FEATURE DATASET")
print("=" * 50)

df = pd.read_csv(INPUT_FILE)

print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print()


# ========================================
# BASIC INFORMATION
# ========================================

print("=" * 50)
print("DATASET COLUMNS")
print("=" * 50)

for i, col in enumerate(df.columns, 1):
    print(f"{i:2}. {col}")

print()


# ========================================
# DATA TYPES
# ========================================

print("=" * 50)
print("DATA TYPES")
print("=" * 50)

print(df.dtypes)
print()


# ========================================
# DUPLICATES
# ========================================

print("=" * 50)
print("DUPLICATE CHECK")
print("=" * 50)

duplicate_rows = df.duplicated().sum()

duplicate_company_snapshots = df.duplicated(
    subset=["company", "snapshot_date"]
).sum()

print(f"Duplicate rows: {duplicate_rows:,}")
print(f"Duplicate company/snapshot pairs: {duplicate_company_snapshots:,}")
print()


# ========================================
# MISSING VALUES
# ========================================

print("=" * 50)
print("MISSING VALUES")
print("=" * 50)

missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)

missing_table = pd.DataFrame({
    "Missing": missing,
    "Percentage": missing_pct
})

missing_table = missing_table[
    missing_table["Missing"] > 0
].sort_values("Missing", ascending=False)

if len(missing_table) == 0:
    print("No missing values.")
else:
    print(missing_table)

print()


# ========================================
# TARGET DISTRIBUTION
# ========================================

print("=" * 50)
print("TARGET DISTRIBUTION")
print("=" * 50)

target_counts = df["layoff_next_90d"].value_counts().sort_index()
target_pct = (
    df["layoff_next_90d"]
    .value_counts(normalize=True)
    .sort_index() * 100
).round(2)

for target_value in target_counts.index:
    label = "No layoff" if target_value == 0 else "Layoff"
    print(
        f"{target_value} ({label}): "
        f"{target_counts[target_value]:,} "
        f"({target_pct[target_value]:.2f}%)"
    )

print()


# ========================================
# COMPANY STATISTICS
# ========================================

print("=" * 50)
print("COMPANY STATISTICS")
print("=" * 50)

print(f"Unique companies: {df['company'].nunique():,}")
print(
    f"Average snapshots/company: "
    f"{df.groupby('company').size().mean():.2f}"
)
print(
    f"Minimum snapshots/company: "
    f"{df.groupby('company').size().min()}"
)
print(
    f"Maximum snapshots/company: "
    f"{df.groupby('company').size().max()}"
)

print()


# ========================================
# DATE RANGE
# ========================================

print("=" * 50)
print("DATE RANGE")
print("=" * 50)

df["snapshot_date"] = pd.to_datetime(
    df["snapshot_date"],
    errors="coerce"
)

print(f"Start: {df['snapshot_date'].min().date()}")
print(f"End:   {df['snapshot_date'].max().date()}")

print()


# ========================================
# NUMERIC FEATURES
# ========================================

print("=" * 50)
print("NUMERIC FEATURE SUMMARY")
print("=" * 50)

numeric_cols = df.select_dtypes(
    include=[np.number]
).columns.tolist()

print(
    df[numeric_cols]
    .describe()
    .T
    .round(3)
)

print()


# ========================================
# CONSTANT FEATURES
# ========================================

print("=" * 50)
print("CONSTANT FEATURES")
print("=" * 50)

constant_features = []

for col in numeric_cols:
    if df[col].nunique(dropna=True) <= 1:
        constant_features.append(col)

if constant_features:
    for col in constant_features:
        print(col)
else:
    print("No constant numeric features.")

print()


# ========================================
# EXTREME VALUES
# ========================================

print("=" * 50)
print("EXTREME VALUE CHECK")
print("=" * 50)

market_features = [
    "return_30d",
    "return_90d",
    "volatility_30d",
    "volatility_90d",
    "drawdown_30d",
    "drawdown_90d",
    "volume_change_30d"
]

for col in market_features:

    if col not in df.columns:
        continue

    series = df[col].dropna()

    if len(series) == 0:
        continue

    print(f"\n{col}")
    print(f"  Minimum : {series.min():.4f}")
    print(f"  Maximum : {series.max():.4f}")
    print(f"  Median  : {series.median():.4f}")

print()


# ========================================
# HISTORICAL LAYOFF FEATURES
# ========================================

print("=" * 50)
print("LAYOFF FEATURE SUMMARY")
print("=" * 50)

layoff_features = [
    "company_previous_layoffs",
    "company_previous_laid_off",
    "days_since_last_layoff",
    "company_layoffs_30d",
    "company_layoffs_90d",
    "company_layoffs_365d",
    "industry_layoffs_30d",
    "industry_layoffs_90d",
    "industry_layoffs_365d"
]

existing_layoff_features = [
    col for col in layoff_features
    if col in df.columns
]

print(
    df[existing_layoff_features]
    .describe()
    .T
    .round(3)
)

print()


# ========================================
# ZERO / NEGATIVE DAYS SINCE LAYOFF
# ========================================

if "days_since_last_layoff" in df.columns:

    print("=" * 50)
    print("DAYS SINCE LAST LAYOFF")
    print("=" * 50)

    print(
        f"No previous layoff (-1): "
        f"{(df['days_since_last_layoff'] == -1).sum():,}"
    )

    print(
        f"Zero days: "
        f"{(df['days_since_last_layoff'] == 0).sum():,}"
    )

    print(
        f"Positive days: "
        f"{(df['days_since_last_layoff'] > 0).sum():,}"
    )

    print()


# ========================================
# INDUSTRY DISTRIBUTION
# ========================================

print("=" * 50)
print("TOP INDUSTRIES")
print("=" * 50)

print(
    df["Industry"]
    .value_counts()
    .head(15)
)

print()


# ========================================
# TICKER COVERAGE
# ========================================

if "ticker" in df.columns:

    print("=" * 50)
    print("TICKER COVERAGE")
    print("=" * 50)

    ticker_missing = df["ticker"].isna().sum()

    print(
        f"Rows with ticker: "
        f"{len(df) - ticker_missing:,} "
        f"({(1 - ticker_missing / len(df)) * 100:.2f}%)"
    )

    print(
        f"Rows without ticker: "
        f"{ticker_missing:,} "
        f"({ticker_missing / len(df) * 100:.2f}%)"
    )

    print()


# ========================================
# CORRELATION WITH TARGET
# ========================================

print("=" * 50)
print("FEATURE CORRELATION WITH TARGET")
print("=" * 50)

correlations = (
    df[numeric_cols]
    .corr()["layoff_next_90d"]
    .drop("layoff_next_90d")
    .sort_values(key=abs, ascending=False)
)

print(correlations.round(4))

print()


# ========================================
# TARGET RATE BY YEAR
# ========================================

print("=" * 50)
print("LAYOFF RATE BY YEAR")
print("=" * 50)

df["year"] = df["snapshot_date"].dt.year

yearly_rate = (
    df.groupby("year")["layoff_next_90d"]
    .agg(["count", "sum", "mean"])
)

yearly_rate["rate_%"] = (
    yearly_rate["mean"] * 100
).round(2)

print(
    yearly_rate[
        ["count", "sum", "rate_%"]
    ]
)

print()


# ========================================
# FINAL SUMMARY
# ========================================

print("=" * 50)
print("FINAL DATASET SUMMARY")
print("=" * 50)

print(f"Rows                    : {len(df):,}")
print(f"Companies               : {df['company'].nunique():,}")
print(f"Features                : {len(df.columns):,}")
print(f"Positive labels         : {target_counts.get(1, 0):,}")
print(f"Negative labels         : {target_counts.get(0, 0):,}")
print(f"Duplicate rows          : {duplicate_rows:,}")
print(f"Duplicate company/date  : {duplicate_company_snapshots:,}")

print()
print("=" * 50)
print("INSPECTION COMPLETE")
print("=" * 50)