import pandas as pd

FILE = "data/training/layoff_labels.csv"

df = pd.read_csv(FILE)

df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])

print("\n========== LABEL VALIDATION ==========")

# 1. Basic information
print(f"\nRows: {len(df):,}")
print(f"Companies: {df['company'].nunique():,}")

# 2. Missing values
print("\nMissing values:")
print(df.isnull().sum())

# 3. Duplicate rows
duplicates = df.duplicated().sum()
print(f"\nDuplicate rows: {duplicates:,}")

# 4. Target distribution
print("\nTarget distribution:")
print(df["layoff_next_90d"].value_counts())

print("\nTarget percentage:")
print(
    df["layoff_next_90d"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

# 5. Date range
print("\nSnapshot date range:")
print("From:", df["snapshot_date"].min().date())
print("To:  ", df["snapshot_date"].max().date())

# 6. Check invalid targets
invalid_targets = df[
    ~df["layoff_next_90d"].isin([0, 1])
]

print(f"\nInvalid target values: {len(invalid_targets)}")

# 7. Check duplicate company + snapshot
company_date_duplicates = df.duplicated(
    subset=["company", "snapshot_date"]
).sum()

print(
    f"Duplicate company/snapshot combinations: "
    f"{company_date_duplicates:,}"
)

# 8. Number of snapshots per company
snapshot_counts = df.groupby("company").size()

print("\nSnapshots per company:")
print(snapshot_counts.describe())

# 9. Target by year
df["year"] = df["snapshot_date"].dt.year

print("\nTarget distribution by year:")
print(
    pd.crosstab(
        df["year"],
        df["layoff_next_90d"],
        normalize="index"
    ).round(3)
)

print("\n========== VALIDATION COMPLETE ==========")