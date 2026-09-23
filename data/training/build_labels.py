import pandas as pd
from datasets import load_dataset
from pathlib import Path


# -----------------------------
# Paths
# -----------------------------

OUTPUT_DIR = Path("data/training")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_FILE = OUTPUT_DIR / "layoff_events.csv"
LABEL_FILE = OUTPUT_DIR / "layoff_labels.csv"


# -----------------------------
# Load Layoffs.fyi dataset
# -----------------------------

print("Loading historical layoff dataset...")

dataset = load_dataset(
    "hakunamatata1997/Layoffs_Data",
    split="train"
)

df = dataset.to_pandas()

print(f"Dataset loaded: {len(df)} rows")


# -----------------------------
# Save raw dataset locally
# -----------------------------

df.to_csv(RAW_FILE, index=False)

print(f"Raw dataset saved to: {RAW_FILE}")


# -----------------------------
# Clean basic columns
# -----------------------------

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)

df["Company"] = (
    df["Company"]
    .astype(str)
    .str.strip()
)

# Remove invalid records
df = df.dropna(
    subset=["Date", "Company"]
)

df = df.sort_values(
    ["Company", "Date"]
)


# -----------------------------
# Create monthly snapshots
# -----------------------------

start_date = df["Date"].min().replace(day=1)
end_date = df["Date"].max().replace(day=1)

snapshot_dates = pd.date_range(
    start=start_date,
    end=end_date,
    freq="MS"
)


# -----------------------------
# Generate 90-day labels
# -----------------------------

rows = []

companies = df["Company"].unique()

print(f"Companies found: {len(companies)}")
print("Creating 90-day labels...")


for company in companies:

    company_events = df[
        df["Company"] == company
    ]

    # First observed event for this company
    first_event = company_events["Date"].min()

    for snapshot_date in snapshot_dates:

        # Don't create snapshots before company appears
        if snapshot_date < first_event:
            continue

        # Future events only
        future_events = company_events[
            company_events["Date"] > snapshot_date
        ]

        # Events within next 90 days
        future_90_days = future_events[
            future_events["Date"]
            <= snapshot_date + pd.Timedelta(days=90)
        ]

        layoff_next_90d = int(
            len(future_90_days) > 0
        )

        rows.append({
            "snapshot_date": snapshot_date,
            "company": company,
            "layoff_next_90d": layoff_next_90d
        })


# -----------------------------
# Create dataframe
# -----------------------------

labels = pd.DataFrame(rows)


# -----------------------------
# Save labels
# -----------------------------

labels.to_csv(
    LABEL_FILE,
    index=False
)


# -----------------------------
# Summary
# -----------------------------

print("\n--------------------------------")
print("PHASE 2 COMPLETE")
print("--------------------------------")

print(f"Total snapshots: {len(labels)}")
print(f"Companies: {labels['company'].nunique()}")

print("\nTarget distribution:")

print(
    labels["layoff_next_90d"]
    .value_counts()
)

print("\nTarget percentage:")

print(
    labels["layoff_next_90d"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nFiles created:")

print(RAW_FILE)
print(LABEL_FILE)