from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "training" / "ml_dataset_v2.csv"
MODEL_PATH = ROOT / "data" / "training" / "models" / "xgboost_v2.pkl"
OUTPUT_PATH = ROOT / "data" / "training" / "models" / "xgboost_v2_calibrated.pkl"

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("FINAL MODEL CALIBRATION")
print("=" * 60)

print(f"Dataset rows    : {len(df):,}")
print(f"Dataset columns : {len(df.columns)}")

# ============================================================
# SPLIT
# ============================================================

train_df = df[df["snapshot_year"] <= 2022].copy()
cal_df = df[df["snapshot_year"] == 2023].copy()
test_df = df[df["snapshot_year"] == 2024].copy()

target = "layoff_next_90d"

drop_cols = [
    target,
    "snapshot_year",
    "snapshot_month",
]

X_cal = cal_df.drop(columns=drop_cols)
y_cal = cal_df[target].astype(int)

X_test = test_df.drop(columns=drop_cols)
y_test = test_df[target].astype(int)

print(f"\n2023 calibration rows: {len(cal_df):,}")
print(f"2024 test rows       : {len(test_df):,}")

# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model_package = joblib.load(MODEL_PATH)

# Support either raw model or packaged model
if isinstance(model_package, dict):
    model = model_package["model"]
    imputer = model_package.get("imputer")
    features = model_package.get("features")
else:
    model = model_package
    imputer = None
    features = list(X_cal.columns)

if features is not None:
    X_cal = X_cal.reindex(columns=features, fill_value=0)
    X_test = X_test.reindex(columns=features, fill_value=0)

# ============================================================
# IMPUTATION
# ============================================================

if imputer is not None:
    X_cal_imp = imputer.transform(X_cal)
    X_test_imp = imputer.transform(X_test)
else:
    # Training model should normally contain its own imputer.
    # This fallback handles raw XGBoost models.
    from sklearn.impute import SimpleImputer

    temp_imputer = SimpleImputer(strategy="median")
    X_cal_imp = temp_imputer.fit_transform(X_cal)
    X_test_imp = temp_imputer.transform(X_test)
    imputer = temp_imputer

# ============================================================
# RAW PROBABILITIES
# ============================================================

cal_raw = model.predict_proba(X_cal_imp)[:, 1]
test_raw = model.predict_proba(X_test_imp)[:, 1]

print("\n" + "=" * 60)
print("2023 RAW MODEL")
print("=" * 60)

print(f"ROC-AUC : {roc_auc_score(y_cal, cal_raw):.4f}")
print(f"PR-AUC  : {average_precision_score(y_cal, cal_raw):.4f}")
print(f"Brier   : {brier_score_loss(y_cal, cal_raw):.4f}")

# ============================================================
# ISOTONIC CALIBRATION
# ============================================================

calibrator = IsotonicRegression(
    y_min=0,
    y_max=1,
    out_of_bounds="clip"
)

calibrator.fit(cal_raw, y_cal)

cal_prob = calibrator.predict(cal_raw)
test_prob = calibrator.predict(test_raw)

print("\n" + "=" * 60)
print("2023 CALIBRATED MODEL")
print("=" * 60)

print(f"ROC-AUC : {roc_auc_score(y_cal, cal_prob):.4f}")
print(f"PR-AUC  : {average_precision_score(y_cal, cal_prob):.4f}")
print(f"Brier   : {brier_score_loss(y_cal, cal_prob):.4f}")

# ============================================================
# HIGH THRESHOLD
# Best F1 on 2023 calibration set
# ============================================================

thresholds = np.linspace(
    max(0.01, np.percentile(cal_prob, 50)),
    min(0.50, np.percentile(cal_prob, 99.5)),
    500
)

best_threshold = 0.5
best_f1 = -1

for threshold in thresholds:

    pred = (cal_prob >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_cal,
        pred,
        average="binary",
        zero_division=0
    )

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

high_threshold = float(best_threshold)

pred_high = (cal_prob >= high_threshold).astype(int)

precision, recall, f1, _ = precision_recall_fscore_support(
    y_cal,
    pred_high,
    average="binary",
    zero_division=0
)

print("\n" + "=" * 60)
print("HIGH-RISK THRESHOLD")
print("=" * 60)

print(f"Threshold : {high_threshold:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1        : {f1:.4f}")

# ============================================================
# EXTREME HIGH
#
# Use the 95th percentile of CALIBRATION probabilities,
# but force a meaningful gap from HIGH.
# ============================================================

extreme_threshold = float(np.percentile(cal_prob, 95))

# Minimum separation from HIGH
minimum_gap = 0.02

if extreme_threshold < high_threshold + minimum_gap:
    extreme_threshold = high_threshold + minimum_gap

# Never allow extreme threshold above 95th percentile
# by blindly forcing it down.
# We only require a meaningful category separation.

print("\n" + "=" * 60)
print("EXTREME-HIGH THRESHOLD")
print("=" * 60)

print(f"Threshold : {extreme_threshold:.4f}")

# ============================================================
# 2024 FINAL UNSEEN TEST
# ============================================================

print("\n" + "=" * 60)
print("2024 FINAL UNSEEN TEST")
print("=" * 60)

print("\nRAW PROBABILITIES")

print(f"ROC-AUC : {roc_auc_score(y_test, test_raw):.4f}")
print(f"PR-AUC  : {average_precision_score(y_test, test_raw):.4f}")
print(f"Brier   : {brier_score_loss(y_test, test_raw):.4f}")

print("\nCALIBRATED PROBABILITIES")

print(f"ROC-AUC : {roc_auc_score(y_test, test_prob):.4f}")
print(f"PR-AUC  : {average_precision_score(y_test, test_prob):.4f}")
print(f"Brier   : {brier_score_loss(y_test, test_prob):.4f}")

# ============================================================
# 2024 HIGH-RISK CLASSIFICATION
# ============================================================

test_high_pred = (test_prob >= high_threshold).astype(int)

print("\n" + "=" * 60)
print("2024 HIGH-RISK CLASSIFICATION")
print("=" * 60)

print(
    classification_report(
        y_test,
        test_high_pred,
        target_names=["No Layoff", "Layoff"],
        zero_division=0
    )
)

print("Confusion Matrix:")
print(confusion_matrix(y_test, test_high_pred))

# ============================================================
# 2024 RISK DISTRIBUTION
# ============================================================

def risk_category(prob):

    if prob >= extreme_threshold:
        return "EXTREME HIGH"

    elif prob >= high_threshold:
        return "HIGH"

    else:
        return "LOW"


risk_categories = np.array([
    risk_category(p)
    for p in test_prob
])

unique, counts = np.unique(
    risk_categories,
    return_counts=True
)

print("\n" + "=" * 60)
print("2024 RISK DISTRIBUTION")
print("=" * 60)

for category in ["LOW", "HIGH", "EXTREME HIGH"]:

    count = int(
        counts[unique.tolist().index(category)]
    ) if category in unique else 0

    percentage = count / len(test_prob) * 100

    print(
        f"{category:<15} {count:>6,} "
        f"({percentage:>6.2f}%)"
    )

# ============================================================
# PROBABILITY SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("RISK PROBABILITY SUMMARY")
print("=" * 60)

print(f"Minimum probability : {test_prob.min():.4f}")
print(f"Maximum probability : {test_prob.max():.4f}")
print(f"Mean probability    : {test_prob.mean():.4f}")
print(f"Median probability  : {np.median(test_prob):.4f}")

# ============================================================
# SAVE FINAL MODEL
# ============================================================

final_package = {
    "model": model,
    "imputer": imputer,
    "calibrator": calibrator,
    "features": features,

    "high_threshold": high_threshold,
    "extreme_threshold": extreme_threshold,

    "calibration_year": 2023,
    "test_year": 2024,

    "risk_categories": {
        "LOW": f"< {high_threshold:.4f}",
        "HIGH": (
            f"{high_threshold:.4f} - "
            f"{extreme_threshold:.4f}"
        ),
        "EXTREME HIGH": f">= {extreme_threshold:.4f}",
    }
}

joblib.dump(
    final_package,
    OUTPUT_PATH
)

# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 60)
print("CALIBRATION COMPLETE")
print("=" * 60)

print(f"Saved model: {OUTPUT_PATH}")

print("\nFinal thresholds:")
print(f"HIGH         : {high_threshold:.4f}")
print(f"EXTREME HIGH : {extreme_threshold:.4f}")

print("\nPipeline:")
print("2020-2022 -> Training")
print("2023      -> Calibration + Threshold Selection")
print("2024      -> Final Unseen Evaluation")

print("=" * 60)
