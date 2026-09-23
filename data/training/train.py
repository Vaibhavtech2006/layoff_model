
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix
)

from xgboost import XGBClassifier


# ============================================================
# CONFIG
# ============================================================

DATA_FILE = Path(
    "data/training/ml_dataset_v2.csv"
)

MODEL_DIR = Path(
    "data/training/models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_FILE)

print(
    f"Dataset shape: {df.shape}"
)


# ============================================================
# BASIC CHECKS
# ============================================================

if "layoff_next_90d" not in df.columns:

    raise ValueError(
        "Target column not found."
    )

if "snapshot_year" not in df.columns:

    raise ValueError(
        "snapshot_year is required for time split."
    )


df["layoff_next_90d"] = pd.to_numeric(
    df["layoff_next_90d"],
    errors="coerce"
)

df["snapshot_year"] = pd.to_numeric(
    df["snapshot_year"],
    errors="coerce"
)


# ============================================================
# TIME SPLIT
# ============================================================

train_df = df[
    df["snapshot_year"] <= 2022
].copy()

calibration_df = df[
    df["snapshot_year"] == 2023
].copy()

test_df = df[
    df["snapshot_year"] == 2024
].copy()


print("\n========================================")
print("TIME SPLIT")
print("========================================")

print(
    f"Train       : {len(train_df):,}"
)

print(
    f"Calibration : {len(calibration_df):,}"
)

print(
    f"Test        : {len(test_df):,}"
)


# ============================================================
# TARGET
# ============================================================

TARGET = "layoff_next_90d"

y_train = train_df[TARGET].astype(int)
y_cal = calibration_df[TARGET].astype(int)
y_test = test_df[TARGET].astype(int)


# ============================================================
# FEATURES
# ============================================================

DROP_COLUMNS = [
    TARGET,
    "snapshot_year",
    "snapshot_month"
]

X_train = train_df.drop(
    columns=DROP_COLUMNS,
    errors="ignore"
)

X_cal = calibration_df.drop(
    columns=DROP_COLUMNS,
    errors="ignore"
)

X_test = test_df.drop(
    columns=DROP_COLUMNS,
    errors="ignore"
)


# ============================================================
# ENSURE SAME COLUMNS
# ============================================================

feature_columns = X_train.columns.tolist()

X_cal = X_cal.reindex(
    columns=feature_columns
)

X_test = X_test.reindex(
    columns=feature_columns
)


# ============================================================
# NUMERIC CONVERSION
# ============================================================

for column in feature_columns:

    X_train[column] = pd.to_numeric(
        X_train[column],
        errors="coerce"
    )

    X_cal[column] = pd.to_numeric(
        X_cal[column],
        errors="coerce"
    )

    X_test[column] = pd.to_numeric(
        X_test[column],
        errors="coerce"
    )


# ============================================================
# INFINITE VALUES
# ============================================================

X_train = X_train.replace(
    [np.inf, -np.inf],
    np.nan
)

X_cal = X_cal.replace(
    [np.inf, -np.inf],
    np.nan
)

X_test = X_test.replace(
    [np.inf, -np.inf],
    np.nan
)


# ============================================================
# TRAINING-ONLY IMPUTER
# ============================================================
#
# IMPORTANT:
# The imputer is fitted ONLY on training data.
#
# Calibration and test data are transformed using
# training statistics.
#
# ============================================================

imputer = SimpleImputer(
    strategy="median"
)

X_train_imputed = imputer.fit_transform(
    X_train
)

X_cal_imputed = imputer.transform(
    X_cal
)

X_test_imputed = imputer.transform(
    X_test
)


# ============================================================
# CLASS IMBALANCE
# ============================================================

negative = (y_train == 0).sum()
positive = (y_train == 1).sum()

scale_pos_weight = (
    negative / positive
)

print("\n========================================")
print("CLASS DISTRIBUTION")
print("========================================")

print(
    f"Negative : {negative:,}"
)

print(
    f"Positive : {positive:,}"
)

print(
    f"Scale positive weight: "
    f"{scale_pos_weight:.2f}"
)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    name,
    model,
    X,
    y
):

    probabilities = model.predict_proba(X)[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    precision = precision_score(
        y,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y,
        probabilities
    )

    pr_auc = average_precision_score(
        y,
        probabilities
    )

    brier = brier_score_loss(
        y,
        probabilities
    )

    cm = confusion_matrix(
        y,
        predictions
    )

    print(
        f"\n{name}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1        : {f1:.4f}"
    )

    print(
        f"ROC-AUC   : {roc_auc:.4f}"
    )

    print(
        f"PR-AUC    : {pr_auc:.4f}"
    )

    print(
        f"Brier     : {brier:.4f}"
    )

    print(
        "Confusion Matrix:"
    )

    print(cm)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier": brier
    }


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

print("\n========================================")
print("TRAINING LOGISTIC REGRESSION")
print("========================================")

logistic_model = Pipeline(
    steps=[
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=RANDOM_STATE
            )
        )
    ]
)

logistic_model.fit(
    X_train_imputed,
    y_train
)


# ============================================================
# LOGISTIC EVALUATION
# ============================================================

logistic_results = evaluate_model(
    "Logistic Regression - 2024 Test",
    logistic_model,
    X_test_imputed,
    y_test
)


# ============================================================
# SAVE LOGISTIC MODEL
# ============================================================

joblib.dump(
    {
        "model": logistic_model,
        "imputer": imputer,
        "features": feature_columns
    },
    MODEL_DIR / "logistic_regression_v2.pkl"
)


# ============================================================
# XGBOOST
# ============================================================

print("\n========================================")
print("TRAINING XGBOOST")
print("========================================")

xgb_model = XGBClassifier(

    n_estimators=400,

    max_depth=5,

    learning_rate=0.05,

    subsample=0.8,

    colsample_bytree=0.8,

    objective="binary:logistic",

    eval_metric="aucpr",

    scale_pos_weight=scale_pos_weight,

    random_state=RANDOM_STATE,

    n_jobs=-1
)


xgb_model.fit(
    X_train_imputed,
    y_train
)


# ============================================================
# XGBOOST EVALUATION
# ============================================================

xgb_results = evaluate_model(
    "XGBoost - 2024 Test",
    xgb_model,
    X_test_imputed,
    y_test
)


# ============================================================
# SAVE XGBOOST
# ============================================================

joblib.dump(
    {
        "model": xgb_model,
        "imputer": imputer,
        "features": feature_columns
    },
    MODEL_DIR / "xgboost_v2.pkl"
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({

    "feature": feature_columns,

    "importance": (
        xgb_model.feature_importances_
    )

})

importance = importance.sort_values(
    "importance",
    ascending=False
)

importance.to_csv(
    MODEL_DIR /
    "xgboost_v2_feature_importance.csv",
    index=False
)


# ============================================================
# MODEL COMPARISON
# ============================================================

print("\n========================================")
print("MODEL COMPARISON")
print("========================================")

print(
    f"{'Metric':<15}"
    f"{'Logistic':>12}"
    f"{'XGBoost':>12}"
)

print("-" * 40)

for metric in [
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "brier"
]:

    print(
        f"{metric:<15}"
        f"{logistic_results[metric]:>12.4f}"
        f"{xgb_results[metric]:>12.4f}"
    )


# ============================================================
# TOP FEATURES
# ============================================================

print("\n========================================")
print("TOP 20 XGBOOST FEATURES")
print("========================================")

print(
    importance.head(20).to_string(
        index=False
    )
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n========================================")
print("TRAINING COMPLETED")
print("========================================")

print(
    f"Features used : {len(feature_columns):,}"
)

print(
    f"Train rows    : {len(X_train):,}"
)

print(
    f"Test rows     : {len(X_test):,}"
)

print(
    "\nModels saved in:"
)

print(
    MODEL_DIR
)

print("========================================\n")

