# ============================================================
# SHAP Explainability
# Explains why the model predicts a company's layoff risk
# ============================================================

import os
import joblib
import pandas as pd
import numpy as np
import shap


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

MODEL_PATH = "data/training/models/xgboost_final_calibrated.pkl"


# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

def load_model():

    saved_model = joblib.load(MODEL_PATH)

    model = saved_model["model"]
    imputer = saved_model["imputer"]
    calibrator = saved_model["calibrator"]
    features = saved_model["features"]

    return model, imputer, calibrator, features


# ------------------------------------------------------------
# Generate SHAP explanation
# ------------------------------------------------------------

def explain_prediction(input_data, top_n=5):

    """
    input_data:
        Dictionary containing model features.

    Example:
        {
            "company_previous_layoffs": 2,
            "company_previous_laid_off": 500,
            "days_since_last_layoff": 120,
            ...
        }

    Returns:
        probability
        top positive factors
        top negative factors
    """

    model, imputer, calibrator, features = load_model()


    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    X = pd.DataFrame([input_data])

    # Make sure every trained feature exists
    for feature in features:

        if feature not in X.columns:
            X[feature] = 0


    # Keep exact training feature order
    X = X[features]


    # --------------------------------------------------------
    # Apply training imputer
    # --------------------------------------------------------

    X_processed = imputer.transform(X)


    # --------------------------------------------------------
    # Raw model probability
    # --------------------------------------------------------

    raw_probability = model.predict_proba(
        X_processed
    )[:, 1][0]


    # --------------------------------------------------------
    # Calibrated probability
    # --------------------------------------------------------

    probability = calibrator.predict(
        [raw_probability]
    )[0]


    # --------------------------------------------------------
    # SHAP
    # --------------------------------------------------------

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(
        X_processed
    )


    # --------------------------------------------------------
    # Handle SHAP output
    # --------------------------------------------------------

    if isinstance(shap_values, list):

        shap_values = shap_values[1]

    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 2:

        shap_values = shap_values[0]


    # --------------------------------------------------------
    # Create explanation table
    # --------------------------------------------------------

    explanation = pd.DataFrame({
        "feature": features,
        "value": X_processed[0],
        "shap_value": shap_values
    })


    # --------------------------------------------------------
    # Positive SHAP = increases layoff risk
    # Negative SHAP = decreases layoff risk
    # --------------------------------------------------------

    positive = (
        explanation[
            explanation["shap_value"] > 0
        ]
        .sort_values(
            "shap_value",
            ascending=False
        )
        .head(top_n)
    )


    negative = (
        explanation[
            explanation["shap_value"] < 0
        ]
        .sort_values(
            "shap_value",
            ascending=True
        )
        .head(top_n)
    )


    return {
        "probability": float(probability),
        "raw_probability": float(raw_probability),
        "positive_factors": positive,
        "negative_factors": negative,
        "all_features": explanation
    }


# ------------------------------------------------------------
# Test
# ------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("SHAP EXPLAINABILITY TEST")
    print("=" * 60)


    # --------------------------------------------------------
    # Example company data
    #
    # This is ONLY a test input.
    # It is not an actual company's data.
    # --------------------------------------------------------

    example_data = {

        "company_previous_layoffs": 2,

        "company_previous_laid_off": 500,

        "days_since_last_layoff": 120,

        "company_layoffs_30d": 1,

        "company_layoffs_90d": 2,

        "company_layoffs_365d": 3,

        "industry_layoffs_30d": 10,

        "industry_layoffs_90d": 25,

        "industry_layoffs_365d": 80,

        "return_30d": -0.05,

        "return_90d": -0.12,

        "volatility_30d": 0.35,

        "volatility_90d": 0.40,

        "drawdown_30d": -0.08,

        "drawdown_90d": -0.20,

        "volume_change_30d": 0.10,

        "snapshot_year": 2024,

        "snapshot_month": 5,

        "return_30d_available": 1,

        "return_90d_available": 1,

        "volatility_30d_available": 1,

        "volatility_90d_available": 1,

        "drawdown_30d_available": 1,

        "drawdown_90d_available": 1,

        "volume_change_30d_available": 1,
    }


    # --------------------------------------------------------
    # Industry
    # --------------------------------------------------------

    example_data["industry_Data"] = 1


    result = explain_prediction(
        example_data,
        top_n=5
    )


    # --------------------------------------------------------
    # Print result
    # --------------------------------------------------------

    print("\nProbability:")
    print(
        f"{result['probability'] * 100:.2f}%"
    )


    print("\nRaw probability:")
    print(
        f"{result['raw_probability'] * 100:.2f}%"
    )


    print("\n" + "-" * 60)
    print("FACTORS INCREASING LAYOFF RISK")
    print("-" * 60)

    for _, row in result["positive_factors"].iterrows():

        print(
            f"{row['feature']:<40} "
            f"SHAP: {row['shap_value']:.4f}"
        )


    print("\n" + "-" * 60)
    print("FACTORS REDUCING LAYOFF RISK")
    print("-" * 60)

    for _, row in result["negative_factors"].iterrows():

        print(
            f"{row['feature']:<40} "
            f"SHAP: {row['shap_value']:.4f}"
        )


    print("\n" + "=" * 60)
    print("SHAP TEST COMPLETE")
    print("=" * 60)