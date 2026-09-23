# Corporate Layoff Predictor — Automated Researcher & ML Model

An AI/ML module for the **Corporate Layoff Predictor** major project.
The system collects company information and recent market/news signals, processes historical layoff data, and predicts the probability of a layoff occurring within the **next 90 days**.

## Features

* Historical layoff data processing
* Company-level 90-day layoff labeling
* Historical company and industry layoff features
* Stock-market features using **yFinance**
* Recent company news collection using **NewsAPI**
* Financial-news sentiment analysis using **FinBERT**
* XGBoost-based layoff prediction
* Isotonic probability calibration
* Risk classification:

  * **LOW**
  * **HIGH**
  * **EXTREME HIGH**
* SHAP-based prediction explanations
* Optional Apify-based company research

---

## Project Pipeline

```text
Company Name
     ↓
Automated Researcher
     ↓
Historical Layoff Data
     ↓
yFinance Market Data
     ↓
NewsAPI → FinBERT Sentiment
     ↓
Feature Engineering
     ↓
XGBoost Model
     ↓
Isotonic Calibration
     ↓
90-Day Layoff Probability
     ↓
LOW / HIGH / EXTREME HIGH
     ↓
SHAP Explanation
```

---

## Project Structure

```text
layoff_model/
│
├── collectors/
│   ├── apify_collector.py
│   ├── news_collector.py
│   └── yfinance_collector.py
│
├── prediction/
│   ├── explainer.py
│   └── predict.py
│
├── sentiment/
│   └── sentiment_model.py
│
├── data/
│   └── training/
│       ├── build_labels.py
│       ├── build_sentiment_dataset.py
│       ├── calibrate.py
│       ├── historical_features.py
│       ├── preprocess.py
│       ├── train.py
│       ├── validate_labels.py
│       ├── company_tickers.csv
│       ├── historical_features.csv
│       ├── historical_features_v2.csv
│       ├── historical_sentiment.csv
│       ├── layoff_events.csv
│       ├── layoff_labels.csv
│       ├── ml_dataset.csv
│       ├── ml_dataset_v2.csv
│       │
│       └── models/
│           ├── logistic_regression_v2.pkl
│           ├── xgboost_v2.pkl
│           ├── xgboost_v2_calibrated.pkl
│           └── xgboost_v2_feature_importance.csv
│
├── main.py
├── requirements.txt
└── .gitignore
```

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/Vaibhavtech2006/layoff_model.git
cd layoff_model
```

## 2. Create a virtual environment

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

## 3. Install dependencies

```powershell
pip install -r requirements.txt
```

---

# API Configuration

Create a `.env` file in the project root:

```text
APIFY_API_TOKEN=your_apify_token
NEWS_API_KEY=your_newsapi_key
```

**Never commit `.env` to GitHub.**

The repository already contains `.gitignore` rules for environment files.

---

# Running the Model

The main prediction script is:

```powershell
python prediction/predict.py
```

The script collects the required live data and generates the prediction.

The company name can be provided through the prediction workflow implemented in `predict.py`.

---

# Model

The final model is:

**XGBoost + Isotonic Calibration**

The model was trained using a time-based split to reduce temporal leakage:

```text
Training      → 2020–2022
Calibration   → 2023
Final Test    → 2024
```

The final calibrated model is:

```text
data/training/models/xgboost_v2_calibrated.pkl
```

---

# Prediction Target

The target represents whether a company experiences a layoff within **90 days after the snapshot date**.

```text
0 → No layoff within 90 days
1 → Layoff within 90 days
```

---

# Risk Categories

The calibrated probability is converted into three risk categories:

| Probability      | Risk         |
| ---------------- | ------------ |
| `< 7.70%`        | LOW          |
| `7.70% – 11.17%` | HIGH         |
| `≥ 11.17%`       | EXTREME HIGH |

These thresholds were determined using the 2023 calibration period and then evaluated on the unseen 2024 test period.

---

# Main Features

The model uses several groups of features.

### Company Layoff History

* Previous company layoffs
* Previous employees laid off
* Days since previous layoff
* Company layoffs in the last 30 days
* Company layoffs in the last 90 days
* Company layoffs in the last 365 days

### Industry Signals

* Industry layoffs in the last 30 days
* Industry layoffs in the last 90 days
* Industry layoffs in the last 365 days
* Industry category

### Market Signals

Collected using yFinance:

* 30-day return
* 90-day return
* 30-day volatility
* 90-day volatility
* 30-day drawdown
* 90-day drawdown
* 30-day volume change

### News & Sentiment

Recent company news is collected using NewsAPI and analyzed using FinBERT.

Signals include:

* News count
* Negative news ratio
* Positive news ratio
* Neutral news ratio
* Average sentiment
* Layoff-related keywords
* Restructuring keywords
* Cost-cutting keywords
* Hiring-freeze keywords

---

# Explainability

SHAP is used to identify which model features contributed most to an individual prediction.

Example output:

```text
FINAL PREDICTION

Company       : Microsoft
Layoff Risk   : HIGH
Probability   : 8.51%

Main Factors Increasing Risk:
+ sentiment_age_days
+ days_since_last_layoff
+ layoff_keyword_count
+ company_layoffs_30d

Factors Reducing Risk:
- industry_layoffs_30d
- company_previous_layoffs
- industry_layoffs_90d
- return_90d
```

SHAP explanations describe the model's feature contributions; they should not be interpreted as proof that a specific factor will cause a layoff.

---

# Important Limitations

### 1. Prediction is probabilistic

The model estimates the probability of a layoff occurring within 90 days. It does **not** predict layoffs with certainty.

### 2. Historical market coverage

Only a subset of historical companies have reliable stock-market data because many companies in the historical dataset are private.

### 3. Historical sentiment coverage

Historical news/sentiment data has limited coverage. Therefore, sentiment is treated as an additional research signal rather than a guaranteed predictor.

### 4. Private companies

Companies without a stock ticker may have fewer available market features.

### 5. Live data can change

News, stock prices, company information, and other external signals change over time. Running the model at different times can therefore produce different results.

### 6. Apify data

Apify is used as an auxiliary live company-research source. Its current features are not directly treated as historically trained ML features because equivalent historical snapshots are unavailable.

---

# Team Integration

For integration with the main project, the primary prediction interface is:

```text
prediction/predict.py
```

The main project can provide a company name and consume:

```text
Company
Probability
Risk Category
SHAP Factors
```

Example conceptual output:

```json
{
  "company": "Microsoft",
  "probability": 0.0851,
  "risk": "HIGH",
  "factors_increasing": [],
  "factors_reducing": []
}
```

The exact integration format can be adapted to the main project's backend/API.

---

# Security

API credentials must be stored locally in `.env`.

Do **NOT** commit:

```text
.env
.venv/
data/raw/
```

API keys should never be hard-coded into Python files or uploaded to GitHub.

---

# Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* SHAP
* PyTorch
* Hugging Face Transformers
* FinBERT
* yFinance
* NewsAPI
* Apify
* Joblib

---

## Project Status

**ML Model:** Completed
**Historical Dataset:** Completed
**Feature Engineering:** Completed
**XGBoost Training:** Completed
**Probability Calibration:** Completed
**SHAP Explainability:** Completed
**Live Data Collection:** Implemented
**GitHub Integration:** Completed
