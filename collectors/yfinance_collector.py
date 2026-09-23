import os
import json
import yfinance as yf
import pandas as pd


def get_market_features(ticker_symbol):
    """
    Collect market and financial features using yfinance.
    """

    print(f"Collecting yfinance data for {ticker_symbol}...")

    ticker = yf.Ticker(ticker_symbol)

    # --------------------------------
    # Historical price data
    # --------------------------------

    history = ticker.history(period="1y")

    if history.empty:
        raise ValueError(
            f"No historical data found for {ticker_symbol}"
        )

    close = history["Close"]
    volume = history["Volume"]

    # --------------------------------
    # Returns
    # --------------------------------

    return_7d = calculate_return(close, 7)
    return_30d = calculate_return(close, 30)
    return_90d = calculate_return(close, 90)

    # --------------------------------
    # Volatility
    # --------------------------------

    daily_returns = close.pct_change()

    volatility_30d = (
        daily_returns.tail(30).std()
        * (252 ** 0.5)
    )

    volatility_90d = (
        daily_returns.tail(90).std()
        * (252 ** 0.5)
    )

    # --------------------------------
    # Drawdown
    # --------------------------------

    drawdown_30d = calculate_drawdown(
        close.tail(30)
    )

    drawdown_90d = calculate_drawdown(
        close.tail(90)
    )

    # --------------------------------
    # Volume change
    # --------------------------------

    volume_change_30d = calculate_volume_change(
        volume
    )

    # --------------------------------
    # Current price
    # --------------------------------

    current_price = float(close.iloc[-1])

    # --------------------------------
    # Company information
    # --------------------------------

    try:
        info = ticker.info
    except Exception:
        info = {}

    market_cap = info.get("marketCap")

    # --------------------------------
    # Financial statements
    # --------------------------------

    financial_features = get_financial_features(
        ticker
    )

    # --------------------------------
    # Final feature dictionary
    # --------------------------------

    features = {

        # Market
        "ticker": ticker_symbol,
        "current_price": current_price,
        "market_cap": market_cap,

        "return_7d": return_7d,
        "return_30d": return_30d,
        "return_90d": return_90d,

        "volatility_30d": volatility_30d,
        "volatility_90d": volatility_90d,

        "drawdown_30d": drawdown_30d,
        "drawdown_90d": drawdown_90d,

        "volume_change_30d": volume_change_30d,

        # Financial
        **financial_features
    }

    return features


def calculate_return(close, days):
    """
    Calculate percentage return over given number of days.
    """

    if len(close) <= days:
        return None

    old_price = close.iloc[-days - 1]
    current_price = close.iloc[-1]

    return float(
        (current_price / old_price) - 1
    )


def calculate_drawdown(close):
    """
    Calculate maximum drawdown.
    """

    if len(close) == 0:
        return None

    running_max = close.cummax()

    drawdown = (
        close - running_max
    ) / running_max

    return float(drawdown.min())


def calculate_volume_change(volume):
    """
    Compare recent 30-day average volume
    against previous 30-day average volume.
    """

    if len(volume) < 60:
        return None

    recent_avg = volume.tail(30).mean()

    previous_avg = (
        volume.iloc[-60:-30].mean()
    )

    if previous_avg == 0:
        return None

    return float(
        (recent_avg / previous_avg) - 1
    )


def get_financial_features(ticker):
    """
    Extract important financial indicators.
    """

    features = {
        "revenue": None,
        "revenue_growth": None,
        "net_income": None,
        "profit_margin": None,
        "total_debt": None,
        "cash": None,
        "debt_to_equity": None
    }

    # --------------------------------
    # Income statement
    # --------------------------------

    try:
        income = ticker.quarterly_income_stmt

        if not income.empty:

            latest_column = income.columns[0]

            revenue = get_statement_value(
                income,
                "Total Revenue",
                latest_column
            )

            net_income = get_statement_value(
                income,
                "Net Income",
                latest_column
            )

            features["revenue"] = revenue
            features["net_income"] = net_income

            # Profit margin
            if (
                revenue is not None
                and net_income is not None
                and revenue != 0
            ):
                features["profit_margin"] = (
                    net_income / revenue
                )

            # Revenue growth
            if len(income.columns) >= 2:

                previous_column = income.columns[1]

                previous_revenue = get_statement_value(
                    income,
                    "Total Revenue",
                    previous_column
                )

                if (
                    revenue is not None
                    and previous_revenue is not None
                    and previous_revenue != 0
                ):
                    features["revenue_growth"] = (
                        revenue / previous_revenue
                    ) - 1

    except Exception as e:

        print(
            f"Financial statement warning: {e}"
        )

    # --------------------------------
    # Balance sheet
    # --------------------------------

    try:
        balance = ticker.quarterly_balance_sheet

        if not balance.empty:

            latest_column = balance.columns[0]

            features["total_debt"] = get_statement_value(
                balance,
                "Total Debt",
                latest_column
            )

            features["cash"] = get_statement_value(
                balance,
                "Cash Cash Equivalents And Short Term Investments",
                latest_column
            )

    except Exception as e:

        print(
            f"Balance sheet warning: {e}"
        )

    # --------------------------------
    # Debt-to-equity
    # --------------------------------

    try:

        info = ticker.info

        debt_to_equity = info.get(
            "debtToEquity"
        )

        if debt_to_equity is not None:
            features["debt_to_equity"] = (
                debt_to_equity / 100
            )

    except Exception:
        pass

    return features


def get_statement_value(
    statement,
    row_name,
    column
):
    """
    Safely retrieve a financial statement value.
    """

    if row_name not in statement.index:
        return None

    value = statement.loc[
        row_name,
        column
    ]

    if pd.isna(value):
        return None

    return float(value)