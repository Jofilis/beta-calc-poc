import pandas as pd
import yfinance as yf


def download_prices(ticker, start, end):
    data = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False
    )

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data

def calculate_returns(prices, frequency="daily"):
    close_prices = prices["Close"]

    if hasattr(close_prices, "columns"):
        close_prices = close_prices.iloc[:, 0]

    if frequency == "daily":
        sampled_prices = close_prices

    elif frequency == "weekly":
        sampled_prices = close_prices.resample("W-FRI").last()

    elif frequency == "monthly":
        sampled_prices = close_prices.resample("ME").last()

    else:
        raise ValueError(
            "Frequency must be 'daily', 'weekly', or 'monthly'."
        )

    returns = sampled_prices.pct_change().dropna()

    return returns