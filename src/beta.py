import numpy as np
import statsmodels.api as sm
import pandas as pd

def validate_returns(asset_returns, market_returns):
    if len(asset_returns) != len(market_returns):
        raise ValueError(
            "Asset and market returns must have the same length."
        )

    if len(asset_returns) < 2:
        raise ValueError(
            "At least two return observations are required."
        )

    if np.var(market_returns, ddof=1) == 0:
        raise ValueError(
            "Market returns must have non-zero variance."
        )


def calculate_beta(asset_returns, market_returns):
    validate_returns(asset_returns, market_returns)

    asset_returns = np.array(asset_returns)
    market_returns = np.array(market_returns)

    covariance = np.cov(
        asset_returns,
        market_returns,
        ddof=1
    )[0, 1]

    market_variance = np.var(
        market_returns,
        ddof=1
    )

    beta = covariance / market_variance

    return beta


def calculate_beta_regression(asset_returns, market_returns):
    validate_returns(asset_returns, market_returns)

    X = pd.DataFrame({
        "market_returns": market_returns
    })

    X = sm.add_constant(X)

    model = sm.OLS(
        asset_returns,
        X
    ).fit()

    beta = model.params["market_returns"]

    return beta