import pandas as pd
import statsmodels.api as sm
import numpy as np


def validate_risk_returns(asset_returns, market_returns):
    """
    Validate asset and market return series for risk calculations.
    """

    if len(asset_returns) != len(market_returns):
        raise ValueError(
            "Asset and market returns must have the same length."
        )

    if len(asset_returns) < 2:
        raise ValueError(
            "At least two return observations are required."
        )


def calculate_volatility(returns):
    """
    Calculate historical volatility of returns.

    Volatility = standard deviation of returns.
    """

    returns = np.asarray(returns, dtype=float)

    if len(returns) < 2:
        raise ValueError(
            "At least two return observations are required."
        )

    return np.std(
        returns,
        ddof=1
    )


def calculate_correlation(asset_returns, market_returns):
    """
    Calculate Pearson correlation between asset and market returns.
    """

    validate_risk_returns(
        asset_returns,
        market_returns
    )

    asset_returns = np.asarray(
        asset_returns,
        dtype=float
    )

    market_returns = np.asarray(
        market_returns,
        dtype=float
    )

    asset_std = np.std(
        asset_returns,
        ddof=1
    )

    market_std = np.std(
        market_returns,
        ddof=1
    )

    if asset_std == 0 or market_std == 0:
        raise ValueError(
            "Asset and market returns must have non-zero variance."
        )

    return np.corrcoef(
        asset_returns,
        market_returns
    )[0, 1]


def calculate_residual_returns(
    asset_returns,
    market_returns,
    alpha,
    beta
):
    """
    Calculate residual returns from the market model.

    R_i = alpha + beta * R_m + epsilon

    epsilon = R_i - (alpha + beta * R_m)
    """

    validate_risk_returns(
        asset_returns,
        market_returns
    )

    asset_returns = np.asarray(
        asset_returns,
        dtype=float
    )

    market_returns = np.asarray(
        market_returns,
        dtype=float
    )

    predicted_returns = (
        alpha
        + beta * market_returns
    )

    residual_returns = (
        asset_returns
        - predicted_returns
    )

    return residual_returns


def calculate_idiosyncratic_risk(
    asset_returns,
    market_returns,
    alpha,
    beta
):
    """
    Calculate idiosyncratic risk as the standard deviation
    of the OLS residual returns.
    """

    residual_returns = calculate_residual_returns(
        asset_returns,
        market_returns,
        alpha,
        beta
    )

    return np.std(
        residual_returns,
        ddof=1
    )

def calculate_variance_decomposition(
    asset_returns,
    market_returns
):
    """
    Decompose the variance of asset returns into:

    - Systematic variance
    - Residual variance

    Based on the OLS market model:

        Ri = alpha + beta * Rm + epsilon

    Systematic variance:
        beta² * Var(Rm)

    Residual variance:
        Var(epsilon)

    Total variance:
        Var(Ri)
    """

    asset_returns = pd.Series(asset_returns).dropna()
    market_returns = pd.Series(market_returns).dropna()

    combined = pd.concat(
        [asset_returns, market_returns],
        axis=1,
        join="inner"
    ).dropna()

    if len(combined) < 2:
        raise ValueError(
            "At least two observations are required."
        )

    asset = combined.iloc[:, 0]
    market = combined.iloc[:, 1]

    X = sm.add_constant(market)

    model = sm.OLS(
        asset,
        X
    ).fit()

    beta = model.params.iloc[1]

    total_variance = asset.var(ddof=1)

    systematic_variance = (
        beta ** 2
        * market.var(ddof=1)
    )

    residual_variance = model.resid.var(ddof=1)

    systematic_percentage = (
        systematic_variance
        / total_variance
    )

    residual_percentage = (
        residual_variance
        / total_variance
    )

    return {
        "total_variance": total_variance,
        "systematic_variance": systematic_variance,
        "residual_variance": residual_variance,
        "systematic_percentage": systematic_percentage,
        "residual_percentage": residual_percentage,
        "r_squared": model.rsquared,
        "beta": beta,
        "observations": len(combined),
    }