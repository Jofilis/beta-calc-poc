import numpy as np
import pytest
import pandas as pd

from src.risk import (
    calculate_volatility,
    calculate_correlation,
    calculate_residual_returns,
    calculate_idiosyncratic_risk,
)

from src.risk import calculate_variance_decomposition


def test_calculate_volatility():

    returns = np.array([
        0.01,
        -0.02,
        0.03,
        0.02,
        -0.01
    ])

    result = calculate_volatility(
        returns
    )

    expected = np.std(
        returns,
        ddof=1
    )

    assert result == pytest.approx(
        expected
    )


def test_calculate_volatility_requires_two_observations():

    with pytest.raises(ValueError):

        calculate_volatility(
            [0.01]
        )


def test_calculate_correlation():

    market_returns = np.array([
        0.01,
        -0.02,
        0.03,
        0.02,
        -0.01
    ])

    asset_returns = (
        0.001
        + 1.5 * market_returns
    )

    result = calculate_correlation(
        asset_returns,
        market_returns
    )

    assert result == pytest.approx(
        1.0
    )


def test_calculate_correlation_requires_matching_lengths():

    asset_returns = [
        0.01,
        0.02,
        0.03
    ]

    market_returns = [
        0.01,
        0.02
    ]

    with pytest.raises(ValueError):

        calculate_correlation(
            asset_returns,
            market_returns
        )


def test_calculate_correlation_requires_non_zero_variance():

    asset_returns = [
        0.01,
        0.02,
        0.03
    ]

    market_returns = [
        0.01,
        0.01,
        0.01
    ]

    with pytest.raises(ValueError):

        calculate_correlation(
            asset_returns,
            market_returns
        )


def test_calculate_residual_returns():

    market_returns = np.array([
        0.01,
        -0.02,
        0.03,
        0.02,
        -0.01
    ])

    alpha = 0.001
    beta = 1.5

    asset_returns = (
        alpha
        + beta * market_returns
    )

    residuals = calculate_residual_returns(
        asset_returns,
        market_returns,
        alpha,
        beta
    )

    assert np.allclose(
        residuals,
        0.0
    )


def test_calculate_idiosyncratic_risk():

    market_returns = np.array([
        0.01,
        -0.02,
        0.03,
        0.02,
        -0.01
    ])

    alpha = 0.001
    beta = 1.5

    asset_returns = (
        alpha
        + beta * market_returns
    )

    result = calculate_idiosyncratic_risk(
        asset_returns,
        market_returns,
        alpha,
        beta
    )

    assert result == pytest.approx(
        0.0
    )

def test_variance_decomposition():
    dates = pd.date_range(
        "2021-01-01",
        periods=100,
        freq="B"
    )

    market_returns = pd.Series(
        [
            0.01,
            -0.005,
            0.003,
            0.008,
            -0.004,
            0.006,
            -0.002
        ] * 14 + [0.01, -0.005],
        index=dates
    )

    asset_returns = (
        0.001
        + 1.5 * market_returns
    )

    result = calculate_variance_decomposition(
        asset_returns,
        market_returns
    )

    assert result["total_variance"] > 0

    assert result["systematic_variance"] > 0

    assert result["residual_variance"] >= 0

    assert result["systematic_percentage"] > 0

    assert result["residual_percentage"] >= 0

    assert result["systematic_percentage"] + \
           result["residual_percentage"] == pytest.approx(
               1.0,
               abs=0.000001
           )

    assert result["r_squared"] == pytest.approx(
        result["systematic_percentage"],
        abs=0.000001
    )


def test_variance_decomposition_consistency():
    market_returns = pd.Series(
        [0.01, -0.005, 0.003, 0.008, -0.004]
    )

    asset_returns = (
        0.001
        + 1.5 * market_returns
    )

    result = calculate_variance_decomposition(
        asset_returns,
        market_returns
    )

    reconstructed_variance = (
        result["systematic_variance"]
        + result["residual_variance"]
    )

    assert reconstructed_variance == pytest.approx(
        result["total_variance"],
        abs=0.000001
    )