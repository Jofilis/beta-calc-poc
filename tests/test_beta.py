import pytest

from src.beta import calculate_beta
from src.beta import calculate_beta_regression


asset_returns = [0.02, 0.04, -0.01, 0.03, 0.05]
market_returns = [0.01, 0.02, -0.005, 0.015, 0.025]


def test_beta_covariance():
    beta = calculate_beta(asset_returns, market_returns)

    assert beta == pytest.approx(2.0)


def test_beta_regression():
    beta = calculate_beta_regression(
        asset_returns,
        market_returns
    )

    assert beta == pytest.approx(2.0)


def test_beta_methods_are_equivalent():
    beta_covariance = calculate_beta(
        asset_returns,
        market_returns
    )

    beta_regression = calculate_beta_regression(
        asset_returns,
        market_returns
    )

    assert beta_covariance == pytest.approx(beta_regression)


def test_returns_must_have_same_length():
    with pytest.raises(ValueError):
        calculate_beta(
            [0.02, 0.04, 0.01],
            [0.01, 0.02]
        )


def test_at_least_two_observations_required():
    with pytest.raises(ValueError):
        calculate_beta(
            [0.02],
            [0.01]
        )


def test_market_must_have_variance():
    with pytest.raises(ValueError):
        calculate_beta(
            [0.02, 0.03, 0.04],
            [0.01, 0.01, 0.01]
        )