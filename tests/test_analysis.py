import numpy as np
import pandas as pd
import pytest

import src.analysis as analysis


def create_test_prices():
    dates = pd.date_range(
        "2021-01-01",
        periods=750,
        freq="B"
    )

    market_returns = np.array(
        [0.01, -0.005, 0.003, 0.008, -0.004, 0.006, -0.002]
        * 108
    )[:750]

    asset_returns = (
        0.001
        + 1.5 * market_returns
    )

    market_prices = 100 * np.cumprod(
        1 + market_returns
    )

    asset_prices = 100 * np.cumprod(
        1 + asset_returns
    )

    market = pd.DataFrame(
        {"Close": market_prices},
        index=dates
    )

    asset = pd.DataFrame(
        {"Close": asset_prices},
        index=dates
    )

    return asset, market


@pytest.fixture
def test_data():
    return create_test_prices()


@pytest.fixture
def mock_download_prices(monkeypatch, test_data):
    asset_prices, market_prices = test_data

    def fake_download_prices(ticker, start, end):
        if ticker == "ASSET":
            prices = asset_prices
        elif ticker == "MARKET":
            prices = market_prices
        else:
            raise ValueError(f"Unknown ticker: {ticker}")

        return prices.loc[
            pd.Timestamp(start):pd.Timestamp(end)
        ].copy()

    monkeypatch.setattr(
        analysis,
        "download_prices",
        fake_download_prices
    )


def test_calculate_asset_beta(
    mock_download_prices
):
    result = analysis.calculate_asset_beta(
        "ASSET",
        "MARKET",
        "2021-01-01",
        "2023-11-30"
    )

    assert result["beta"] == pytest.approx(
        1.5,
        abs=0.02
    )

    assert result["beta_regression"] == pytest.approx(
        1.5,
        abs=0.02
    )

    assert result["alpha"] == pytest.approx(
        0.001,
        abs=0.00001
    )

    assert result["r_squared"] > 0.99

    assert result["observations"] > 700


def test_calculate_asset_beta_weekly(
    mock_download_prices
):
    result = analysis.calculate_asset_beta(
        "ASSET",
        "MARKET",
        "2021-01-01",
        "2023-11-30",
        frequency="weekly"
    )

    assert result["beta"] > 0
    assert result["beta_regression"] > 0
    assert result["observations"] > 50


def test_calculate_asset_beta_monthly(
    mock_download_prices
):
    result = analysis.calculate_asset_beta(
        "ASSET",
        "MARKET",
        "2021-01-01",
        "2023-11-30",
        frequency="monthly"
    )

    assert result["beta"] > 0
    assert result["beta_regression"] > 0
    assert result["observations"] > 20


def test_compare_frequencies(
    mock_download_prices
):
    results = analysis.compare_frequencies(
        "ASSET",
        "MARKET",
        "2021-01-01",
        "2023-11-30"
    )

    assert set(results.keys()) == {
        "daily",
        "weekly",
        "monthly"
    }

    for frequency in [
        "daily",
        "weekly",
        "monthly"
    ]:
        assert results[frequency]["beta"] is not None
        assert results[frequency]["beta_regression"] is not None
        assert results[frequency]["alpha"] is not None
        assert results[frequency]["r_squared"] is not None
        assert results[frequency]["observations"] > 0


def test_compare_periods(
    mock_download_prices
):
    periods = [
        (
            "1_year",
            "2022-01-01",
            "2022-12-31"
        ),
        (
            "3_years",
            "2021-01-01",
            "2023-12-31"
        )
    ]

    results = analysis.compare_periods(
        "ASSET",
        "MARKET",
        periods,
        frequency="daily"
    )

    assert set(results.keys()) == {
        "1_year",
        "3_years"
    }

    assert results["1_year"]["observations"] > 200
    assert results["3_years"]["observations"] > 700

    assert results["1_year"]["beta"] > 0
    assert results["3_years"]["beta"] > 0


def test_calculate_beta_from_prices(
    test_data
):
    asset_prices, market_prices = test_data

    beta = analysis.calculate_beta_from_prices(
        asset_prices,
        market_prices
    )

    assert beta == pytest.approx(
        1.5,
        abs=0.02
    )


def test_calculate_asset_beta_from_prices(
    test_data
):
    asset_prices, market_prices = test_data

    result = analysis.calculate_asset_beta_from_prices(
        asset_prices,
        market_prices
    )

    assert result["beta"] == pytest.approx(
        1.5,
        abs=0.02
    )

    assert result["beta_regression"] == pytest.approx(
        1.5,
        abs=0.02
    )

    assert result["alpha"] == pytest.approx(
        0.001,
        abs=0.00001
    )

    assert result["r_squared"] > 0.99

    assert result["observations"] > 700