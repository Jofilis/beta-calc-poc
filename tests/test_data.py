import pandas as pd
import pytest

import src.data as data


def test_download_prices(monkeypatch):
    expected_df = pd.DataFrame(
        {"Close": [30.0, 31.0]},
        index=pd.date_range("2025-01-01", periods=2)
    )

    monkeypatch.setattr(
        data,
        "yf",
        type(
            "FakeYFinance",
            (),
            {
                "download": staticmethod(
                    lambda ticker, start, end, auto_adjust, progress:
                    expected_df
                )
            }
        )
    )

    result = data.download_prices(
        "PETR4.SA",
        "2025-01-01",
        "2025-12-31"
    )

    pd.testing.assert_frame_equal(result, expected_df)


def test_calculate_daily_returns():
    prices = pd.DataFrame(
        {"Close": [100.0, 105.0, 102.0]}
    )

    returns = data.calculate_returns(
        prices,
        frequency="daily"
    )

    assert len(returns) == 2

    assert returns.iloc[0] == pytest.approx(0.05)

    assert returns.iloc[1] == pytest.approx(
        -0.02857142857142858
    )


def test_calculate_weekly_returns():
    dates = pd.date_range(
        "2025-01-01",
        periods=14,
        freq="D"
    )

    prices = pd.DataFrame(
        {"Close": range(100, 114)},
        index=dates
    )

    returns = data.calculate_returns(
        prices,
        frequency="weekly"
    )

    # Weekly prices:
    # 2025-01-03 -> 102
    # 2025-01-10 -> 109
    # 2025-01-14 -> 113
    #
    # Therefore:
    # 109 / 102 - 1
    # 113 / 109 - 1

    expected_returns = pd.Series(
        [
            109 / 102 - 1,
            113 / 109 - 1
        ]
    )

    assert len(returns) == 2

    for actual, expected in zip(
        returns.tolist(),
        expected_returns.tolist()
    ):
        assert actual == pytest.approx(expected)


def test_calculate_monthly_returns():
    dates = pd.date_range(
        "2025-01-01",
        periods=90,
        freq="D"
    )

    prices = pd.DataFrame(
        {"Close": range(100, 190)},
        index=dates
    )

    returns = data.calculate_returns(
        prices,
        frequency="monthly"
    )

    # Last prices of each month:
    # January   -> 130
    # February  -> 158
    # March     -> 189
    #
    # Therefore:
    # 158 / 130 - 1
    # 189 / 158 - 1

    expected_returns = pd.Series(
        [
            158 / 130 - 1,
            189 / 158 - 1
        ]
    )

    assert len(returns) == 2

    for actual, expected in zip(
        returns.tolist(),
        expected_returns.tolist()
    ):
        assert actual == pytest.approx(expected)


def test_invalid_frequency():
    prices = pd.DataFrame(
        {"Close": [100.0, 105.0, 102.0]}
    )

    with pytest.raises(ValueError):
        data.calculate_returns(
            prices,
            frequency="yearly"
        )