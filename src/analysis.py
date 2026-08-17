import statsmodels.api as sm

from src.data import download_prices, calculate_returns
from src.beta import calculate_beta


def calculate_asset_beta(
    asset_ticker,
    market_ticker,
    start,
    end,
    frequency="daily"
):
    asset_prices = download_prices(
        asset_ticker,
        start,
        end
    )

    market_prices = download_prices(
        market_ticker,
        start,
        end
    )

    return calculate_asset_beta_from_prices(
        asset_prices,
        market_prices,
        frequency=frequency
    )


def compare_frequencies(asset_ticker, market_ticker, start, end):
    results = {}

    for frequency in ["daily", "weekly", "monthly"]:
        try:
            results[frequency] = calculate_asset_beta(
                asset_ticker,
                market_ticker,
                start,
                end,
                frequency=frequency
            )
        except ValueError as e:
            print(f"⚠️ Amostra insuficiente ou falha ao calcular frequência '{frequency}': {e}")

    return results


def compare_periods(
    asset_ticker,
    market_ticker,
    periods,
    frequency="daily"
):
    results = {}

    for period_name, start, end in periods:
        results[period_name] = calculate_asset_beta(
            asset_ticker,
            market_ticker,
            start,
            end,
            frequency=frequency
        )

    return results


def calculate_beta_from_prices(
    asset_prices,
    market_prices,
    frequency="daily"
):
    asset_returns = calculate_returns(
        asset_prices,
        frequency=frequency
    )

    market_returns = calculate_returns(
        market_prices,
        frequency=frequency
    )

    combined_returns = (
        asset_returns.to_frame(name="asset")
        .join(
            market_returns.to_frame(name="market"),
            how="inner"
        )
        .dropna()
    )

    asset = combined_returns["asset"]
    market = combined_returns["market"]

    return calculate_beta(asset, market)


def calculate_asset_beta_from_prices(
    asset_prices,
    market_prices,
    frequency="daily"
):
    asset_returns = calculate_returns(
        asset_prices,
        frequency=frequency
    )

    market_returns = calculate_returns(
        market_prices,
        frequency=frequency
    )

    combined_returns = (
        asset_returns.to_frame(name="asset")
        .join(
            market_returns.to_frame(name="market"),
            how="inner"
        )
        .dropna()
    )

    asset = combined_returns["asset"]
    market = combined_returns["market"]

    beta_covariance = calculate_beta(
        asset,
        market
    )

    X = sm.add_constant(market)

    model = sm.OLS(
        asset,
        X
    ).fit()

    beta_regression = model.params["market"]
    alpha = model.params["const"]
    r_squared = model.rsquared

    return {
        "beta": beta_covariance,
        "beta_regression": beta_regression,
        "alpha": alpha,
        "r_squared": r_squared,
        "observations": len(combined_returns),
    }


if __name__ == "__main__":
    result = calculate_asset_beta(
        "PETR4.SA",
        "^BVSP",
        "2025-01-01",
        "2025-12-31"
    )

    print(f"Beta: {result['beta']:.4f}")
    print(f"Beta (regression): {result['beta_regression']:.4f}")
    print(f"Alpha: {result['alpha']:.6f}")
    print(f"R²: {result['r_squared']:.4f}")
    print(f"Observations: {result['observations']}")