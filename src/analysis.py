import statsmodels.api as sm

from src.data import (
    download_prices,
    calculate_returns
)

from src.beta import calculate_beta

from src.capm import (
    calculate_expected_return,
    compare_observed_expected_return,
    interpret_capm_difference
)

from src.risk import (
    calculate_volatility,
    calculate_correlation,
    calculate_idiosyncratic_risk,
    calculate_variance_decomposition
)


def calculate_asset_beta(
    asset_ticker,
    market_ticker,
    start,
    end,
    frequency="daily",
    risk_free_rate=None,
    market_expected_return=None
):
    """
    Download historical prices and calculate the complete
    beta and risk analysis for an asset against a market benchmark.
    """

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
        frequency=frequency,
        risk_free_rate=risk_free_rate,
        market_expected_return=market_expected_return
    )


def compare_frequencies(
    asset_ticker,
    market_ticker,
    start,
    end
):
    """
    Compare beta and risk metrics across daily, weekly,
    and monthly frequencies.
    """

    results = {}

    for frequency in [
        "daily",
        "weekly",
        "monthly"
    ]:
        try:
            results[frequency] = calculate_asset_beta(
                asset_ticker,
                market_ticker,
                start,
                end,
                frequency=frequency
            )

        except ValueError as e:
            print(
                f"⚠️ Amostra insuficiente ou falha ao calcular "
                f"frequência '{frequency}': {e}"
            )

    return results


def compare_periods(
    asset_ticker,
    market_ticker,
    periods,
    frequency="daily"
):
    """
    Compare beta across different historical periods.
    """

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
    """
    Calculate beta directly from price data.
    """

    asset_returns = calculate_returns(
        asset_prices,
        frequency=frequency
    )

    market_returns = calculate_returns(
        market_prices,
        frequency=frequency
    )

    combined_returns = (
        asset_returns
        .to_frame(name="asset")
        .join(
            market_returns.to_frame(name="market"),
            how="inner"
        )
        .dropna()
    )

    asset = combined_returns["asset"]
    market = combined_returns["market"]

    return calculate_beta(
        asset,
        market
    )


def calculate_asset_beta_from_prices(
    asset_prices,
    market_prices,
    frequency="daily",
    risk_free_rate=None,
    market_expected_return=None
):
    """
    Calculate the complete quantitative risk analysis
    directly from historical price data.
    """

    # --------------------------------------------------
    # 1. Calculate returns
    # --------------------------------------------------

    asset_returns = calculate_returns(
        asset_prices,
        frequency=frequency
    )

    market_returns = calculate_returns(
        market_prices,
        frequency=frequency
    )

    # --------------------------------------------------
    # 2. Align asset and market observations
    # --------------------------------------------------

    combined_returns = (
        asset_returns
        .to_frame(name="asset")
        .join(
            market_returns.to_frame(name="market"),
            how="inner"
        )
        .dropna()
    )

    if combined_returns.empty:
        raise ValueError(
            "No overlapping return observations were found "
            "for the selected asset and market."
        )

    asset = combined_returns["asset"]
    market = combined_returns["market"]

    # --------------------------------------------------
    # 3. Observed cumulative return
    # --------------------------------------------------

    observed_return = (
        (1 + asset).prod() - 1
    )

    # --------------------------------------------------
    # 4. Beta through covariance / variance
    # --------------------------------------------------

    beta_covariance = calculate_beta(
        asset,
        market
    )

    # --------------------------------------------------
    # 5. OLS regression
    #
    # Ri = alpha + beta * Rm + epsilon
    # --------------------------------------------------

    X = sm.add_constant(market)

    model = sm.OLS(
        asset,
        X
    ).fit()

    beta_regression = model.params["market"]
    alpha = model.params["const"]
    r_squared = model.rsquared

    # --------------------------------------------------
    # 6. Volatility
    # --------------------------------------------------

    asset_volatility = calculate_volatility(
        asset
    )

    market_volatility = calculate_volatility(
        market
    )

    # --------------------------------------------------
    # 7. Correlation
    # --------------------------------------------------

    correlation = calculate_correlation(
        asset,
        market
    )

    # --------------------------------------------------
    # 8. Idiosyncratic risk
    # --------------------------------------------------

    idiosyncratic_risk = calculate_idiosyncratic_risk(
        asset,
        market,
        alpha,
        beta_regression
    )

    # --------------------------------------------------
    # 9. Variance decomposition
    # --------------------------------------------------

    variance_decomposition = calculate_variance_decomposition(
        asset,
        market
    )

    # --------------------------------------------------
    # 10. CAPM expected return
    #
    # CAPM is calculated only when external
    # Rf and E(Rm) are supplied.
    # --------------------------------------------------

    expected_return = None

    if (
        risk_free_rate is not None
        and market_expected_return is not None
    ):
        expected_return = calculate_expected_return(
            beta_regression,
            market_expected_return,
            risk_free_rate
        )

    # --------------------------------------------------
    # 11. Difference between observed and CAPM return
    # --------------------------------------------------

    capm_difference = None

    if expected_return is not None:
        capm_difference = compare_observed_expected_return(
            observed_return,
            expected_return
        )

    # --------------------------------------------------
    # 12. CAPM interpretation
    # --------------------------------------------------

    capm_interpretation = None

    if capm_difference is not None:
        capm_interpretation = interpret_capm_difference(
            capm_difference
        )

    # --------------------------------------------------
    # 13. Return complete analysis
    # --------------------------------------------------

    return {
        "beta": beta_covariance,
        "beta_regression": beta_regression,
        "alpha": alpha,
        "r_squared": r_squared,
        "observations": len(combined_returns),

        "asset_volatility": asset_volatility,
        "market_volatility": market_volatility,
        "correlation": correlation,
        "idiosyncratic_risk": idiosyncratic_risk,

        "expected_return": expected_return,
        "observed_return": observed_return,
        "capm_difference": capm_difference,
        "capm_interpretation": capm_interpretation,

        "variance_decomposition": variance_decomposition,
    }


if __name__ == "__main__":

    result = calculate_asset_beta(
        "PETR4.SA",
        "^BVSP",
        "2025-01-01",
        "2025-12-31",
        risk_free_rate=0.04,
        market_expected_return=0.10
    )

    print(
        f"Beta: "
        f"{result['beta']:.4f}"
    )

    print(
        f"Beta (regression): "
        f"{result['beta_regression']:.4f}"
    )

    print(
        f"Alpha: "
        f"{result['alpha']:.6f}"
    )

    print(
        f"R²: "
        f"{result['r_squared']:.4f}"
    )

    print(
        f"Observations: "
        f"{result['observations']}"
    )

    print(
        f"CAPM Expected Return: "
        f"{result['expected_return']:.4%}"
    )

    print(
        f"Observed Return: "
        f"{result['observed_return']:.4%}"
    )

    print(
        f"CAPM Difference: "
        f"{result['capm_difference']:.4%}"
    )

    print(
        f"CAPM Interpretation: "
        f"{result['capm_interpretation']}"
    )

    print(
        f"Asset Volatility: "
        f"{result['asset_volatility']:.4%}"
    )

    print(
        f"Market Volatility: "
        f"{result['market_volatility']:.4%}"
    )

    print(
        f"Correlation: "
        f"{result['correlation']:.4f}"
    )

    print(
        f"Idiosyncratic Risk: "
        f"{result['idiosyncratic_risk']:.4%}"
    )

    variance = result["variance_decomposition"]

    print()
    print("Variance Decomposition:")

    print(
        f"Total Variance: "
        f"{variance['total_variance']:.8f}"
    )

    print(
        f"Systematic Variance: "
        f"{variance['systematic_variance']:.8f}"
    )

    print(
        f"Residual Variance: "
        f"{variance['residual_variance']:.8f}"
    )

    print(
        f"Systematic Variance %: "
        f"{variance['systematic_percentage']:.2%}"
    )

    print(
        f"Residual Variance %: "
        f"{variance['residual_percentage']:.2%}"
    )