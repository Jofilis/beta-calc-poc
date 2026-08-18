def calculate_market_risk_premium(
    market_expected_return,
    risk_free_rate
):
    """
    Calculate the market risk premium.

    Market Risk Premium = E(Rm) - Rf
    """
    return market_expected_return - risk_free_rate


def calculate_expected_return(
    beta,
    market_expected_return,
    risk_free_rate
):
    """
    Calculate the expected return using the CAPM.

    E(Ri) = Rf + Beta * [E(Rm) - Rf]
    """
    market_risk_premium = calculate_market_risk_premium(
        market_expected_return,
        risk_free_rate
    )

    return risk_free_rate + beta * market_risk_premium


def compare_observed_expected_return(
    observed_return,
    expected_return
):
    """
    Calculate the difference between observed and
    CAPM expected return.

    Difference = Observed Return - Expected Return
    """
    return observed_return - expected_return


def interpret_capm_difference(
    capm_difference
):
    """
    Interpret the difference between observed and
    CAPM expected return.
    """
    if capm_difference > 0:
        return "Observed return was above the CAPM expected return."

    if capm_difference < 0:
        return "Observed return was below the CAPM expected return."

    return "Observed return was equal to the CAPM expected return."