import pytest

from src.capm import (
    calculate_market_risk_premium,
    calculate_expected_return,
    compare_observed_expected_return,
    interpret_capm_difference
)


def test_market_risk_premium():
    market_return = 0.10
    risk_free_rate = 0.04

    result = calculate_market_risk_premium(
        market_return,
        risk_free_rate
    )

    assert result == pytest.approx(0.06)


def test_expected_return():
    beta = 1.5
    market_return = 0.10
    risk_free_rate = 0.04

    result = calculate_expected_return(
        beta,
        market_return,
        risk_free_rate
    )

    assert result == pytest.approx(0.13)


def test_compare_observed_expected_return():
    observed_return = 0.15
    expected_return = 0.13

    result = compare_observed_expected_return(
        observed_return,
        expected_return
    )

    assert result == pytest.approx(0.02)


def test_interpret_capm_difference():
    assert (
        interpret_capm_difference(0.02)
        == "Observed return was above the CAPM expected return."
    )

    assert (
        interpret_capm_difference(-0.02)
        == "Observed return was below the CAPM expected return."
    )

    assert (
        interpret_capm_difference(0)
        == "Observed return was equal to the CAPM expected return."
    )