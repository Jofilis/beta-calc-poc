from src.reports import generate_economic_interpretation


def create_metrics(
    beta=1.0,
    alpha=0.0,
    r_squared=0.50,
    observations=249
):
    return {
        "beta": beta,
        "alpha": alpha,
        "r_squared": r_squared,
        "observations": observations,
    }


def test_report_contains_asset_and_market():
    metrics = create_metrics()

    report = generate_economic_interpretation(
        metrics,
        "PETR4.SA",
        "^BVSP"
    )

    assert "PETR4.SA" in report
    assert "^BVSP" in report
    assert "249" in report


def test_beta_aggressive_classification():
    metrics = create_metrics(
        beta=1.50
    )

    report = generate_economic_interpretation(
        metrics,
        "ASSET",
        "MARKET"
    )

    assert "AGGRESSIVE" in report
    assert "1.50" in report


def test_beta_neutral_classification():
    metrics = create_metrics(
        beta=1.00
    )

    report = generate_economic_interpretation(
        metrics,
        "ASSET",
        "MARKET"
    )

    assert "NEUTRAL" in report


def test_beta_defensive_classification():
    metrics = create_metrics(
        beta=0.70
    )

    report = generate_economic_interpretation(
        metrics,
        "ASSET",
        "MARKET"
    )

    assert "DEFENSIVE" in report


def test_beta_negative_classification():
    metrics = create_metrics(
        beta=-0.50
    )

    report = generate_economic_interpretation(
        metrics,
        "ASSET",
        "MARKET"
    )

    assert "NEGATIVE" in report


def test_beta_zero_classification():
    metrics = create_metrics(
        beta=0.0
    )

    report = generate_economic_interpretation(
        metrics,
        "ASSET",
        "MARKET"
    )

    assert "ZERO" in report
    assert "NEGATIVE" not in report


def test_r_squared_high_explanatory_power():
    metrics = create_metrics(
        r_squared=0.75
    )

    report = generate_economic_interpretation(
        metrics,
        "ASSET",
        "MARKET"
    )

    assert "75.0%" in report
    assert "linearly explained" in report


def test_r_squared_medium_explanatory_power():
    metrics = create_metrics(
        r_squared=0.40
    )

    report = generate_economic_interpretation(
        metrics,
        "ASSET",
        "MARKET"
    )

    assert "40.0%" in report
    assert "other factors also play a relevant role" in report


def test_r_squared_low_explanatory_power():
    metrics = create_metrics(
        r_squared=0.20
    )

    report = generate_economic_interpretation(
        metrics,
        "ASSET",
        "MARKET"
    )

    assert "20.0%" in report
    assert "low explanatory power" in report


def test_positive_alpha():
    metrics = create_metrics(
        alpha=0.001
    )

    report = generate_economic_interpretation(
        metrics,
        "ASSET",
        "MARKET"
    )

    assert "POSITIVE" in report
    assert "0.1000%" in report


def test_negative_alpha():
    metrics = create_metrics(
        alpha=-0.001
    )

    report = generate_economic_interpretation(
        metrics,
        "ASSET",
        "MARKET"
    )

    assert "NEGATIVE" in report
    assert "-0.1000%" in report


def test_zero_alpha():
    metrics = create_metrics(
        alpha=0.0
    )

    report = generate_economic_interpretation(
        metrics,
        "ASSET",
        "MARKET"
    )

    assert "ZERO" in report