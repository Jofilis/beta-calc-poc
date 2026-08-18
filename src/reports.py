def generate_economic_interpretation(
    metrics,
    asset_ticker,
    market_ticker
):
    """
    Translate statistical model outputs into economic interpretation.
    """

    beta = metrics["beta"]
    alpha = metrics["alpha"]
    r_squared = metrics["r_squared"]
    obs = metrics["observations"]

    # --------------------------------------------------
    # 1. Beta classification
    # --------------------------------------------------

    if beta > 1.15:
        risk_profile = (
            "AGGRESSIVE (High sensitivity to market movements)"
        )

        beta_explanation = (
            f"A 1.0% movement in {market_ticker} is associated, "
            f"on average, with an approximately {beta:.2f}% movement "
            f"in {asset_ticker}, in the same direction."
        )

    elif 0.85 <= beta <= 1.15:
        risk_profile = (
            "NEUTRAL (Sensitivity close to the market)"
        )

        beta_explanation = (
            f"{asset_ticker} presents sensitivity to {market_ticker} "
            f"that is close to one-to-one."
        )

    elif 0 < beta < 0.85:
        risk_profile = (
            "DEFENSIVE (Lower sensitivity to the market)"
        )

        beta_explanation = (
            f"{asset_ticker} presents lower sensitivity to movements "
            f"in {market_ticker}."
        )

    elif beta < 0:
        risk_profile = (
            "NEGATIVE (Average movement opposite to the market)"
        )

        beta_explanation = (
            f"The negative Beta indicates an inverse average relationship "
            f"between the returns of {asset_ticker} and {market_ticker}."
        )

    else:
        risk_profile = (
            "ZERO (No linear sensitivity to the market)"
        )

        beta_explanation = (
            f"A Beta close to zero indicates little or no linear "
            f"sensitivity of {asset_ticker} to movements in {market_ticker}."
        )

    # --------------------------------------------------
    # 2. R-squared
    # --------------------------------------------------

    r2_pct = r_squared * 100

    if r_squared >= 0.60:
        r2_eval = (
            f"{r2_pct:.1f}% of the variation in the asset's returns "
            f"is linearly explained by market returns in the sample."
        )

    elif r_squared >= 0.30:
        r2_eval = (
            f"{r2_pct:.1f}% of the variation in the asset's returns "
            f"is linearly explained by market returns in the sample, "
            f"indicating that other factors also play a relevant role."
        )

    else:
        r2_eval = (
            f"{r2_pct:.1f}% of the variation in the asset's returns "
            f"is linearly explained by market returns in the sample, "
            f"indicating low explanatory power of the market model."
        )

    # --------------------------------------------------
    # 3. Regression Alpha
    # --------------------------------------------------

    if alpha > 0:
        alpha_eval = (
            f"POSITIVE ({alpha * 100:.4f}% per observation period). "
            f"The positive intercept indicates that the estimated "
            f"average return of the asset was above the component "
            f"associated with the market in the fitted regression."
        )

    elif alpha < 0:
        alpha_eval = (
            f"NEGATIVE ({alpha * 100:.4f}% per observation period). "
            f"The negative intercept indicates that the estimated "
            f"average return of the asset was below the component "
            f"associated with the market in the fitted regression."
        )

    else:
        alpha_eval = (
            "ZERO. The regression intercept is approximately zero."
        )

    # --------------------------------------------------
    # 4. Final report
    # --------------------------------------------------

    report = f"""
======================================================================
ECONOMIC INTERPRETATION REPORT — BETA ANALYSIS
----------------------------------------------------------------------
Asset Analyzed   : {asset_ticker}
Market Benchmark : {market_ticker}
Sample           : {obs} observations
======================================================================

1. SYSTEMATIC RISK PROFILE (BETA)
   - Beta Value   : {beta:.4f}
   - Classification: {risk_profile}
   - Interpretation: {beta_explanation}

2. MARKET ADHERENCE (R²)
   - R² Value     : {r_squared:.4f}
   - Evaluation   : {r2_eval}

3. REGRESSION ALPHA
   - Alpha        : {alpha:.6f}
   - Evaluation   : {alpha_eval}

======================================================================
"""

    return report