def generate_economic_interpretation(
    metrics,
    asset_ticker,
    market_ticker
):
    """Traduz os indicadores estatísticos do modelo em interpretação econômica."""

    beta = metrics["beta"]
    alpha = metrics["alpha"]
    r_squared = metrics["r_squared"]
    obs = metrics["observations"]

    # 1. Classificação do Beta
    if beta > 1.15:
        risk_profile = (
            "AGRESSIVO (Alta sensibilidade às variações do mercado)"
        )

        beta_explanation = (
            f"Um movimento de 1.0% no {market_ticker} está associado, "
            f"em média, a um movimento de aproximadamente {beta:.2f}% "
            f"no {asset_ticker}, na mesma direção."
        )

    elif 0.85 <= beta <= 1.15:
        risk_profile = (
            "NEUTRO (Sensibilidade próxima à do mercado)"
        )

        beta_explanation = (
            f"O {asset_ticker} apresenta sensibilidade ao {market_ticker} "
            f"próxima de um para um."
        )

    elif 0 < beta < 0.85:
        risk_profile = (
            "DEFENSIVO (Menor sensibilidade ao mercado)"
        )

        beta_explanation = (
            f"O {asset_ticker} apresenta menor sensibilidade às variações "
            f"do {market_ticker}."
        )

    else:
        risk_profile = (
            "NEGATIVO (Movimento médio em direção oposta ao mercado)"
        )

        beta_explanation = (
            f"O Beta negativo indica uma relação média inversa entre os "
            f"retornos do {asset_ticker} e do {market_ticker}."
        )

    # 2. R²
    r2_pct = r_squared * 100

    if r_squared >= 0.60:
        r2_eval = (
            f"{r2_pct:.1f}% da variação dos retornos do ativo é explicada "
            f"linearmente pelos retornos do mercado na amostra."
        )

    elif r_squared >= 0.30:
        r2_eval = (
            f"{r2_pct:.1f}% da variação dos retornos do ativo é explicada "
            f"linearmente pelos retornos do mercado na amostra, indicando "
            f"que outros fatores também têm papel relevante."
        )

    else:
        r2_eval = (
            f"{r2_pct:.1f}% da variação dos retornos do ativo é explicada "
            f"linearmente pelos retornos do mercado na amostra, indicando "
            f"baixa capacidade explicativa do modelo de mercado."
        )

    # 3. Alpha da regressão
    if alpha > 0:
        alpha_eval = (
            f"POSITIVO ({alpha * 100:.4f}% por período de observação). "
            f"O intercepto positivo indica que o retorno médio do ativo "
            f"ficou acima do componente de retorno previsto pelo mercado "
            f"no modelo estimado."
        )

    elif alpha < 0:
        alpha_eval = (
            f"NEGATIVO ({alpha * 100:.4f}% por período de observação). "
            f"O intercepto negativo indica que o retorno médio do ativo "
            f"ficou abaixo do componente de retorno previsto pelo mercado "
            f"no modelo estimado."
        )

    else:
        alpha_eval = (
            "NULO. O intercepto da regressão é aproximadamente zero."
        )

    report = f"""
======================================================================
RELATÓRIO DE INTERPRETAÇÃO ECONÔMICA DA ANÁLISE DE BETA
----------------------------------------------------------------------
Ativo Analisado  : {asset_ticker}
Índice de Mercado: {market_ticker}
Amostra          : {obs} observações
======================================================================

1. PERFIL DE RISCO SISTEMÁTICO (BETA)
   - Valor do Beta: {beta:.4f}
   - Classificação: {risk_profile}
   - Interpretação: {beta_explanation}

2. ADERÊNCIA AO MERCADO (R²)
   - Valor do R²  : {r_squared:.4f}
   - Avaliação    : {r2_eval}

3. ALPHA DA REGRESSÃO
   - Alpha         : {alpha:.6f}
   - Avaliação     : {alpha_eval}

======================================================================
"""

    return report