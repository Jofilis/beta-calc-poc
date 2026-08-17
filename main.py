import argparse
import pandas as pd

from src.analysis import calculate_asset_beta, compare_frequencies
from src.data import download_prices, calculate_returns
from src.plots import plot_frequency_comparison, plot_regression
from src.reports import generate_economic_interpretation
from src.tickers import IBOVESPA_TICKERS, MARKET_BENCHMARKS, SP500_TICKERS


def run_single_pipeline(
    asset_ticker="PETR4.SA",
    market_ticker="^BVSP",
    start_date="2025-01-01",
    end_date="2025-12-31",
):
    print(
        f"\n🚀 Iniciando Análise Beta: "
        f"{asset_ticker} vs {market_ticker}..."
    )

    print("📥 [1/5] Baixando e processando histórico de dados...")

    asset_prices = download_prices(
        asset_ticker,
        start_date,
        end_date
    )

    market_prices = download_prices(
        market_ticker,
        start_date,
        end_date
    )

    asset_returns = calculate_returns(
        asset_prices,
        frequency="daily"
    )

    market_returns = calculate_returns(
        market_prices,
        frequency="daily"
    )

    combined = (
        asset_returns
        .to_frame(name="asset")
        .join(
            market_returns.to_frame(name="market"),
            how="inner"
        )
        .dropna()
    )

    if combined.empty:
        raise ValueError(
            "No overlapping return observations were found "
            "for the selected asset and market."
        )

    print(
        "📊 [2/5] Calculando matriz de risco, "
        "Beta OLS, Alpha e R²..."
    )

    analysis = calculate_asset_beta(
        asset_ticker,
        market_ticker,
        start_date,
        end_date,
        frequency="daily"
    )

    print("📈 [3/5] Gerando análises visuais...")

    reg_plot = plot_regression(
        combined_returns=combined,
        beta_regression=analysis["beta_regression"],
        alpha=analysis["alpha"],
        asset_ticker=asset_ticker,
        market_ticker=market_ticker,
        save_path=None,
    )

    freq_results = compare_frequencies(
        asset_ticker,
        market_ticker,
        start_date,
        end_date
    )

    freq_plot = plot_frequency_comparison(
        freq_results,
        asset_ticker,
        save_path=None,
    )

    print("📝 [4/5] Compilando relatório de interpretação econômica...")

    report_text = generate_economic_interpretation(
        analysis,
        asset_ticker,
        market_ticker
    )

    print("\n" + report_text)

    print("📊 [5/5] Análise concluída.")
    print(f"   Observações utilizadas: {analysis['observations']}")
    print(f"   Gráfico de regressão: {reg_plot}")
    print(f"   Comparação de frequência: {freq_plot}")


def run_batch_analysis(
    index_name="IBOV",
    start_date="2025-01-01",
    end_date="2025-12-31"
):
    tickers = (
        IBOVESPA_TICKERS
        if index_name == "IBOV"
        else SP500_TICKERS
    )

    market = MARKET_BENCHMARKS.get(
        index_name,
        "^BVSP"
    )

    print(
        f"\n⚡ Processando lote de {len(tickers)} ativos "
        f"do {index_name} vs {market}..."
    )

    results = []

    for ticker in tickers:
        try:
            res = calculate_asset_beta(
                ticker,
                market,
                start_date,
                end_date,
                frequency="daily"
            )

            results.append(
                {
                    "Ticker": ticker,
                    "Beta": res["beta"],
                    "Alpha": res["alpha"],
                    "R2": res["r_squared"],
                    "Observacoes": res["observations"],
                }
            )

            print(
                f"  ✓ {ticker:<10} | "
                f"Beta: {res['beta']:.2f} | "
                f"Alpha: {res['alpha']:.5f} | "
                f"R²: {res['r_squared']:.2f}"
            )

        except Exception as e:
            print(
                f"  ✗ {ticker:<10} | "
                f"Erro ao processar: {e}"
            )

    if not results:
        raise ValueError(
            "No assets were successfully analyzed."
        )

    df_results = pd.DataFrame(results).sort_values(
        by="Beta",
        ascending=False
    )

    print("\n📊 Ranking calculado:")
    print(df_results.to_string(index=False))

    return df_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Análise de Beta e Risco de Ativos"
    )

    parser.add_argument(
        "--asset",
        type=str,
        default="PETR4.SA",
        help="Ticker do ativo (ex: VALE3.SA, AAPL)",
    )

    parser.add_argument(
        "--market",
        type=str,
        default="^BVSP",
        help="Ticker do mercado (ex: ^BVSP, ^GSPC)",
    )

    parser.add_argument(
        "--start",
        type=str,
        default="2025-01-01",
        help="Data inicial (AAAA-MM-DD)",
    )

    parser.add_argument(
        "--end",
        type=str,
        default="2025-12-31",
        help="Data final (AAAA-MM-DD)",
    )

    parser.add_argument(
        "--batch",
        type=str,
        choices=["IBOV", "SP500"],
        help="Executa lote de um índice inteiro (IBOV ou SP500)",
    )

    args = parser.parse_args()

    try:
        if args.batch:
            run_batch_analysis(
                index_name=args.batch,
                start_date=args.start,
                end_date=args.end,
            )
        else:
            run_single_pipeline(
                asset_ticker=args.asset,
                market_ticker=args.market,
                start_date=args.start,
                end_date=args.end,
            )
    except ValueError as e:
        print("\n" + "=" * 70)
        print("❌ ERRO NA EXECUÇÃO DO PIPELINE DE DADOS")
        print("=" * 70)
        print(f"Mensagem: {e}")
        print("\n💡 Dicas para resolver:")
        print(" 1. Verifique se o ticker possui o sufixo correto (ex: '.SA' para ações brasileiras).")
        print(" 2. Verifique se o período selecionado contém dias úteis de negociação.")
        print(" 3. Confirme se há interseção de datas válidas entre o ativo e o índice de mercado.")
        print("=" * 70 + "\n")
    except Exception as e:
        print(f"\n❌ Ocorreu um erro inesperado: {e}\n")