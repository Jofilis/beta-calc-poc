import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_regression(
    combined_returns,
    beta_regression,
    alpha,
    asset_ticker,
    market_ticker,
    save_path=None,
):
    """Gera o scatter plot dos retornos com a linha de regressão OLS."""

    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Dispersão dos retornos
    ax.scatter(
        combined_returns["market"],
        combined_returns["asset"],
        alpha=0.5,
        color="#1f77b4",
        edgecolors="none",
        label="Retornos",
    )

    # Linha de regressão:
    # R_asset = Alpha + Beta * R_market
    x_vals = np.array(
        [
            combined_returns["market"].min(),
            combined_returns["market"].max(),
        ]
    )

    y_vals = alpha + beta_regression * x_vals

    ax.plot(
        x_vals,
        y_vals,
        color="#d62728",
        linewidth=2,
        label=f"Reta de Regressão (Beta = {beta_regression:.2f})",
    )

    ax.set_title(
        f"Regressão Linear: {asset_ticker} vs {market_ticker}",
        fontsize=14,
        pad=15,
    )

    ax.set_xlabel(
        f"Retornos do Mercado ({market_ticker})",
        fontsize=11,
    )

    ax.set_ylabel(
        f"Retornos do Ativo ({asset_ticker})",
        fontsize=11,
    )

    ax.axhline(
        0,
        color="black",
        linestyle="--",
        linewidth=0.8,
        alpha=0.7,
    )

    ax.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=0.8,
        alpha=0.7,
    )

    ax.legend(frameon=True)

    plt.tight_layout()

    # Só salva se o usuário/programa fornecer explicitamente um caminho.
    if save_path is not None:
        fig.savefig(save_path, dpi=300)

    return fig


def plot_frequency_comparison(
    frequency_results,
    asset_ticker,
    save_path=None,
):
    """Gera um gráfico comparando o Beta por frequência."""

    sns.set_theme(style="whitegrid")

    freqs = list(frequency_results.keys())
    betas = [
        frequency_results[frequency]["beta"]
        for frequency in freqs
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(
        freqs,
        betas,
        color=["#2ca02c", "#ff7f0e", "#9467bd"],
        width=0.5,
    )

    ax.set_title(
        f"Sensibilidade do Beta por Frequência - {asset_ticker}",
        fontsize=13,
        pad=15,
    )

    ax.set_ylabel(
        "Valor do Beta",
        fontsize=11,
    )

    ax.axhline(
        1.0,
        color="red",
        linestyle="--",
        linewidth=1,
        label="Beta Neutro = 1.0",
    )

    for bar in bars:
        yval = bar.get_height()

        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 0.02,
            f"{yval:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.legend()

    plt.tight_layout()

    # Só salva se o usuário/programa fornecer explicitamente um caminho.
    if save_path is not None:
        fig.savefig(save_path, dpi=300)

    return fig