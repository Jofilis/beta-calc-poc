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
    """
    Generate a scatter plot of asset and market returns
    with the OLS regression line.
    """

    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter plot of returns
    ax.scatter(
        combined_returns["market"],
        combined_returns["asset"],
        alpha=0.5,
        color="#1f77b4",
        edgecolors="none",
        label="Returns",
    )

    # OLS regression line:
    #
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
        label=f"OLS Regression (Beta = {beta_regression:.2f})",
    )

    ax.set_title(
        f"Linear Regression: {asset_ticker} vs {market_ticker}",
        fontsize=14,
        pad=15,
    )

    ax.set_xlabel(
        f"Market Returns ({market_ticker})",
        fontsize=11,
    )

    ax.set_ylabel(
        f"Asset Returns ({asset_ticker})",
        fontsize=11,
    )

    # Reference lines
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

    # Save only when explicitly requested.
    if save_path is not None:
        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

    # Display the figure.
    plt.show()

    return fig


def plot_frequency_comparison(
    frequency_results,
    asset_ticker,
    save_path=None,
):
    """
    Generate a chart comparing Beta across frequencies.
    """

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
        f"Beta Sensitivity by Frequency - {asset_ticker}",
        fontsize=13,
        pad=15,
    )

    ax.set_xlabel(
        "Frequency",
        fontsize=11,
    )

    ax.set_ylabel(
        "Beta",
        fontsize=11,
    )

    # Neutral Beta reference
    ax.axhline(
        1.0,
        color="red",
        linestyle="--",
        linewidth=1,
        label="Neutral Beta = 1.0",
    )

    # Beta values above the bars
    for bar in bars:
        yval = bar.get_height()

        offset = 0.02 if yval >= 0 else -0.05

        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + offset,
            f"{yval:.2f}",
            ha="center",
            va="bottom" if yval >= 0 else "top",
            fontsize=10,
            fontweight="bold",
        )

    ax.legend()

    plt.tight_layout()

    # Save only when explicitly requested.
    if save_path is not None:
        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

    # Display the figure.
    plt.show()

    return fig