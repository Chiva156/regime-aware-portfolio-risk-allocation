from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_cumulative_returns(strategy_returns: pd.DataFrame, title: str = "Cumulative returns"):
    cumulative = (1.0 + strategy_returns).cumprod()

    ax = cumulative.plot(figsize=(11, 5))
    ax.set_title(title)
    ax.set_ylabel("Growth of $1")
    ax.grid(True, alpha=0.3)

    return ax


def plot_drawdowns(strategy_returns: pd.DataFrame, title: str = "Drawdowns"):
    cumulative = (1.0 + strategy_returns).cumprod()
    drawdowns = cumulative / cumulative.cummax() - 1.0

    ax = drawdowns.plot(figsize=(11, 5))
    ax.set_title(title)
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)

    return ax


def plot_regime_probabilities(probabilities: pd.DataFrame, title: str = "Regime probabilities"):
    ax = probabilities.plot(figsize=(11, 5))
    ax.set_title(title)
    ax.set_ylabel("Probability")
    ax.grid(True, alpha=0.3)

    return ax
