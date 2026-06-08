from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureConfig:
    """Windows and conventions for market-regime features."""

    vol_window: int = 21
    corr_window: int = 63
    drawdown_window: int = 252
    momentum_window: int = 63
    annualization: int = 252
    equity_proxy: str = "SPY"


def realized_volatility(
    returns: pd.DataFrame | pd.Series,
    window: int = 21,
    annualization: int = 252,
) -> pd.DataFrame | pd.Series:
    """Rolling annualized realized volatility."""

    return returns.rolling(window).std() * np.sqrt(annualization)


def rolling_drawdown(prices: pd.Series, window: int = 252) -> pd.Series:
    """Rolling drawdown relative to the rolling maximum."""

    prices = prices.astype(float)
    rolling_max = prices.rolling(window=window, min_periods=max(5, window // 5)).max()
    return prices / rolling_max - 1.0


def rolling_average_correlation(returns: pd.DataFrame, window: int = 63) -> pd.Series:
    """Rolling average pairwise correlation across assets.

    A high average correlation is often a useful proxy for diversification
    breakdown during market stress.
    """

    returns = returns.dropna(how="all")
    values: list[float] = []
    index: list[pd.Timestamp] = []
    for end in range(window, len(returns) + 1):
        sample = returns.iloc[end - window : end].dropna(axis=1, how="any")
        date = returns.index[end - 1]
        index.append(date)
        if sample.shape[1] < 2:
            values.append(np.nan)
            continue
        corr = sample.corr().to_numpy()
        mask = ~np.eye(corr.shape[0], dtype=bool)
        values.append(float(np.nanmean(corr[mask])))
    return pd.Series(values, index=pd.Index(index, name=returns.index.name), name="avg_corr")


def downside_semivolatility(
    returns: pd.Series,
    window: int = 63,
    annualization: int = 252,
) -> pd.Series:
    """Rolling annualized semivolatility of negative returns."""

    negative = returns.where(returns < 0.0, 0.0)
    return negative.rolling(window).std() * np.sqrt(annualization)


def momentum(prices: pd.Series | pd.DataFrame, window: int = 63) -> pd.Series | pd.DataFrame:
    """Simple momentum over a lookback window."""

    return prices / prices.shift(window) - 1.0


def make_feature_panel(
    prices: pd.DataFrame,
    returns: pd.DataFrame | None = None,
    config: FeatureConfig | None = None,
    extra_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a compact feature panel for regime detection.

    The default feature set is deliberately interpretable. It should be extended
    only after the baseline pipeline is robust.
    """

    cfg = config or FeatureConfig()
    if returns is None:
        returns = np.log(prices).diff()

    equity = cfg.equity_proxy
    if equity not in prices.columns:
        equity = prices.columns[0]

    equity_returns = returns[equity]
    vol = realized_volatility(equity_returns, cfg.vol_window, cfg.annualization).rename(
        f"{equity}_realized_vol"
    )
    dd = rolling_drawdown(prices[equity], cfg.drawdown_window).rename(f"{equity}_drawdown")
    corr = rolling_average_correlation(returns, cfg.corr_window)
    mom = momentum(prices[equity], cfg.momentum_window).rename(f"{equity}_momentum")
    semivol = downside_semivolatility(
        equity_returns, cfg.corr_window, cfg.annualization
    ).rename(f"{equity}_downside_semivol")

    features = pd.concat([vol, dd, corr, mom, semivol], axis=1)

    if extra_features is not None:
        extra = extra_features.reindex(features.index).ffill()
        features = pd.concat([features, extra], axis=1)

    features = features.replace([np.inf, -np.inf], np.nan)
    return features.dropna(how="any")


def stress_score(features: pd.DataFrame) -> pd.Series:
    """Build a transparent stress score from standardized feature columns.

    Higher values should correspond to more fragile market conditions. The signs
    are chosen heuristically and should be documented in notebooks.
    """

    if features.empty:
        raise ValueError("features must not be empty")

    z = (features - features.mean()) / features.std(ddof=0).replace(0.0, np.nan)
    signed = pd.DataFrame(index=z.index)

    for col in z.columns:
        lower = col.lower()
        sign = 1.0
        if "momentum" in lower:
            sign = -1.0
        signed[col] = sign * z[col]

    score = signed.mean(axis=1).rename("stress_score")
    return score.replace([np.inf, -np.inf], np.nan)
