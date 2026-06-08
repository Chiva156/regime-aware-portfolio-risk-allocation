from __future__ import annotations

import numpy as np
import pandas as pd

from regime_portfolio.risk import drawdown_series, historical_cvar


def performance_summary(
    returns: pd.Series,
    annualization: int = 252,
    cvar_alpha: float = 0.05,
) -> pd.Series:
    """Common performance metrics for a daily return series."""

    r = returns.dropna().astype(float)
    if r.empty:
        raise ValueError("returns must not be empty")

    ann_return = float(r.mean() * annualization)
    ann_vol = float(r.std(ddof=1) * np.sqrt(annualization))
    downside = r.where(r < 0.0, 0.0).std(ddof=1) * np.sqrt(annualization)
    dd = drawdown_series(r)
    max_dd = float(dd.min())
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
    sortino = ann_return / downside if downside > 0 else np.nan
    calmar = ann_return / abs(max_dd) if max_dd < 0 else np.nan

    return pd.Series(
        {
            "ann_return": ann_return,
            "ann_vol": ann_vol,
            "sharpe_0rf": sharpe,
            "sortino_0rf": sortino,
            "max_drawdown": max_dd,
            "calmar": calmar,
            "cvar_5pct": historical_cvar(r, cvar_alpha),
            "skew": float(r.skew()),
            "kurtosis": float(r.kurtosis()),
        }
    )


def compare_strategies(strategy_returns: pd.DataFrame) -> pd.DataFrame:
    """Return a performance table for several strategies."""

    rows = {name: performance_summary(strategy_returns[name]) for name in strategy_returns.columns}
    return pd.DataFrame(rows).T.sort_values("sharpe_0rf", ascending=False)


def turnover(weights: pd.DataFrame) -> pd.Series:
    """One-way portfolio turnover from a weight history."""

    return weights.diff().abs().sum(axis=1) / 2.0
