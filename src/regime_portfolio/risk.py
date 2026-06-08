from __future__ import annotations

import numpy as np
import pandas as pd


def annualized_return(returns: pd.Series, annualization: int = 252) -> float:
    """Annualized arithmetic mean return."""

    return float(returns.mean() * annualization)


def annualized_volatility(returns: pd.Series, annualization: int = 252) -> float:
    """Annualized volatility."""

    return float(returns.std(ddof=1) * np.sqrt(annualization))


def historical_cvar(returns: pd.Series, alpha: float = 0.05) -> float:
    """Historical conditional value-at-risk of returns.

    Returned value is a positive loss number when the lower tail is negative.
    """

    x = returns.dropna().sort_values()
    if x.empty:
        return np.nan
    n_tail = max(1, int(np.ceil(alpha * len(x))))
    return float(-x.iloc[:n_tail].mean())


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Drawdown series from simple or log returns.

    For small daily returns this is robust enough for diagnostics. If returns are
    log returns, exponentiating cumulative sums gives the wealth curve.
    """

    wealth = np.exp(returns.fillna(0.0).cumsum())
    peak = wealth.cummax()
    return wealth / peak - 1.0


def max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown as a negative number."""

    dd = drawdown_series(returns)
    return float(dd.min()) if not dd.empty else np.nan


def covariance_by_regime(
    returns: pd.DataFrame,
    regimes: pd.Series,
    annualization: int = 252,
) -> dict[int, pd.DataFrame]:
    """Estimate annualized covariance matrix by regime."""

    aligned = returns.join(regimes.rename("regime"), how="inner").dropna()
    out: dict[int, pd.DataFrame] = {}
    for regime, sample in aligned.groupby("regime"):
        r = sample.drop(columns="regime")
        out[int(regime)] = r.cov() * annualization
    return out


def correlation_by_regime(returns: pd.DataFrame, regimes: pd.Series) -> dict[int, pd.DataFrame]:
    """Estimate correlation matrix by regime."""

    aligned = returns.join(regimes.rename("regime"), how="inner").dropna()
    out: dict[int, pd.DataFrame] = {}
    for regime, sample in aligned.groupby("regime"):
        r = sample.drop(columns="regime")
        out[int(regime)] = r.corr()
    return out


def regime_risk_summary(
    returns: pd.DataFrame,
    regimes: pd.Series,
    portfolio_returns: pd.Series | None = None,
    annualization: int = 252,
    cvar_alpha: float = 0.05,
) -> pd.DataFrame:
    """Compute interpretable risk diagnostics by regime."""

    if portfolio_returns is None:
        portfolio_returns = returns.mean(axis=1)

    aligned = pd.concat(
        [returns, portfolio_returns.rename("portfolio"), regimes.rename("regime")], axis=1
    ).dropna()
    rows: list[dict[str, float | int]] = []
    for regime, sample in aligned.groupby("regime"):
        pr = sample["portfolio"]
        rows.append(
            {
                "regime": int(regime),
                "n_obs": int(len(sample)),
                "ann_return": annualized_return(pr, annualization),
                "ann_vol": annualized_volatility(pr, annualization),
                "sharpe_0rf": annualized_return(pr, annualization)
                / max(annualized_volatility(pr, annualization), 1e-12),
                "cvar_5pct": historical_cvar(pr, cvar_alpha),
                "max_drawdown": max_drawdown(pr),
                "avg_pairwise_corr": _average_pairwise_corr(sample[returns.columns]),
            }
        )
    return pd.DataFrame(rows).sort_values("regime").reset_index(drop=True)


def _average_pairwise_corr(sample: pd.DataFrame) -> float:
    corr = sample.corr().to_numpy()
    if corr.shape[0] < 2:
        return np.nan
    mask = ~np.eye(corr.shape[0], dtype=bool)
    return float(np.nanmean(corr[mask]))
