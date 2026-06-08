from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _clean_returns(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.replace([np.inf, -np.inf], np.nan).dropna(how="any")


def _normalize_weights(weights: pd.Series) -> pd.Series:
    weights = weights.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    total = weights.sum()
    if abs(total) < 1e-12:
        return pd.Series(1.0 / len(weights), index=weights.index)
    return weights / total


def equal_weight(assets: list[str] | pd.Index) -> pd.Series:
    """Equal-weight allocation."""

    assets = pd.Index(assets)
    return pd.Series(1.0 / len(assets), index=assets, name="weight")


def inverse_volatility_weight(returns: pd.DataFrame, floor: float = 1e-8) -> pd.Series:
    """Long-only inverse-volatility allocation."""

    r = _clean_returns(returns)
    if r.empty:
        return equal_weight(returns.columns)
    vol = r.std(ddof=1).clip(lower=floor)
    weights = 1.0 / vol
    return _normalize_weights(weights).rename("weight")


def minimum_variance_weight(
    returns: pd.DataFrame,
    long_only: bool = True,
    ridge: float = 1e-6,
) -> pd.Series:
    """Long-only minimum-variance allocation using scipy.

    The ridge term stabilizes the covariance matrix in short windows.
    """

    r = _clean_returns(returns)
    assets = returns.columns
    if len(assets) == 0:
        raise ValueError("returns must contain at least one asset")
    if r.shape[0] < 2:
        return equal_weight(assets)

    cov = r.cov().to_numpy()
    cov = cov + ridge * np.eye(cov.shape[0])
    n = cov.shape[0]
    x0 = np.full(n, 1.0 / n)

    def objective(w: np.ndarray) -> float:
        return float(w @ cov @ w)

    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = [(0.0, 1.0) for _ in range(n)] if long_only else None
    result = minimize(objective, x0=x0, bounds=bounds, constraints=constraints, method="SLSQP")
    if not result.success:
        return equal_weight(assets)
    return pd.Series(result.x, index=assets, name="weight")


def target_volatility_overlay(
    weights: pd.Series,
    returns: pd.DataFrame,
    target_vol: float = 0.10,
    annualization: int = 252,
    max_leverage: float = 1.5,
) -> tuple[pd.Series, float]:
    """Scale risky weights to a target annualized volatility.

    Returns scaled weights and the leverage multiplier. The caller decides how to
    allocate residual cash if the multiplier is below 1.
    """

    aligned = returns[weights.index].dropna(how="any")
    if aligned.shape[0] < 2:
        return weights, 1.0
    cov = aligned.cov() * annualization
    vol = float(np.sqrt(weights.to_numpy() @ cov.to_numpy() @ weights.to_numpy()))
    if vol <= 0 or np.isnan(vol):
        return weights, 1.0
    multiplier = min(max_leverage, target_vol / vol)
    return weights * multiplier, multiplier


def stress_blend_weight(
    risky_weights: pd.Series,
    defensive_asset: str = "SHY",
    risk_budget: float = 1.0,
) -> pd.Series:
    """Blend risky allocation with a defensive asset.

    Parameters
    ----------
    risky_weights:
        Long-only weights that sum to one across the risky universe.
    defensive_asset:
        Asset used to absorb the unallocated risk budget.
    risk_budget:
        Scalar in [0, 1]. A value of 0 means fully defensive; 1 means fully
        invested in the risky allocation.
    """

    risk_budget = float(np.clip(risk_budget, 0.0, 1.0))
    weights = risky_weights.copy().astype(float) * risk_budget
    if defensive_asset not in weights.index:
        weights.loc[defensive_asset] = 0.0
    weights.loc[defensive_asset] += 1.0 - risk_budget
    return _normalize_weights(weights).rename("weight")


def regime_risk_budget(
    regime: int,
    budgets: Mapping[int, float] | None = None,
) -> float:
    """Map ordered regimes to risk budgets."""

    default = {0: 1.0, 1: 0.7, 2: 0.35, 3: 0.15}
    mapping = dict(default if budgets is None else budgets)
    return float(mapping.get(int(regime), min(mapping.values())))


def make_regime_weight_function(
    method: str = "minimum_variance",
    defensive_asset: str = "SHY",
    budgets: Mapping[int, float] | None = None,
):
    """Factory returning a backtest-compatible strategy function."""

    def strategy(history: pd.DataFrame, current_regime: int | None = None) -> pd.Series:
        if method == "equal_weight":
            risky = equal_weight(history.columns)
        elif method == "inverse_volatility":
            risky = inverse_volatility_weight(history)
        elif method == "minimum_variance":
            risky = minimum_variance_weight(history)
        else:
            raise ValueError(f"Unknown method: {method}")

        if current_regime is None:
            return risky
        budget = regime_risk_budget(current_regime, budgets)
        return stress_blend_weight(risky, defensive_asset=defensive_asset, risk_budget=budget)

    return strategy
