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

def _regularized_covariance_matrix(returns: pd.DataFrame, ridge: float = 1e-4) -> np.ndarray:
    """Return a symmetric positive-definite covariance matrix.

    The minimum-variance optimizer is sensitive to nearly singular covariance
    matrices, especially with highly correlated bond ETFs. This helper makes the
    covariance matrix numerically safer while preserving the empirical structure.
    """
    cov = returns.cov().to_numpy(dtype=float)
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    cov = 0.5 * (cov + cov.T)

    n_assets = cov.shape[0]
    average_variance = float(np.trace(cov) / max(n_assets, 1))
    ridge_scale = ridge * max(average_variance, 1e-12)

    cov = cov + ridge_scale * np.eye(n_assets)

    min_eigenvalue = float(np.linalg.eigvalsh(cov).min())
    if min_eigenvalue <= 0.0:
        cov = cov + (abs(min_eigenvalue) + ridge_scale) * np.eye(n_assets)

    return cov

def minimum_variance_weight(
    returns: pd.DataFrame,
    long_only: bool = True,
    ridge: float = 1e-4,
) -> pd.Series:
    """Minimum-variance allocation with robust numerical fallbacks.

    The optimizer solves the long-only minimum-variance problem when
    ``long_only=True``. If the numerical optimizer fails, the function falls back
    to inverse-volatility weights, which correspond to a diagonal covariance
    approximation.

    The returned Series carries lightweight diagnostics in ``attrs``.
    """
    r = _clean_returns(returns)
    assets = returns.columns

    if len(assets) == 0:
        raise ValueError("returns must contain at least one asset")

    if r.shape[0] < 2:
        out = equal_weight(assets)
        out.attrs["optimizer_status"] = "fallback_equal_weight_insufficient_history"
        return out

    cov = _regularized_covariance_matrix(r[assets], ridge=ridge)
    n_assets = len(assets)

    if not long_only:
        ones = np.ones(n_assets)
        try:
            raw = np.linalg.solve(cov, ones)
            weights = raw / raw.sum()
            out = pd.Series(weights, index=assets, name="weight")
            out.attrs["optimizer_status"] = "success_unconstrained_closed_form"
            return out
        except np.linalg.LinAlgError:
            out = inverse_volatility_weight(r[assets])
            out.attrs["optimizer_status"] = "fallback_inverse_volatility_linalg_error"
            return out

    def objective(weights: np.ndarray) -> float:
        return float(weights @ cov @ weights)

    constraints = ({"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0},)
    bounds = [(0.0, 1.0) for _ in range(n_assets)]

    equal_start = np.full(n_assets, 1.0 / n_assets)

    inverse_vol_start = inverse_volatility_weight(r[assets]).reindex(assets).to_numpy(dtype=float)
    inverse_vol_start = np.nan_to_num(inverse_vol_start, nan=1.0 / n_assets)
    inverse_vol_start = inverse_vol_start / inverse_vol_start.sum()

    starts = [equal_start, inverse_vol_start]

    # Add concentrated starts to reduce the chance of SLSQP stopping at the
    # equal-weight initial point when the covariance surface is ill-conditioned.
    for i in range(n_assets):
        start = np.full(n_assets, 0.01 / max(n_assets - 1, 1))
        start[i] = 0.99
        starts.append(start / start.sum())

    best_weights: np.ndarray | None = None
    best_value = np.inf
    best_message = ""

    for start in starts:
        result = minimize(
            objective,
            x0=start,
            bounds=bounds,
            constraints=constraints,
            method="SLSQP",
            options={"ftol": 1e-12, "maxiter": 1_000, "disp": False},
        )

        if not result.success or not np.all(np.isfinite(result.x)):
            best_message = str(result.message)
            continue

        candidate = np.clip(result.x, 0.0, 1.0)
        candidate_sum = candidate.sum()

        if candidate_sum <= 1e-12:
            continue

        candidate = candidate / candidate_sum
        candidate_value = objective(candidate)

        if candidate_value < best_value:
            best_value = candidate_value
            best_weights = candidate
            best_message = str(result.message)

    if best_weights is None:
        out = inverse_volatility_weight(r[assets])
        out.attrs["optimizer_status"] = "fallback_inverse_volatility_optimizer_failed"
        out.attrs["optimizer_message"] = best_message
        return out

    out = pd.Series(best_weights, index=assets, name="weight")
    out.attrs["optimizer_status"] = "success_long_only_slsqp"
    out.attrs["optimizer_objective"] = best_value
    out.attrs["optimizer_message"] = best_message

    return out

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
