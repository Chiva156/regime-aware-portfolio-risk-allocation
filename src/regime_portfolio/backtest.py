from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from regime_portfolio.allocation import equal_weight
from regime_portfolio.metrics import performance_summary, turnover

StrategyFunction = Callable[[pd.DataFrame, int | None], pd.Series]


@dataclass(frozen=True)
class BacktestResult:
    """Container for walk-forward backtest outputs."""

    returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    summary: pd.Series


def _rebalance_dates(index: pd.DatetimeIndex, frequency: str) -> pd.DatetimeIndex:
    """Select the last observed timestamp in each rebalance period.

    Pandas resampling labels periods by calendar period-end dates. For financial
    data, the calendar period-end may not be an observed trading date. This
    helper returns the actual last timestamp available in each period so that the
    backtest can rebalance on dates that exist in the return index.
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("index must be a pandas DatetimeIndex")

    if index.empty:
        return pd.DatetimeIndex([])

    observed_dates = pd.Series(index, index=index)
    rebalance_dates = observed_dates.resample(frequency).last().dropna()

    return pd.DatetimeIndex(rebalance_dates.to_list())


def run_backtest(
    returns: pd.DataFrame,
    strategy_fn: StrategyFunction,
    regimes: pd.Series | None = None,
    lookback: int = 252,
    rebalance_frequency: str = "ME",
    transaction_cost_bps: float = 2.0,
    initial_weights: pd.Series | None = None,
) -> BacktestResult:
    """Run a simple walk-forward backtest.

    Design choices:
    - weights are computed using information available up to the rebalance date;
    - the new weights are applied from the following trading day;
    - transaction costs are charged on rebalance days from one-way turnover.
    """

    returns = returns.dropna(how="any").sort_index()
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("returns must be indexed by a DatetimeIndex")
    if len(returns) <= lookback + 2:
        raise ValueError("returns history is too short for the requested lookback")

    regimes_aligned = None
    if regimes is not None:
        regimes_aligned = regimes.reindex(returns.index).ffill()

    rebal_dates = _rebalance_dates(returns.index[lookback:], rebalance_frequency)
    weights = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)

    current_weights = initial_weights if initial_weights is not None else equal_weight(returns.columns)
    current_weights = current_weights.reindex(returns.columns).fillna(0.0)

    for date in returns.index:
        weights.loc[date] = current_weights
        if date not in rebal_dates:
            continue
        loc = returns.index.get_loc(date)
        history = returns.iloc[max(0, loc - lookback) : loc]
        current_regime = None
        if regimes_aligned is not None and not pd.isna(regimes_aligned.loc[date]):
            current_regime = int(regimes_aligned.loc[date])
        new_weights = strategy_fn(history, current_regime)
        current_weights = new_weights.reindex(returns.columns).fillna(0.0)

    weights = weights.ffill().fillna(0.0)
    raw_portfolio_returns = (weights.shift(1).fillna(0.0) * returns).sum(axis=1)
    daily_turnover = turnover(weights).reindex(raw_portfolio_returns.index).fillna(0.0)
    costs = daily_turnover * transaction_cost_bps / 10_000.0
    net_returns = (raw_portfolio_returns - costs).rename("portfolio_return")
    return BacktestResult(
        returns=net_returns,
        weights=weights,
        turnover=daily_turnover,
        summary=performance_summary(net_returns),
    )
