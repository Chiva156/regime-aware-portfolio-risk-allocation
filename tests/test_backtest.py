from __future__ import annotations

import numpy as np
import pandas as pd

from regime_portfolio.allocation import make_regime_weight_function
from regime_portfolio.backtest import run_backtest
from regime_portfolio.backtest import _rebalance_dates


def test_run_backtest_returns_summary_and_weights() -> None:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=600)
    returns = pd.DataFrame(
        rng.normal(0.0002, 0.01, size=(600, 4)),
        index=dates,
        columns=["SPY", "TLT", "GLD", "SHY"],
    )
    regimes = pd.Series(rng.integers(0, 3, size=600), index=dates)
    strategy = make_regime_weight_function(method="inverse_volatility", defensive_asset="SHY")
    result = run_backtest(returns, strategy, regimes=regimes, lookback=126)
    assert not result.returns.empty
    assert result.weights.shape == returns.shape
    assert "sharpe_0rf" in result.summary.index


def test_rebalance_dates_returns_actual_observed_dates():

    index = pd.to_datetime(
        [
            "2024-01-30",
            "2024-01-31",
            "2024-02-27",
            "2024-02-28",
            "2024-03-28",
        ]
    )

    dates = _rebalance_dates(pd.DatetimeIndex(index), "ME")

    expected = pd.DatetimeIndex(
        [
            "2024-01-31",
            "2024-02-28",
            "2024-03-28",
        ]
    )

    assert dates.equals(expected)
