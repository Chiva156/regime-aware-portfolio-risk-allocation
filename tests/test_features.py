from __future__ import annotations

import numpy as np
import pandas as pd

from regime_portfolio.features import make_feature_panel, rolling_average_correlation


def test_make_feature_panel_returns_non_empty_panel() -> None:
    rng = np.random.default_rng(123)
    dates = pd.bdate_range("2020-01-01", periods=400)
    returns = pd.DataFrame(
        rng.normal(0.0002, 0.01, size=(400, 4)),
        index=dates,
        columns=["SPY", "TLT", "GLD", "SHY"],
    )
    prices = 100 * np.exp(returns.cumsum())
    features = make_feature_panel(prices, returns)
    assert not features.empty
    assert "avg_corr" in features.columns
    assert features.index.is_monotonic_increasing


def test_rolling_average_correlation_bounds() -> None:
    rng = np.random.default_rng(123)
    returns = pd.DataFrame(rng.normal(size=(100, 3)), columns=list("ABC"))
    corr = rolling_average_correlation(returns, window=20)
    assert corr.dropna().between(-1, 1).all()
