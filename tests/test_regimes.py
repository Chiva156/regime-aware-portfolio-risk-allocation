from __future__ import annotations

import numpy as np
import pandas as pd

from regime_portfolio.risk import historical_cvar, regime_risk_summary


def test_historical_cvar_positive_for_negative_tail() -> None:
    returns = pd.Series([0.01, 0.02, -0.05, -0.03, 0.0])
    assert historical_cvar(returns, alpha=0.4) > 0


def test_regime_risk_summary_has_one_row_per_regime() -> None:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=100)
    returns = pd.DataFrame(rng.normal(0, 0.01, size=(100, 3)), index=dates, columns=list("ABC"))
    regimes = pd.Series([0] * 50 + [1] * 50, index=dates)
    summary = regime_risk_summary(returns, regimes)
    assert set(summary["regime"]) == {0, 1}
    assert "cvar_5pct" in summary.columns
