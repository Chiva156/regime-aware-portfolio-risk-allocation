from __future__ import annotations

import numpy as np
import pandas as pd

from regime_portfolio.allocation import make_regime_weight_function
from regime_portfolio.backtest import run_backtest
from regime_portfolio.features import make_feature_panel
from regime_portfolio.regimes import RegimeDetector
from regime_portfolio.risk import regime_risk_summary


def make_synthetic_prices(n_days: int = 1_500, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic multi-asset panel with a high-volatility block."""

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    assets = ["SPY", "TLT", "IEF", "GLD", "DBC", "UUP", "SHY"]
    mu = np.array([0.07, 0.03, 0.025, 0.04, 0.035, 0.015, 0.01]) / 252
    vol = np.array([0.16, 0.12, 0.06, 0.15, 0.20, 0.08, 0.015]) / np.sqrt(252)
    shocks = rng.normal(mu, vol, size=(n_days, len(assets)))
    shocks[650:850, 0] += rng.normal(-0.0008, 0.025, size=200)
    shocks[650:850, 1] += rng.normal(0.0004, 0.012, size=200)
    prices = 100 * np.exp(np.cumsum(shocks, axis=0))
    return pd.DataFrame(prices, index=dates, columns=assets)


def main() -> None:
    prices = make_synthetic_prices()
    returns = np.log(prices).diff().dropna()
    features = make_feature_panel(prices, returns)

    detector = RegimeDetector(n_regimes=3, random_state=42).fit(features)
    regimes = detector.predict(features)

    print("Transition matrix")
    print(detector.transition_matrix().round(3))
    print()

    summary = regime_risk_summary(returns, regimes)
    print("Regime risk summary")
    print(summary.round(4))
    print()

    strategy = make_regime_weight_function(method="minimum_variance", defensive_asset="SHY")
    result = run_backtest(returns, strategy, regimes=regimes, lookback=252)
    print("Backtest summary")
    print(result.summary.round(4))


if __name__ == "__main__":
    main()
