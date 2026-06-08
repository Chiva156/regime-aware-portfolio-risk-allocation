"""Regime-aware portfolio risk and allocation.

The package is intentionally modular:

- data: public data loading and return computation;
- features: volatility, drawdown, correlation and momentum features;
- regimes: latent regime detection and economic relabeling;
- risk: regime-conditioned risk estimates;
- allocation: simple allocation policies and optimizers;
- smdp_policy: stress-state and intervention-policy utilities;
- backtest: walk-forward backtesting;
- metrics: performance diagnostics.
"""

from regime_portfolio.allocation import (
    equal_weight,
    inverse_volatility_weight,
    minimum_variance_weight,
    stress_blend_weight,
)
from regime_portfolio.backtest import BacktestResult, run_backtest
from regime_portfolio.data import DEFAULT_TICKERS, compute_returns, load_price_panel
from regime_portfolio.features import FeatureConfig, make_feature_panel
from regime_portfolio.metrics import performance_summary
from regime_portfolio.regimes import RegimeDetector
from regime_portfolio.risk import regime_risk_summary
from regime_portfolio.smdp_policy import (
    StressPolicyConfig,
    build_stress_states,
    estimate_transition_matrix,
    threshold_risk_budget_policy,
)

__all__ = [
    "DEFAULT_TICKERS",
    "BacktestResult",
    "FeatureConfig",
    "RegimeDetector",
    "StressPolicyConfig",
    "build_stress_states",
    "compute_returns",
    "equal_weight",
    "estimate_transition_matrix",
    "inverse_volatility_weight",
    "load_price_panel",
    "make_feature_panel",
    "minimum_variance_weight",
    "performance_summary",
    "regime_risk_summary",
    "run_backtest",
    "stress_blend_weight",
    "threshold_risk_budget_policy",
]
