# Project Log

## Notebook pipeline polish

Implemented the five-notebook research pipeline as thin visual interfaces over `src/regime_portfolio`:

1. `01_data_pipeline.ipynb` exports cleaned prices, returns, and interpretable risk features.
2. `02_regime_detection.ipynb` estimates ordered HMM regimes and exports regime labels/probabilities.
3. `03_regime_conditioned_risk.ipynb` validates whether regimes correspond to different volatility, CVaR, drawdown, and correlation states.
4. `04_allocation_backtest.ipynb` compares transparent constant-risk baselines with regime-aware risk-budget allocation.
5. `05_smdp_stress_policy.ipynb` builds finite stress states and tests a threshold-style degradation-control policy.

Research positioning: the notebooks emphasize regime-aware risk control, drawdown management, stress-state interpretation, and model diagnostics rather than overclaiming return alpha.

## 2026-06-10 — Project scaffold validated

### What changed
- Created initial Python package structure.
- Added notebooks for data, regimes, risk, allocation, and SMDP stress policy.
- Configured local and Codespaces execution.
- Validated test suite.

### Current status
- Tests pass locally and in Codespaces.
- Notebooks run with public ETF data.
- yfinance is used as default public data source.

### Main finding
The regime-aware inverse-volatility strategy improves risk-adjusted and tail-risk metrics relative to equal-weight allocation.

### Main issue
The raw three-regime HMM produces unstable one-day alternating regimes, so regime smoothing or a two-regime model is required.

### Next step
Improve regime detection and make the SMDP stress-policy layer the central project contribution.
