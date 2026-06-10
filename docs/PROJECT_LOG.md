# Project Log

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
