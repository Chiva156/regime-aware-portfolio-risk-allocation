# Regime-Aware Portfolio Risk and Allocation

Dynamic multi-asset portfolio allocation using hidden market regimes, regime-conditioned risk estimates, and a stress-aware decision layer inspired by semi-Markov degradation-control models.

> **Research positioning.** This repository is not meant to be another generic "HMM bull/bear trading strategy". It treats portfolio fragility as a latent stress/degradation process and asks whether dynamic risk-budget intervention policies improve drawdown and tail-risk control relative to robust constant-risk policies.

---

## Why this project exists

Many regime-switching allocation examples stop at:

```text
HMM regimes -> label bull/bear -> rotate assets
```

This project uses a more research-oriented pipeline:

```text
market data
  -> risk and macro features
  -> latent regimes
  -> regime-conditioned covariance/dependence
  -> stress-state construction
  -> constant vs regime-dependent risk budgets
  -> SMDP-inspired intervention policy
  -> walk-forward backtest
```

The distinctive angle is the translation of ideas from degradation-control models to portfolio risk:

| Degradation-control model | Portfolio analogue |
| --- | --- |
| degradation state | market / portfolio stress state |
| production speed | risk budget, leverage, target volatility |
| degradation intensity | probability of moving to worse stress state |
| maintenance | de-risking, hedging, defensive rebalancing |
| revenue rate | expected return or risk-adjusted reward |
| average reward | long-run risk-adjusted performance |
| threshold policy | forced intervention under severe stress |

The project therefore speaks to **quant research**, **systematic macro**, **portfolio construction**, **risk management**, and **model validation**.

---

## Academic and technical anchors

The first version intentionally uses standard, reproducible tools and public data. The research layer is in the framing, diagnostics, robustness checks, and policy comparison.

Useful references and sources:

- Hamilton, J. D. (1989), *A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle*.
- Ang, A. and Bekaert, G. (2002), *International Asset Allocation with Regime Shifts*.
- Bulla, J. (2011), *Hidden Markov Models with t Components: Increased Persistence and Other Aspects of Portfolio Allocation*.
- Engle, R. (2002), *Dynamic Conditional Correlation*.
- Pelletier, D. (2006), *Regime Switching for Dynamic Correlations*.
- López de Prado, M. (2016), *Building Diversified Portfolios that Outperform Out of Sample*.
- FRED API: https://fred.stlouisfed.org/docs/api/fred/
- Kenneth French Data Library: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
- Cboe VIX historical data: https://www.cboe.com/tradable_products/vix/vix_historical_data/
- pandas-datareader Stooq reader: https://pandas-datareader.readthedocs.io/en/latest/readers/stooq.html
- hmmlearn documentation: https://hmmlearn.readthedocs.io/
- PyPortfolioOpt: https://pyportfolioopt.readthedocs.io/
- Riskfolio-Lib: https://riskfolio-lib.readthedocs.io/

---

## Asset universe

The default universe is deliberately simple and liquid:

| Ticker | Role |
| --- | --- |
| SPY | US equities |
| TLT | long-duration US Treasuries |
| IEF | intermediate US Treasuries |
| GLD | gold |
| DBC | commodities |
| UUP | US dollar |
| SHY | short Treasuries / defensive proxy |

You can extend this to global equities, credit ETFs, commodities, FX, futures proxies, or factor portfolios.

---

## Repository structure

```text
regime-aware-portfolio-risk-allocation/
├── README.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── LICENSE
├── notebooks/
│   ├── 01_data_pipeline.ipynb
│   ├── 02_regime_detection.ipynb
│   ├── 03_regime_conditioned_risk.ipynb
│   ├── 04_allocation_backtest.ipynb
│   └── 05_smdp_stress_policy.ipynb
├── src/
│   └── regime_portfolio/
│       ├── __init__.py
│       ├── allocation.py
│       ├── backtest.py
│       ├── data.py
│       ├── features.py
│       ├── metrics.py
│       ├── regimes.py
│       ├── risk.py
│       └── smdp_policy.py
├── tests/
│   ├── test_backtest.py
│   ├── test_features.py
│   └── test_risk.py
├── examples/
│   └── quick_start.py
└── docs/
    └── research_positioning.md
```

---

## Quick start

```bash
git clone <your-repo-url>
cd regime-aware-portfolio-risk-allocation
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .[dev]
```

Run tests:

```bash
pytest
```

Run a quick synthetic example:

```bash
python examples/quick_start.py
```

Open notebooks:

```bash
jupyter lab notebooks/
```

---

## Methodology overview

### 1. Data pipeline

The project fetches ETF prices and optional macro/risk series, aligns them to a common calendar, and computes log returns.

### 2. Feature construction

Default features include:

- rolling realized volatility;
- rolling drawdown;
- rolling average cross-asset correlation;
- momentum/trend features;
- optional VIX, credit-spread and yield-curve features.

### 3. Regime detection

A Gaussian HMM is used as the first regime detector. The raw hidden states are then economically ordered by stress score rather than interpreted directly.

### 4. Regime-conditioned risk

For each regime, the project estimates:

- average return;
- realized volatility;
- covariance/correlation matrix;
- CVaR;
- max drawdown;
- average regime duration;
- transition probabilities.

### 5. Allocation policies

The project compares:

- equal weight;
- inverse volatility;
- minimum variance;
- constant risk budget;
- regime-dependent risk budget;
- stress-aware defensive allocation.

### 6. SMDP-inspired stress policy

The signature layer constructs finite stress states and compares risk-budget interventions. It is intentionally transparent and auditable before any reinforcement-learning extension is attempted.

---

## Scientific questions

The first public version should answer:

1. Can we detect economically interpretable market regimes?
2. Do regimes correspond to materially different covariance and correlation structures?
3. Does regime-conditioned allocation reduce drawdowns?
4. Does dynamic risk-budget control outperform a robust constant-risk policy?
5. When does regime awareness fail or add no value?

The fifth question is important: the goal is credible research, not overfitted strategy marketing.

---

## Disclaimer

This repository is for research and educational purposes only. It is not investment advice and does not recommend buying, selling, or holding any security.
