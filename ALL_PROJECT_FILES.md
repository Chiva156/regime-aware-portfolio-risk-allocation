# Full project files content

## `.gitignore`

```text
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
.venv/
venv/
env/
build/
dist/
*.egg-info/
.pytest_cache/
.ruff_cache/

# Jupyter
.ipynb_checkpoints/

# Data and outputs
data/raw/*.csv
data/processed/*.csv
reports/figures/*.png
reports/figures/*.pdf

# OS / IDE
.DS_Store
.vscode/
.idea/

```

## `README.md`

```md
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

```

## `docs/research_positioning.md`

```md
# Research positioning note

## Core idea

This project translates degradation-control thinking into portfolio risk management.

In a degradation model, a system moves through increasingly damaged states. A controller chooses an operating intensity and occasionally performs maintenance. The objective is to balance revenue, deterioration, and intervention costs.

In this portfolio project, markets move through increasingly fragile regimes. A controller chooses a risk budget and occasionally de-risks. The objective is to balance expected return, volatility, drawdown, tail risk, and transaction costs.

## Why this is credible for a quant-finance pivot

A strong public GitHub project should not claim to have discovered alpha from a few ETFs. Instead, it should demonstrate that advanced stochastic modeling can be turned into a disciplined and reproducible research pipeline.

The message is:

> I can translate stochastic processes, Bayesian regime models, dependence modeling, and stochastic control into an implementable portfolio risk framework.

## What to emphasize in the README and LinkedIn post

- Latent regimes are treated as risk states, not only return predictors.
- Dependence breakdown is central: correlations are regime-conditioned.
- The baseline comparison is honest: robust constant-risk policies are hard to beat.
- The SMDP layer is interpretable and close to model-risk expectations.
- The framework can be extended to dynamic Bayesian networks, copulas, structural breaks, and reinforcement learning.

## Future extensions

1. Replace Gaussian HMM with Bayesian dynamic network regime inference.
2. Add dynamic vine-copula dependence modeling.
3. Add structural-break detection before HMM training.
4. Add transaction-cost-aware convex optimization.
5. Add constrained policy iteration for the stress intervention layer.
6. Add macro nowcasting features from FRED and OECD data.

```

## `examples/quick_start.py`

```py
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

```

## `notebooks/01_data_pipeline.ipynb`

```json
{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# 01 - Data pipeline\n",
        "\n",
        "Load prices, compute returns, and create the first feature panel. The public-data calls are intentionally isolated so the rest of the project can be tested with cached or synthetic data.\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "from regime_portfolio.data import DataConfig, compute_returns, load_price_panel\n",
        "from regime_portfolio.features import make_feature_panel\n",
        "\n",
        "config = DataConfig(start=\"2010-01-01\", source=\"stooq\")\n",
        "prices = load_price_panel(config)\n",
        "returns = compute_returns(prices)\n",
        "features = make_feature_panel(prices, returns)\n",
        "\n",
        "print(prices.tail())\n",
        "print(features.tail())\n"
      ]
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "name": "python",
      "pygments_lexer": "ipython3"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 5
}

```

## `notebooks/02_regime_detection.ipynb`

```json
{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# 02 - Regime detection\n",
        "\n",
        "Fit a Gaussian HMM and relabel raw states from low to high stress. Raw hidden-state labels are arbitrary and should never be interpreted directly.\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "from regime_portfolio.data import DataConfig, compute_returns, load_price_panel\n",
        "from regime_portfolio.features import make_feature_panel\n",
        "from regime_portfolio.regimes import RegimeDetector, regime_durations\n",
        "\n",
        "prices = load_price_panel(DataConfig(start=\"2010-01-01\", source=\"stooq\"))\n",
        "returns = compute_returns(prices)\n",
        "features = make_feature_panel(prices, returns)\n",
        "\n",
        "detector = RegimeDetector(n_regimes=3, random_state=42).fit(features)\n",
        "regimes = detector.predict(features)\n",
        "proba = detector.predict_proba(features)\n",
        "\n",
        "print(detector.transition_matrix().round(3))\n",
        "print(regime_durations(regimes).groupby(\"regime\")[\"duration\"].describe())\n",
        "print(proba.tail())\n"
      ]
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "name": "python",
      "pygments_lexer": "ipython3"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 5
}

```

## `notebooks/03_regime_conditioned_risk.ipynb`

```json
{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# 03 - Regime-conditioned risk\n",
        "\n",
        "Estimate covariance, correlation, CVaR and drawdown diagnostics by ordered regime. This is the dependence/risk layer of the project.\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "from regime_portfolio.data import DataConfig, compute_returns, load_price_panel\n",
        "from regime_portfolio.features import make_feature_panel\n",
        "from regime_portfolio.regimes import RegimeDetector\n",
        "from regime_portfolio.risk import correlation_by_regime, covariance_by_regime, regime_risk_summary\n",
        "\n",
        "prices = load_price_panel(DataConfig(start=\"2010-01-01\", source=\"stooq\"))\n",
        "returns = compute_returns(prices)\n",
        "features = make_feature_panel(prices, returns)\n",
        "regimes = RegimeDetector(n_regimes=3, random_state=42).fit(features).predict(features)\n",
        "\n",
        "summary = regime_risk_summary(returns, regimes)\n",
        "print(summary.round(4))\n",
        "\n",
        "corrs = correlation_by_regime(returns, regimes)\n",
        "print(corrs[max(corrs)].round(2))\n"
      ]
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "name": "python",
      "pygments_lexer": "ipython3"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 5
}

```

## `notebooks/04_allocation_backtest.ipynb`

```json
{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# 04 - Allocation backtest\n",
        "\n",
        "Compare a regime-dependent risk-budget policy against robust baselines. The focus is drawdown and tail-risk reduction, not alpha marketing.\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "from regime_portfolio.allocation import make_regime_weight_function\n",
        "from regime_portfolio.backtest import run_backtest\n",
        "from regime_portfolio.data import DataConfig, compute_returns, load_price_panel\n",
        "from regime_portfolio.features import make_feature_panel\n",
        "from regime_portfolio.metrics import compare_strategies\n",
        "from regime_portfolio.regimes import RegimeDetector\n",
        "\n",
        "prices = load_price_panel(DataConfig(start=\"2010-01-01\", source=\"stooq\"))\n",
        "returns = compute_returns(prices)\n",
        "features = make_feature_panel(prices, returns)\n",
        "regimes = RegimeDetector(n_regimes=3, random_state=42).fit(features).predict(features)\n",
        "\n",
        "strategies = {\n",
        "    \"equal_weight\": make_regime_weight_function(method=\"equal_weight\", budgets={0: 1.0, 1: 1.0, 2: 1.0}),\n",
        "    \"inverse_vol_regime\": make_regime_weight_function(method=\"inverse_volatility\", defensive_asset=\"SHY\"),\n",
        "    \"minvar_regime\": make_regime_weight_function(method=\"minimum_variance\", defensive_asset=\"SHY\"),\n",
        "}\n",
        "\n",
        "results = {}\n",
        "for name, fn in strategies.items():\n",
        "    results[name] = run_backtest(returns, fn, regimes=regimes, lookback=252).returns\n",
        "\n",
        "strategy_returns = __import__(\"pandas\").concat(results, axis=1)\n",
        "print(compare_strategies(strategy_returns).round(4))\n"
      ]
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "name": "python",
      "pygments_lexer": "ipython3"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 5
}

```

## `notebooks/05_smdp_stress_policy.ipynb`

```json
{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# 05 - SMDP-inspired stress policy\n",
        "\n",
        "Construct ordered stress states, estimate empirical transitions, and map stress states to risk budgets. This is the signature layer connecting portfolio risk with degradation-control thinking.\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "from regime_portfolio.data import DataConfig, compute_returns, load_price_panel\n",
        "from regime_portfolio.features import make_feature_panel\n",
        "from regime_portfolio.smdp_policy import build_stress_states, estimate_transition_matrix, threshold_risk_budget_policy\n",
        "\n",
        "prices = load_price_panel(DataConfig(start=\"2010-01-01\", source=\"stooq\"))\n",
        "returns = compute_returns(prices)\n",
        "features = make_feature_panel(prices, returns)\n",
        "\n",
        "states = build_stress_states(features, n_states=4)\n",
        "transition = estimate_transition_matrix(states)\n",
        "budgets = threshold_risk_budget_policy(states)\n",
        "\n",
        "print(transition.round(3))\n",
        "print(budgets.groupby(states).describe().round(3))\n"
      ]
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "name": "python",
      "pygments_lexer": "ipython3"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 5
}

```

## `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "regime-aware-portfolio-risk-allocation"
version = "0.1.0"
description = "Regime-aware portfolio risk and allocation with an SMDP-inspired stress policy layer."
readme = "README.md"
requires-python = ">=3.11"
authors = [
  {name = "Alex Dalla Torre"}
]
license = {text = "MIT"}
classifiers = [
  "Programming Language :: Python :: 3",
  "Programming Language :: Python :: 3.11",
  "License :: OSI Approved :: MIT License",
  "Operating System :: OS Independent",
  "Topic :: Scientific/Engineering :: Information Analysis",
]
dependencies = [
  "numpy>=1.26",
  "pandas>=2.2",
  "scipy>=1.11",
  "scikit-learn>=1.4",
  "matplotlib>=3.8",
  "pandas-datareader>=0.10",
  "yfinance>=0.2.40",
  "hmmlearn>=0.3.2",
]

[project.optional-dependencies]
dev = [
  "pytest>=8.0",
  "ruff>=0.5",
  "jupyterlab>=4.0",
  "ipykernel>=6.0",
]
portfolio = [
  "PyPortfolioOpt>=1.5",
  "riskfolio-lib>=6.0",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP"]
ignore = ["E501"]

```

## `requirements.txt`

```txt
numpy>=1.26
pandas>=2.2
scipy>=1.11
scikit-learn>=1.4
matplotlib>=3.8
pandas-datareader>=0.10
yfinance>=0.2.40
hmmlearn>=0.3.2
pytest>=8.0
ruff>=0.5
jupyterlab>=4.0
ipykernel>=6.0

```

## `src/regime_portfolio/__init__.py`

```py
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

```

## `src/regime_portfolio/allocation.py`

```py
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


def minimum_variance_weight(
    returns: pd.DataFrame,
    long_only: bool = True,
    ridge: float = 1e-6,
) -> pd.Series:
    """Long-only minimum-variance allocation using scipy.

    The ridge term stabilizes the covariance matrix in short windows.
    """

    r = _clean_returns(returns)
    assets = returns.columns
    if len(assets) == 0:
        raise ValueError("returns must contain at least one asset")
    if r.shape[0] < 2:
        return equal_weight(assets)

    cov = r.cov().to_numpy()
    cov = cov + ridge * np.eye(cov.shape[0])
    n = cov.shape[0]
    x0 = np.full(n, 1.0 / n)

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

```

## `src/regime_portfolio/backtest.py`

```py
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
    """Select the last available observation for each rebalance period."""

    dummy = pd.Series(1, index=index)
    return pd.DatetimeIndex(dummy.resample(frequency).last().dropna().index)


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

```

## `src/regime_portfolio/data.py`

```py
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

DEFAULT_TICKERS: tuple[str, ...] = ("SPY", "TLT", "IEF", "GLD", "DBC", "UUP", "SHY")


@dataclass(frozen=True)
class DataConfig:
    """Configuration for reproducible public-data loading."""

    tickers: tuple[str, ...] = DEFAULT_TICKERS
    start: str = "2007-01-01"
    end: str | None = None
    source: Literal["stooq", "yfinance"] = "stooq"


def _as_list(values: Iterable[str]) -> list[str]:
    return [str(v).upper() for v in values]


def _stooq_symbol(ticker: str) -> str:
    """Return Stooq US ETF symbol.

    Stooq expects US tickers as e.g. 'spy.us'.
    """

    symbol = ticker.lower()
    if "." not in symbol:
        symbol = f"{symbol}.us"
    return symbol


def fetch_stooq_prices(
    tickers: Iterable[str] = DEFAULT_TICKERS,
    start: str = "2007-01-01",
    end: str | None = None,
    price_col: str = "Close",
) -> pd.DataFrame:
    """Fetch daily prices from Stooq through pandas-datareader.

    Parameters
    ----------
    tickers:
        ETF or stock tickers. US symbols are automatically mapped to the Stooq
        '.us' suffix.
    start, end:
        Date bounds understood by pandas.
    price_col:
        Usually 'Close'. Stooq does not provide adjusted close in the same way
        as Yahoo; for a research demo this is acceptable but should be documented.

    Returns
    -------
    pandas.DataFrame
        Wide price panel with uppercase tickers as columns.
    """

    from pandas_datareader import data as pdr

    frames: list[pd.Series] = []
    failures: dict[str, str] = {}
    for ticker in _as_list(tickers):
        try:
            raw = pdr.DataReader(_stooq_symbol(ticker), "stooq", start=start, end=end)
            if raw.empty:
                failures[ticker] = "empty response"
                continue
            series = raw.sort_index()[price_col].rename(ticker)
            frames.append(series)
        except Exception as exc:  # pragma: no cover - network dependent
            failures[ticker] = str(exc)

    if not frames:
        message = "No Stooq prices could be downloaded."
        if failures:
            message += f" Failures: {failures}"
        raise RuntimeError(message)

    prices = pd.concat(frames, axis=1).sort_index()
    prices.index = pd.to_datetime(prices.index)
    prices = prices.dropna(how="all")
    if failures:
        prices.attrs["download_failures"] = failures
    return prices


def fetch_yfinance_prices(
    tickers: Iterable[str] = DEFAULT_TICKERS,
    start: str = "2007-01-01",
    end: str | None = None,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """Fetch prices from Yahoo Finance through yfinance.

    This is convenient for research but users should check Yahoo/yfinance terms
    before relying on it in production or redistributing data.
    """

    import yfinance as yf

    tickers_list = _as_list(tickers)
    raw = yf.download(
        tickers=" ".join(tickers_list),
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        progress=False,
        group_by="column",
    )
    if raw.empty:
        raise RuntimeError("No yfinance prices could be downloaded.")

    if isinstance(raw.columns, pd.MultiIndex):
        col = "Close" if "Close" in raw.columns.get_level_values(0) else "Adj Close"
        prices = raw[col].copy()
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers_list[0]})

    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index().dropna(how="all")
    return prices


def load_price_panel(config: DataConfig | None = None, **overrides: object) -> pd.DataFrame:
    """Load a public price panel using a :class:`DataConfig`.

    Examples
    --------
    >>> prices = load_price_panel(start="2015-01-01", source="stooq")
    """

    base = config or DataConfig()
    params = {**base.__dict__, **overrides}
    source = params.pop("source")
    if source == "stooq":
        return fetch_stooq_prices(**params)
    if source == "yfinance":
        return fetch_yfinance_prices(**params)
    raise ValueError(f"Unknown source: {source}")


def compute_returns(
    prices: pd.DataFrame,
    method: Literal["log", "simple"] = "log",
    dropna: bool = True,
) -> pd.DataFrame:
    """Compute daily returns from a price panel."""

    prices = prices.sort_index().astype(float)
    if method == "log":
        out = np.log(prices).diff()
    elif method == "simple":
        out = prices.pct_change()
    else:
        raise ValueError("method must be 'log' or 'simple'")
    return out.dropna(how="all") if dropna else out


def fetch_fred_series(
    series: dict[str, str] | Iterable[str],
    start: str = "2007-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """Fetch FRED series through pandas-datareader.

    Parameters
    ----------
    series:
        Either an iterable of FRED series ids or a mapping {output_name: fred_id}.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by date.
    """

    from pandas_datareader import data as pdr

    if isinstance(series, dict):
        mapping = series
    else:
        mapping = {s: s for s in series}

    frames: list[pd.Series] = []
    for output_name, fred_id in mapping.items():
        raw = pdr.DataReader(fred_id, "fred", start=start, end=end)
        frames.append(raw.iloc[:, 0].rename(output_name))

    out = pd.concat(frames, axis=1).sort_index()
    out.index = pd.to_datetime(out.index)
    return out


def align_to_returns(
    returns: pd.DataFrame,
    features: pd.DataFrame,
    fill_method: Literal["ffill", "none"] = "ffill",
) -> pd.DataFrame:
    """Align lower-frequency or irregular features to a return calendar."""

    features = features.sort_index()
    aligned = features.reindex(returns.index)
    if fill_method == "ffill":
        aligned = aligned.ffill()
    elif fill_method != "none":
        raise ValueError("fill_method must be 'ffill' or 'none'")
    return aligned

```

## `src/regime_portfolio/features.py`

```py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureConfig:
    """Windows and conventions for market-regime features."""

    vol_window: int = 21
    corr_window: int = 63
    drawdown_window: int = 252
    momentum_window: int = 63
    annualization: int = 252
    equity_proxy: str = "SPY"


def realized_volatility(
    returns: pd.DataFrame | pd.Series,
    window: int = 21,
    annualization: int = 252,
) -> pd.DataFrame | pd.Series:
    """Rolling annualized realized volatility."""

    return returns.rolling(window).std() * np.sqrt(annualization)


def rolling_drawdown(prices: pd.Series, window: int = 252) -> pd.Series:
    """Rolling drawdown relative to the rolling maximum."""

    prices = prices.astype(float)
    rolling_max = prices.rolling(window=window, min_periods=max(5, window // 5)).max()
    return prices / rolling_max - 1.0


def rolling_average_correlation(returns: pd.DataFrame, window: int = 63) -> pd.Series:
    """Rolling average pairwise correlation across assets.

    A high average correlation is often a useful proxy for diversification
    breakdown during market stress.
    """

    returns = returns.dropna(how="all")
    values: list[float] = []
    index: list[pd.Timestamp] = []
    for end in range(window, len(returns) + 1):
        sample = returns.iloc[end - window : end].dropna(axis=1, how="any")
        date = returns.index[end - 1]
        index.append(date)
        if sample.shape[1] < 2:
            values.append(np.nan)
            continue
        corr = sample.corr().to_numpy()
        mask = ~np.eye(corr.shape[0], dtype=bool)
        values.append(float(np.nanmean(corr[mask])))
    return pd.Series(values, index=pd.Index(index, name=returns.index.name), name="avg_corr")


def downside_semivolatility(
    returns: pd.Series,
    window: int = 63,
    annualization: int = 252,
) -> pd.Series:
    """Rolling annualized semivolatility of negative returns."""

    negative = returns.where(returns < 0.0, 0.0)
    return negative.rolling(window).std() * np.sqrt(annualization)


def momentum(prices: pd.Series | pd.DataFrame, window: int = 63) -> pd.Series | pd.DataFrame:
    """Simple momentum over a lookback window."""

    return prices / prices.shift(window) - 1.0


def make_feature_panel(
    prices: pd.DataFrame,
    returns: pd.DataFrame | None = None,
    config: FeatureConfig | None = None,
    extra_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a compact feature panel for regime detection.

    The default feature set is deliberately interpretable. It should be extended
    only after the baseline pipeline is robust.
    """

    cfg = config or FeatureConfig()
    if returns is None:
        returns = np.log(prices).diff()

    equity = cfg.equity_proxy
    if equity not in prices.columns:
        equity = prices.columns[0]

    equity_returns = returns[equity]
    vol = realized_volatility(equity_returns, cfg.vol_window, cfg.annualization).rename(
        f"{equity}_realized_vol"
    )
    dd = rolling_drawdown(prices[equity], cfg.drawdown_window).rename(f"{equity}_drawdown")
    corr = rolling_average_correlation(returns, cfg.corr_window)
    mom = momentum(prices[equity], cfg.momentum_window).rename(f"{equity}_momentum")
    semivol = downside_semivolatility(
        equity_returns, cfg.corr_window, cfg.annualization
    ).rename(f"{equity}_downside_semivol")

    features = pd.concat([vol, dd, corr, mom, semivol], axis=1)

    if extra_features is not None:
        extra = extra_features.reindex(features.index).ffill()
        features = pd.concat([features, extra], axis=1)

    features = features.replace([np.inf, -np.inf], np.nan)
    return features.dropna(how="any")


def stress_score(features: pd.DataFrame) -> pd.Series:
    """Build a transparent stress score from standardized feature columns.

    Higher values should correspond to more fragile market conditions. The signs
    are chosen heuristically and should be documented in notebooks.
    """

    if features.empty:
        raise ValueError("features must not be empty")

    z = (features - features.mean()) / features.std(ddof=0).replace(0.0, np.nan)
    signed = pd.DataFrame(index=z.index)

    for col in z.columns:
        lower = col.lower()
        sign = 1.0
        if "momentum" in lower:
            sign = -1.0
        signed[col] = sign * z[col]

    score = signed.mean(axis=1).rename("stress_score")
    return score.replace([np.inf, -np.inf], np.nan)

```

## `src/regime_portfolio/metrics.py`

```py
from __future__ import annotations

import numpy as np
import pandas as pd

from regime_portfolio.risk import drawdown_series, historical_cvar


def performance_summary(
    returns: pd.Series,
    annualization: int = 252,
    cvar_alpha: float = 0.05,
) -> pd.Series:
    """Common performance metrics for a daily return series."""

    r = returns.dropna().astype(float)
    if r.empty:
        raise ValueError("returns must not be empty")

    ann_return = float(r.mean() * annualization)
    ann_vol = float(r.std(ddof=1) * np.sqrt(annualization))
    downside = r.where(r < 0.0, 0.0).std(ddof=1) * np.sqrt(annualization)
    dd = drawdown_series(r)
    max_dd = float(dd.min())
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
    sortino = ann_return / downside if downside > 0 else np.nan
    calmar = ann_return / abs(max_dd) if max_dd < 0 else np.nan

    return pd.Series(
        {
            "ann_return": ann_return,
            "ann_vol": ann_vol,
            "sharpe_0rf": sharpe,
            "sortino_0rf": sortino,
            "max_drawdown": max_dd,
            "calmar": calmar,
            "cvar_5pct": historical_cvar(r, cvar_alpha),
            "skew": float(r.skew()),
            "kurtosis": float(r.kurtosis()),
        }
    )


def compare_strategies(strategy_returns: pd.DataFrame) -> pd.DataFrame:
    """Return a performance table for several strategies."""

    rows = {name: performance_summary(strategy_returns[name]) for name in strategy_returns.columns}
    return pd.DataFrame(rows).T.sort_values("sharpe_0rf", ascending=False)


def turnover(weights: pd.DataFrame) -> pd.Series:
    """One-way portfolio turnover from a weight history."""

    return weights.diff().abs().sum(axis=1) / 2.0

```

## `src/regime_portfolio/regimes.py`

```py
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from regime_portfolio.features import stress_score


@dataclass
class RegimeDetector:
    """Gaussian HMM regime detector with economic relabeling.

    Raw HMM states are arbitrary labels. After fitting, this class orders states
    from low to high stress using either a supplied risk proxy or a transparent
    stress score computed from the features.
    """

    n_regimes: int = 3
    covariance_type: str = "full"
    n_iter: int = 1_000
    random_state: int = 42
    scaler: StandardScaler = field(init=False, default_factory=StandardScaler)
    model_: object | None = field(init=False, default=None)
    state_mapping_: dict[int, int] = field(init=False, default_factory=dict)
    fitted_index_: pd.Index | None = field(init=False, default=None)
    feature_columns_: list[str] = field(init=False, default_factory=list)

    def fit(self, features: pd.DataFrame, risk_proxy: pd.Series | None = None) -> "RegimeDetector":
        """Fit the HMM and construct an economic state ordering."""

        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install hmmlearn to use RegimeDetector: pip install hmmlearn") from exc

        x = features.dropna(how="any").copy()
        if x.shape[0] < max(100, 10 * self.n_regimes):
            raise ValueError("Not enough observations to fit the HMM reliably.")

        self.feature_columns_ = list(x.columns)
        x_scaled = self.scaler.fit_transform(x)
        self.model_ = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        self.model_.fit(x_scaled)
        raw_states = pd.Series(self.model_.predict(x_scaled), index=x.index, name="raw_state")

        if risk_proxy is None:
            proxy = stress_score(x)
        else:
            proxy = risk_proxy.reindex(x.index).ffill().bfill()

        state_scores = proxy.groupby(raw_states).mean().sort_values()
        self.state_mapping_ = {int(raw): int(rank) for rank, raw in enumerate(state_scores.index)}
        self.fitted_index_ = x.index
        return self

    def _check_fitted(self) -> None:
        if self.model_ is None:
            raise RuntimeError("RegimeDetector is not fitted yet.")

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Predict ordered regimes, where 0 is lowest stress."""

        self._check_fitted()
        x = features[self.feature_columns_].dropna(how="any")
        raw = self.model_.predict(self.scaler.transform(x))  # type: ignore[union-attr]
        ordered = [self.state_mapping_.get(int(s), int(s)) for s in raw]
        return pd.Series(ordered, index=x.index, name="regime")

    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        """Return ordered regime probabilities."""

        self._check_fitted()
        x = features[self.feature_columns_].dropna(how="any")
        raw_proba = self.model_.predict_proba(self.scaler.transform(x))  # type: ignore[union-attr]
        out = pd.DataFrame(index=x.index)
        inverse_mapping = {ordered: raw for raw, ordered in self.state_mapping_.items()}
        for ordered_state in range(self.n_regimes):
            raw_state = inverse_mapping.get(ordered_state, ordered_state)
            out[f"regime_{ordered_state}_prob"] = raw_proba[:, raw_state]
        return out

    def transition_matrix(self) -> pd.DataFrame:
        """Return the HMM transition matrix with economically ordered states."""

        self._check_fitted()
        raw_matrix = np.asarray(self.model_.transmat_)  # type: ignore[union-attr]
        matrix = np.zeros_like(raw_matrix)
        for raw_i, ordered_i in self.state_mapping_.items():
            for raw_j, ordered_j in self.state_mapping_.items():
                matrix[ordered_i, ordered_j] = raw_matrix[raw_i, raw_j]
        labels = [f"regime_{i}" for i in range(self.n_regimes)]
        return pd.DataFrame(matrix, index=labels, columns=labels)


def regime_durations(regimes: pd.Series) -> pd.DataFrame:
    """Compute empirical regime spell durations."""

    regimes = regimes.dropna().astype(int)
    if regimes.empty:
        return pd.DataFrame(columns=["regime", "duration"])
    switches = regimes.ne(regimes.shift()).cumsum()
    spells = regimes.groupby(switches).agg(regime="first", duration="size")
    return spells.reset_index(drop=True)

```

## `src/regime_portfolio/risk.py`

```py
from __future__ import annotations

import numpy as np
import pandas as pd


def annualized_return(returns: pd.Series, annualization: int = 252) -> float:
    """Annualized arithmetic mean return."""

    return float(returns.mean() * annualization)


def annualized_volatility(returns: pd.Series, annualization: int = 252) -> float:
    """Annualized volatility."""

    return float(returns.std(ddof=1) * np.sqrt(annualization))


def historical_cvar(returns: pd.Series, alpha: float = 0.05) -> float:
    """Historical conditional value-at-risk of returns.

    Returned value is a positive loss number when the lower tail is negative.
    """

    x = returns.dropna().sort_values()
    if x.empty:
        return np.nan
    n_tail = max(1, int(np.ceil(alpha * len(x))))
    return float(-x.iloc[:n_tail].mean())


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Drawdown series from simple or log returns.

    For small daily returns this is robust enough for diagnostics. If returns are
    log returns, exponentiating cumulative sums gives the wealth curve.
    """

    wealth = np.exp(returns.fillna(0.0).cumsum())
    peak = wealth.cummax()
    return wealth / peak - 1.0


def max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown as a negative number."""

    dd = drawdown_series(returns)
    return float(dd.min()) if not dd.empty else np.nan


def covariance_by_regime(
    returns: pd.DataFrame,
    regimes: pd.Series,
    annualization: int = 252,
) -> dict[int, pd.DataFrame]:
    """Estimate annualized covariance matrix by regime."""

    aligned = returns.join(regimes.rename("regime"), how="inner").dropna()
    out: dict[int, pd.DataFrame] = {}
    for regime, sample in aligned.groupby("regime"):
        r = sample.drop(columns="regime")
        out[int(regime)] = r.cov() * annualization
    return out


def correlation_by_regime(returns: pd.DataFrame, regimes: pd.Series) -> dict[int, pd.DataFrame]:
    """Estimate correlation matrix by regime."""

    aligned = returns.join(regimes.rename("regime"), how="inner").dropna()
    out: dict[int, pd.DataFrame] = {}
    for regime, sample in aligned.groupby("regime"):
        r = sample.drop(columns="regime")
        out[int(regime)] = r.corr()
    return out


def regime_risk_summary(
    returns: pd.DataFrame,
    regimes: pd.Series,
    portfolio_returns: pd.Series | None = None,
    annualization: int = 252,
    cvar_alpha: float = 0.05,
) -> pd.DataFrame:
    """Compute interpretable risk diagnostics by regime."""

    if portfolio_returns is None:
        portfolio_returns = returns.mean(axis=1)

    aligned = pd.concat(
        [returns, portfolio_returns.rename("portfolio"), regimes.rename("regime")], axis=1
    ).dropna()
    rows: list[dict[str, float | int]] = []
    for regime, sample in aligned.groupby("regime"):
        pr = sample["portfolio"]
        rows.append(
            {
                "regime": int(regime),
                "n_obs": int(len(sample)),
                "ann_return": annualized_return(pr, annualization),
                "ann_vol": annualized_volatility(pr, annualization),
                "sharpe_0rf": annualized_return(pr, annualization)
                / max(annualized_volatility(pr, annualization), 1e-12),
                "cvar_5pct": historical_cvar(pr, cvar_alpha),
                "max_drawdown": max_drawdown(pr),
                "avg_pairwise_corr": _average_pairwise_corr(sample[returns.columns]),
            }
        )
    return pd.DataFrame(rows).sort_values("regime").reset_index(drop=True)


def _average_pairwise_corr(sample: pd.DataFrame) -> float:
    corr = sample.corr().to_numpy()
    if corr.shape[0] < 2:
        return np.nan
    mask = ~np.eye(corr.shape[0], dtype=bool)
    return float(np.nanmean(corr[mask]))

```

## `src/regime_portfolio/smdp_policy.py`

```py
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from regime_portfolio.features import stress_score


@dataclass(frozen=True)
class StressPolicyConfig:
    """Configuration for a transparent stress-state policy."""

    n_states: int = 4
    calm_budget: float = 1.0
    elevated_budget: float = 0.7
    stress_budget: float = 0.4
    crisis_budget: float = 0.15
    drawdown_penalty: float = 2.0
    cvar_penalty: float = 1.0
    turnover_penalty: float = 0.1

    @property
    def default_budgets(self) -> dict[int, float]:
        return {
            0: self.calm_budget,
            1: self.elevated_budget,
            2: self.stress_budget,
            3: self.crisis_budget,
        }


def build_stress_states(
    features: pd.DataFrame,
    n_states: int = 4,
    score: pd.Series | None = None,
) -> pd.Series:
    """Discretize a continuous stress score into ordered stress states.

    State 0 is lowest stress and state n_states - 1 is highest stress.
    """

    if score is None:
        score = stress_score(features)
    score = score.dropna()
    if score.nunique() < n_states:
        ranks = score.rank(method="first")
    else:
        ranks = score
    states = pd.qcut(ranks, q=n_states, labels=False, duplicates="drop")
    return states.astype(int).rename("stress_state")


def estimate_transition_matrix(states: pd.Series, n_states: int | None = None) -> pd.DataFrame:
    """Empirical transition matrix for ordered stress states."""

    s = states.dropna().astype(int)
    if n_states is None:
        n_states = int(s.max()) + 1
    matrix = np.zeros((n_states, n_states), dtype=float)
    for current, nxt in zip(s.iloc[:-1], s.iloc[1:], strict=False):
        matrix[int(current), int(nxt)] += 1.0
    row_sums = matrix.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        matrix = np.divide(matrix, row_sums, where=row_sums > 0)
    for i in range(n_states):
        if matrix[i].sum() == 0:
            matrix[i, i] = 1.0
    labels = [f"state_{i}" for i in range(n_states)]
    return pd.DataFrame(matrix, index=labels, columns=labels)


def threshold_risk_budget_policy(
    states: pd.Series,
    budgets: Mapping[int, float] | None = None,
) -> pd.Series:
    """Map stress states to risk budgets using a threshold-like policy."""

    mapping = {0: 1.0, 1: 0.7, 2: 0.4, 3: 0.15} if budgets is None else dict(budgets)
    return states.map(lambda x: mapping.get(int(x), min(mapping.values()))).rename("risk_budget")


def stress_adjusted_reward(
    portfolio_returns: pd.Series,
    stress_states: pd.Series,
    config: StressPolicyConfig | None = None,
) -> pd.Series:
    """Construct a stylized reward penalizing returns earned under severe stress.

    This is an interpretable placeholder before full SMDP policy iteration. It
    helps compare policies in a way aligned with drawdown and stress aversion.
    """

    cfg = config or StressPolicyConfig()
    aligned = pd.concat([portfolio_returns.rename("ret"), stress_states.rename("state")], axis=1).dropna()
    severity = aligned["state"] / max(1, cfg.n_states - 1)
    reward = aligned["ret"] - cfg.drawdown_penalty * severity * aligned["ret"].clip(upper=0.0).abs()
    return reward.rename("stress_adjusted_reward")


def value_iteration(
    transition_by_action: Mapping[str, pd.DataFrame],
    reward_by_action: Mapping[str, pd.Series | np.ndarray | list[float]],
    gamma: float = 0.95,
    tolerance: float = 1e-8,
    max_iter: int = 10_000,
) -> tuple[pd.Series, dict[int, str]]:
    """Finite-state discounted value iteration.

    This is included as a transparent stepping stone toward genuine SMDP policy
    iteration. It can be used once action-dependent transition matrices are
    estimated or simulated.
    """

    actions = list(transition_by_action)
    if not actions:
        raise ValueError("At least one action is required")
    n_states = transition_by_action[actions[0]].shape[0]
    value = np.zeros(n_states)

    rewards: dict[str, np.ndarray] = {}
    for action, r in reward_by_action.items():
        rewards[action] = np.asarray(r, dtype=float)
        if rewards[action].shape[0] != n_states:
            raise ValueError(f"Reward for action {action} has wrong length")

    for _ in range(max_iter):
        candidate_values = []
        for action in actions:
            p = transition_by_action[action].to_numpy(dtype=float)
            candidate_values.append(rewards[action] + gamma * p @ value)
        stacked = np.vstack(candidate_values)
        new_value = stacked.max(axis=0)
        if np.max(np.abs(new_value - value)) < tolerance:
            value = new_value
            break
        value = new_value

    policy: dict[int, str] = {}
    for state in range(n_states):
        action_values = {
            action: rewards[action][state]
            + gamma * transition_by_action[action].to_numpy(dtype=float)[state] @ value
            for action in actions
        }
        policy[state] = max(action_values, key=action_values.get)

    return pd.Series(value, index=[f"state_{i}" for i in range(n_states)], name="value"), policy

```

## `tests/test_backtest.py`

```py
from __future__ import annotations

import numpy as np
import pandas as pd

from regime_portfolio.allocation import make_regime_weight_function
from regime_portfolio.backtest import run_backtest


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

```

## `tests/test_features.py`

```py
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

```

## `tests/test_risk.py`

```py
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

```
