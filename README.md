Regime-Aware Portfolio Risk and Allocation

This project studies dynamic multi-asset allocation under latent market regimes. It combines hidden Markov regime detection, regime-conditioned risk estimation, and a stress-aware allocation policy inspired by semi-Markov degradation-control models. The central idea is to treat portfolio stress as an accumulating latent state: as volatility, drawdowns, and correlation concentration worsen, the portfolio moves through increasingly fragile regimes, and the allocation engine decides whether to maintain, reduce, or defensively rebalance risk exposure.

The project compares robust constant-risk policies against regime-dependent policies, evaluates performance with walk-forward backtests, and focuses especially on drawdown and CVaR reduction during stressed market regimes.
