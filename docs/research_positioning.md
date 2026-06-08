# Research positioning note

## Core idea

This project translates degradation-control thinking into portfolio risk management.

In a degradation model, a system moves through increasingly damaged states. A controller chooses an operating intensity and occasionally performs maintenance. The objective is to balance revenue, deterioration, and intervention costs.

In this portfolio project, markets move through increasingly fragile regimes. A controller chooses a risk budget and occasionally de-risks. The objective is to balance expected return, volatility, drawdown, tail risk, and transaction costs.

## What to emphasize in the README

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
