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
    method: str = "tail",
    quantiles: tuple[float, ...] | None = None,
    smoothing_window: int = 21,
) -> pd.Series:
    """Discretize a continuous stress score into ordered stress states.

    State 0 is lowest stress and state n_states - 1 is highest stress.

    Parameters
    ----------
    features:
        Feature panel used to compute the stress score when ``score`` is not provided.
    n_states:
        Number of ordered stress states.
    score:
        Optional precomputed continuous stress score.
    method:
        ``"equal_frequency"`` creates equally populated states.
        ``"tail"`` creates rarer high-stress states using upper-tail quantiles.
    quantiles:
        Threshold quantiles for ``method="tail"``. For four states, the default
        is ``(0.50, 0.75, 0.90)``, producing normal/elevated/fragile/crisis states.
    smoothing_window:
        Rolling window used to smooth the stress score before discretization.
    """
    if score is None:
        score = stress_score(features)

    smoothed_score = (
        score
        .dropna()
        .rolling(window=smoothing_window, min_periods=max(2, smoothing_window // 3))
        .mean()
        .dropna()
    )

    if smoothed_score.empty:
        raise ValueError("stress score is empty after smoothing")

    if method == "equal_frequency":
        if smoothed_score.nunique() < n_states:
            ranks = smoothed_score.rank(method="first")
        else:
            ranks = smoothed_score

        states = pd.qcut(
            ranks,
            q=n_states,
            labels=False,
            duplicates="drop",
        )

    elif method == "tail":
        if n_states != 4 and quantiles is None:
            raise ValueError(
                "Default tail thresholds are defined for n_states=4. "
                "Pass explicit quantiles for a different number of states."
            )

        if quantiles is None:
            quantiles = (0.50, 0.75, 0.90)

        if len(quantiles) != n_states - 1:
            raise ValueError("quantiles must contain n_states - 1 thresholds")

        thresholds = smoothed_score.quantile(list(quantiles)).to_numpy()

        states = pd.Series(
            np.searchsorted(thresholds, smoothed_score.to_numpy(), side="right"),
            index=smoothed_score.index,
            name="stress_state",
        )

    else:
        raise ValueError(f"Unknown stress-state method: {method}")

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
        matrix = np.divide(
            matrix,
            row_sums,
            out=np.zeros_like(matrix, dtype=float),
            where=row_sums > 0,
        )
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
