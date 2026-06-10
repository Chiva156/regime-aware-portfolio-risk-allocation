from __future__ import annotations

import numpy as np
import pandas as pd


def regime_run_lengths(regimes: pd.Series) -> pd.DataFrame:
    """Compute consecutive run lengths for a regime sequence."""
    regimes = regimes.dropna().astype(int)
    if regimes.empty:
        return pd.DataFrame(columns=["regime", "run_length"])

    groups = regimes.ne(regimes.shift()).cumsum()
    runs = (
        regimes.groupby(groups)
        .agg(regime="first", run_length="size")
        .reset_index(drop=True)
    )
    return runs


def regime_duration_summary(regimes: pd.Series) -> pd.DataFrame:
    """Summarize regime durations."""
    runs = regime_run_lengths(regimes)
    if runs.empty:
        return pd.DataFrame()

    return (
        runs.groupby("regime")["run_length"]
        .describe()
        .rename_axis("regime")
    )


def transition_matrix(states: pd.Series, n_states: int | None = None) -> pd.DataFrame:
    """Estimate empirical transition matrix from a discrete state sequence."""
    states = states.dropna().astype(int)

    if n_states is None:
        n_states = int(states.max()) + 1

    counts = np.zeros((n_states, n_states), dtype=float)

    values = states.to_numpy()
    for current, nxt in zip(values[:-1], values[1:]):
        counts[current, nxt] += 1.0

    row_sums = counts.sum(axis=1, keepdims=True)
    probs = np.divide(
        counts,
        row_sums,
        out=np.zeros_like(counts, dtype=float),
        where=row_sums > 0,
    )

    labels = [f"state_{i}" for i in range(n_states)]
    return pd.DataFrame(probs, index=labels, columns=labels)
