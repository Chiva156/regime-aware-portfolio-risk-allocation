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


def smooth_regime_probabilities(
    probabilities: pd.DataFrame,
    window: int = 5,
) -> pd.DataFrame:
    """Smooth regime probabilities using a rolling mean."""
    return probabilities.rolling(window=window, min_periods=1).mean()


def apply_minimum_regime_duration(
    regimes: pd.Series,
    min_duration: int = 5,
) -> pd.Series:
    """Merge very short regime runs into the previous regime."""
    regimes = regimes.copy().dropna().astype(int)

    if regimes.empty:
        return regimes

    values = regimes.to_numpy().copy()
    start = 0

    for i in range(1, len(values) + 1):
        if i == len(values) or values[i] != values[start]:
            run_length = i - start

            if run_length < min_duration and start > 0:
                values[start:i] = values[start - 1]

            start = i

    return pd.Series(values, index=regimes.index, name=regimes.name)
