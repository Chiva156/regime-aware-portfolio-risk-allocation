import pandas as pd

from regime_portfolio.diagnostics import regime_run_lengths, regime_duration_summary


def test_regime_run_lengths():
    regimes = pd.Series([0, 0, 1, 1, 1, 0])
    runs = regime_run_lengths(regimes)

    assert runs["regime"].tolist() == [0, 1, 0]
    assert runs["run_length"].tolist() == [2, 3, 1]


def test_regime_duration_summary():
    regimes = pd.Series([0, 0, 1, 1, 1, 0])
    summary = regime_duration_summary(regimes)

    assert 0 in summary.index
    assert 1 in summary.index
