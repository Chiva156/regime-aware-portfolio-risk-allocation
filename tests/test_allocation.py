import numpy as np
import pandas as pd

from regime_portfolio.allocation import minimum_variance_weight


def test_minimum_variance_weight_prefers_lower_volatility_asset():
    rng = np.random.default_rng(42)

    returns = pd.DataFrame(
        {
            "HIGH_VOL": rng.normal(0.0, 0.040, size=500),
            "MID_VOL": rng.normal(0.0, 0.020, size=500),
            "LOW_VOL": rng.normal(0.0, 0.005, size=500),
        }
    )

    weights = minimum_variance_weight(returns)

    assert np.isclose(weights.sum(), 1.0)
    assert (weights >= 0.0).all()
    assert weights["LOW_VOL"] > weights["MID_VOL"]
    assert weights["MID_VOL"] > weights["HIGH_VOL"]
    assert weights.attrs["optimizer_status"].startswith("success")