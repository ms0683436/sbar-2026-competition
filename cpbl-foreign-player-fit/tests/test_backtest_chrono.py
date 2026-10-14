"""Amendment 11: reference-pool scaling, paired difference and rolling-origin forecast (synthetic data)."""
import numpy as np
import pandas as pd

from src import backtest_chrono as bc


def test_z_uses_the_reference_pool_not_the_cohort():
    pool = pd.Series([0.0, 1.0, 2.0, 3.0, 4.0])
    z = bc._z(pd.Series([2.0, 10.0]), pool, clip=2.5)
    assert np.isclose(z.iat[0], 0.0) and z.iat[1] == 2.5


def test_paired_difference_is_zero_for_identical_predictors():
    rng = np.random.default_rng(0)
    d = pd.DataFrame({"a": rng.normal(size=30), "y": rng.normal(size=30)})
    d["b"] = d.a
    r = bc.paired_diff(d, "a", "b", "y", np.random.default_rng(1))
    assert r["rho"] == 0 and r["ci_lo"] == 0 and r["ci_hi"] == 0


def test_rolling_forecast_trains_only_on_earlier_arrivals():
    coh = pd.DataFrame({"season": [2024] * 4 + [2025] * 2 + [2026],
                        "english_name": list("abcdefg"),
                        "baseline_score": [0.0, 1.0, 2.0, 3.0, 1.0, 2.0, 0.0],
                        "cpbl_kbb_c": [0.0, 0.1, 0.2, 0.3, 9.0, 9.0, 0.0]})
    f = bc.rolling_forecast(coh, np.random.default_rng(0))
    rows = pd.DataFrame(f["rows"])
    r25 = rows[rows.origin == 2025]
    assert (r25.n_train == 4).all() and np.allclose(r25.m1, [0.1, 0.2])   # the 9.0 outcomes never leak in
    assert np.isclose(r25.m0.iat[0], 0.15)
    assert rows[rows.origin == 2026].n_train.iat[0] == 6
