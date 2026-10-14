"""Amendment 7: export parsing and the reporting rule (synthetic data)."""
import numpy as np
import pandas as pd

from src import backtest_2026


def test_export_percent_strings_become_fractions(tmp_path, monkeypatch):
    f = tmp_path / "pr_pitcher_2026.xlsx"
    pd.DataFrame({"球員": ["甲", "乙"], "打席": [120, 80], "三振%": ["25.0%", "18.5%"],
                  "保送%": ["8.0%", "5.0%"], "揮空%": [30.0, 22.0]}).to_excel(f, index=False)
    monkeypatch.setattr(backtest_2026, "raw", lambda *a: tmp_path)
    d = backtest_2026.load_export("pitcher_2026")
    assert list(d.site_name) == ["甲", "乙"]
    assert np.allclose(d.k_pct, [0.25, 0.185]) and np.allclose(d.whiff_pct, [0.30, 0.22])
    assert backtest_2026._league(d, "k_pct") == (0.25 * 120 + 0.185 * 80) / 200


def test_small_cohort_is_descriptive_only():
    d = pd.DataFrame({"x": np.arange(10.0), "y": np.arange(10.0)})
    out = backtest_2026._tests(d, [("t", "x", "y", 1)], np.random.default_rng(0))
    assert out[0]["n"] == 10 and np.isclose(out[0]["rho"], 1.0)
    assert out[0]["verdict"].startswith("descriptive only")


def test_residual_bootstrap_refits_the_baseline_regression():
    from src import backtest
    rng = np.random.default_rng(1)
    base = rng.normal(size=200)
    x = rng.normal(size=200)
    d = pd.DataFrame({"x": x, "base": base, "y": 2 * base + x + rng.normal(scale=0.5, size=200)})
    est, lo, hi = backtest.boot_resid_ci(d, "x", "base", "y", 300, 0.9, np.random.default_rng(0))
    assert est > 0.7 and lo > 0.5 and hi <= 1
    noise = d.assign(x=rng.normal(size=200))
    est, lo, hi = backtest.boot_resid_ci(noise, "x", "base", "y", 300, 0.9, np.random.default_rng(0))
    assert lo < 0 < hi
