"""Amendment 7: export parsing and the reporting rule (synthetic data)."""
import numpy as np
import pandas as pd

from src import backtest_2026


def test_export_percent_strings_become_fractions(tmp_path, monkeypatch):
    f = tmp_path / "pr_pitcher_2026.csv"
    pd.DataFrame({"球員": ["甲", "乙"], "打席": [120, 80], "三振%": ["25.0%", "18.5%"],
                  "保送%": ["8.0%", "5.0%"], "揮空%": [30.0, 22.0]}).to_csv(f, index=False)
    monkeypatch.setattr(backtest_2026, "raw", lambda *a: tmp_path)
    d = backtest_2026.load_export("pitcher")
    assert list(d.site_name) == ["甲", "乙"]
    assert np.allclose(d.k_pct, [0.25, 0.185]) and np.allclose(d.whiff_pct, [0.30, 0.22])
    assert backtest_2026._league(d, "k_pct") == (0.25 * 120 + 0.185 * 80) / 200


def test_small_cohort_is_descriptive_only():
    d = pd.DataFrame({"x": np.arange(10.0), "y": np.arange(10.0)})
    out = backtest_2026._tests(d, [("t", "x", "y", 1)], np.random.default_rng(0))
    assert out[0]["n"] == 10 and np.isclose(out[0]["rho"], 1.0)
    assert out[0]["verdict"].startswith("descriptive only")
