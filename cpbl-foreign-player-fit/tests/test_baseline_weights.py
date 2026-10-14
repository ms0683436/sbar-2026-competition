"""Amendment 4: baseline weights estimated from 2025 -> 2026 AAA data (synthetic data only)."""
import numpy as np
import pandas as pd
import pytest

from src import baseline, baseline_weights as bw
from src.common import config


def _split(pid, league_id, **stat):
    return {"season": "2025", "player": {"id": pid, "fullName": f"P {pid}"}, "team": {"id": 1, "name": "T"},
            "league": {"id": league_id}, "position": {"abbreviation": "P"}, "stat": stat}


def test_to_frame_keeps_aaa_leagues_and_converts_outs_to_ip():
    df = bw.to_frame([_split(1, 112, outs=181, strikeOuts=50), _split(1, 117, outs=3), _split(2, 999, outs=9)],
                     "pitching")
    assert df.team_league.tolist() == ["PCL", "INT"]
    assert df.pitching_IP.iat[0] == pytest.approx(181 / 3)
    assert df.pitching_SO.iat[0] == 50 and df.pitching_SO.iat[1] == 0


def test_nnls_recovers_nonnegative_weights_and_zeroes_harmful_ones():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 3))
    y = 0.6 * X[:, 0] + 0.2 * X[:, 1] - 0.3 * X[:, 2] + 5 + rng.normal(scale=0.1, size=4000)
    c = bw.nnls_weights(X, y, np.ones(4000))
    assert c.sum() == pytest.approx(1)
    assert c[2] == 0
    assert c[0] / c[1] == pytest.approx(3, rel=0.05)


def test_nnls_uses_player_weights():
    X = np.array([[1.0, 0], [0, 1], [-1, 0], [0, -1]])
    y = np.array([1.0, 0, -1, 0])          # only predictor 0 matters for rows 0/2
    w = np.array([1.0, 1, 1, 1])
    assert bw.nnls_weights(X, y, w)[0] == pytest.approx(1)
    assert bw.nnls_weights(X, -y, w) is None   # every coefficient 0 -> no estimate


def _sample(n, rng, in_sample=True):
    spec = config()["baseline"]["pitcher"]
    d = pd.DataFrame({m: rng.normal(size=n) for m in spec})
    d["target"] = d.k_pct_adj + rng.normal(scale=0.5, size=n)
    d["target_n"] = 300.0
    d["in_sample"] = in_sample
    d["n_predictor_pool"] = n
    return d


def test_estimate_falls_back_to_judgment_weights_below_min_n():
    n = config()["baseline"]["estimation"]["min_n"] - 1
    t = bw.estimate("pitcher", _sample(n, np.random.default_rng(1)))
    judg = baseline.judgment_weights("pitcher")
    assert (t.used == "judgment (fallback)").all()
    assert np.allclose(t.used_weight, [judg[m] for m in t.metric])


def test_estimate_puts_weight_on_the_predictive_metric():
    t = bw.estimate("pitcher", _sample(300, np.random.default_rng(2))).set_index("metric")
    assert (t.used == "estimated").all()
    assert t.used_weight.sum() == pytest.approx(1)
    assert t.used_weight.idxmax() == "k_pct_adj"
    assert t.loc["k_pct_adj", "ci_lo"] > 0.5
    assert t.cv_spearman_estimated.iat[0] > t.spearman_judgment.iat[0]


def test_score_uses_the_registered_weight_source(monkeypatch):
    rng = np.random.default_rng(3)
    d = _sample(30, rng)
    monkeypatch.setitem(config()["baseline"], "weights_source", "judgment")
    a = baseline.score(d, "pitcher").baseline_score
    assert np.allclose(a, baseline.score(d, "pitcher", baseline.judgment_weights("pitcher")).baseline_score)
    monkeypatch.setitem(config()["baseline"], "weights_source", "estimated")
    fake = {m: (1.0 if m == "fip" else 0.0) for m in config()["baseline"]["pitcher"]}
    monkeypatch.setattr(bw, "weights", lambda role: fake)
    s = baseline.score(d, "pitcher")
    assert np.allclose(s.baseline_score, s.z_fip)
    monkeypatch.setitem(config()["baseline"], "weights_source", {"pitcher": "judgment", "hitter": "estimated"})
    assert baseline.active_weights("pitcher") == baseline.judgment_weights("pitcher")
    assert baseline.active_weights("hitter") == fake


def test_fetch_keeps_one_row_per_team_stint(monkeypatch, tmp_path):
    """A traded player must keep his league mix: each club is queried on its own."""
    from src import statsapi_pbp
    monkeypatch.setattr(bw, "raw", lambda *parts: tmp_path.joinpath(*parts))

    def fake_get(url, params=None):
        if url.endswith("/teams"):
            return {"teams": [{"id": 1}, {"id": 2}]}
        lg, pa = (112, 238) if params["teamId"] == 1 else (117, 45)
        sp = _split(7, lg, plateAppearances=pa)
        sp["team"] = {"id": params["teamId"], "name": f"T{params['teamId']}"}
        return {"stats": [{"totalSplits": 1, "splits": [sp]}]}
    monkeypatch.setattr(statsapi_pbp, "_get", fake_get)
    df = bw.fetch("hitting", 2026)
    assert sorted(zip(df.team_league, df.batting_PA)) == [("INT", 45), ("PCL", 238)]


def test_design_standardises_with_the_fitting_rows_only():
    rng = np.random.default_rng(4)
    d = _sample(50, rng)
    train, test = d.iloc[:40], d.iloc[40:]
    X = bw.design("pitcher", train, test)
    mu, sd = train.k_pct_adj.mean(), train.k_pct_adj.std(ddof=0)
    col = list(config()["baseline"]["pitcher"]).index("k_pct_adj")
    assert np.allclose(X[:, col], ((test.k_pct_adj - mu) / sd).clip(-2.5, 2.5))


def test_failed_bootstrap_draws_are_not_counted_as_nonzero(monkeypatch):
    names = list(config()["baseline"]["pitcher"])
    calls = {"n": 0}

    def fake(X, y, w):
        calls["n"] += 1
        if calls["n"] > 1 and calls["n"] % 2 == 0:
            return None                              # every other bootstrap draw fails
        c = np.zeros(len(names)); c[1] = 1.0
        return c
    monkeypatch.setattr(bw, "nnls_weights", fake)
    monkeypatch.setattr(bw, "cv_spearman", lambda *a, **k: np.nan)
    t = bw.estimate("pitcher", _sample(60, np.random.default_rng(5))).set_index("metric")
    draws = config()["baseline"]["estimation"]["bootstrap"]["draws"]
    assert t.boot_draws_all_zero.iat[0] == draws // 2
    assert t.loc[names[0], "share_zero"] == 1.0 and t.loc[names[1], "share_zero"] == 0.0
