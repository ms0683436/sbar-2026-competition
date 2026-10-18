"""Amendment 16 (KBO): helpers on synthetic data, no downloads."""
import numpy as np
import pandas as pd

from src import kbo


def test_age_on_june_30_convention():
    assert kbo._age_on("1994-03-08", 2025) == 31
    assert kbo._age_on("1994-07-01", 2025) == 30          # birthday after 30 June: not yet
    assert kbo._age_on("1994-06-30", 2025) == 31
    assert np.isnan(kbo._age_on(None, 2025))


def test_norm_strips_accents_and_punctuation():
    assert kbo._norm("Raúl Alcántara") == kbo._norm("raul alcantara") == "raulalcantara"
    assert kbo._norm("Jack O'Loughlin") == "jackoloughlin"


def test_verdict_needs_fifteen():
    assert kbo._verdict(14, 0.1, 0.5).startswith("descriptive")
    assert kbo._verdict(15, 0.1, 0.5) == "supports"
    assert kbo._verdict(15, -0.5, -0.1) == "contradicts"
    assert kbo._verdict(15, -0.1, 0.5) == "inconclusive"


def test_cluster_bootstrap_draws_whole_players():
    d = pd.DataFrame({"p": [1, 1, 2, 3, 3, 3]})
    for i in kbo._draws(d, "p", np.random.default_rng(0)):
        got = d.p.to_numpy()[i]
        for p in set(got):                                 # every drawn player brings all his rows
            assert (got == p).sum() % (d.p == p).sum() == 0
        break


def test_rolling_uses_mean_below_min_train_and_counts_no_fitted():
    rng = np.random.default_rng(0)
    d = pd.DataFrame({"dest": np.repeat([2018, 2019, 2020], 5), "x": rng.normal(size=15), "y": rng.normal(size=15)})
    r = kbo.rolling(d, "x", "y", np.random.default_rng(1))
    assert r["n_held_out"] == 10 and r["n_fitted"] == 0     # never 15 earlier pitchers
    assert r["diff"] == 0 and r["verdict"] == "not established"


def test_loo_line_beats_mean_on_a_clean_line():
    x = np.linspace(-1, 1, 30)
    d = pd.DataFrame({"x": x, "y": 2 * x + 0.01 * np.sin(np.arange(30))})
    r = kbo.loo(d, "x", "y", np.random.default_rng(0))
    assert r["n_fitted"] == 30 and r["diff"] < 0 and r["verdict"] == "predictive improvement"


def test_pct_is_share_at_or_below():
    assert kbo._pct(0.02, np.array([0.01, 0.02, 0.03, 0.04])) == 50.0


# ------------------------------------------------------------------ amendments 17-18
def test_zref_uses_reference_moments():
    from src import conversion_pool
    ref = pd.Series([0.0, 2.0])
    assert np.allclose(conversion_pool._zref(pd.Series([1.0, 3.0]), ref), [0.0, 2.0])
    assert conversion_pool._zref(pd.Series([10.0]), ref, 2.5).iat[0] == 2.5


def test_stuff_regression_weights_and_shrinkage(monkeypatch):
    from src import stuff_rank
    stuff_rank._R.clear()
    stuff_rank._R.update({"R": 100.0})
    # two seasons; pitcher 1 has a better (lower) run value in both, pitcher 2 average, 3 worse
    rows = []
    for season in (2025, 2026):
        for pid, rv in ((1, -1.0), (2, 0.0), (3, 1.0)):
            rows.append({"pitcher_id": str(pid), "season": season, "game_pk": season * 10 + pid, "n": 400, "s": rv * 400})
    rows.append({"pitcher_id": "4", "season": 2026, "game_pk": 99, "n": 10, "s": -10.0})   # tiny sample
    g = pd.DataFrame(rows)
    c = stuff_rank.candidate_stuff(g, seasons=[2026, 2025, 2024]).set_index("pitcher_id").stuff_plus
    assert c.loc[1] > 100 > c.loc[3] and abs(c.loc[2] - 100) < 1e-9
    raw1 = stuff_rank.season_stuff(g[g.pitcher_id != "4"]).query("pitcher_id == '1'").stuff_plus.iat[0]
    assert np.isclose(c.loc[1], (raw1 * (3 * 400 + 2 * 400) + 100 * 100) / (3 * 400 + 2 * 400 + 100), atol=0.5)
    assert abs(c.loc[4] - 100) < abs(c.loc[1] - 100)          # 10 pitches: pulled most of the way to 100
    stuff_rank._R.clear()


def test_zref_zero_spread_is_average_not_inf():
    from src import conversion_pool
    z = conversion_pool._zref(pd.Series([1.0, 5.0]), pd.Series([2.0, 2.0]))
    assert (z == 0).all()


def test_additive_scale_can_raise_a_zero_count():
    from src import marcel
    s = pd.DataFrame({"player_id": [1, 2], "season": [2025, 2025], "team_id": [7, 7],
                      "pitching_BB": [0.0, 10.0], "pitching_SO": [20.0, 30.0]})
    sc = pd.DataFrame({"player_id": [1], "season": [2025], "team_id": [7], "+pitching_BB": [1.5], "+pitching_SO": [-2.0]}
                      ).set_index(["player_id", "season", "team_id"])
    out = marcel._scaled(s, sc)
    assert out.pitching_BB.tolist() == [1.5, 10.0] and out.pitching_SO.tolist() == [18.0, 30.0]


def test_rho_test_constant_input_is_undefined():
    d = pd.DataFrame({"x": [1.0] * 20, "y": np.arange(20.0)})
    r = kbo.rho_test("const", d, "x", "y", np.random.default_rng(0))
    assert r["verdict"] in ("undefined", "too few") or np.isnan(r["rho"])


def test_stuff_zero_exposure_pitcher_gets_100(monkeypatch):
    from src import stuff_rank
    stuff_rank._R.clear()
    stuff_rank._R.update({"R": 50.0})
    g = pd.DataFrame({"pitcher_id": ["1", "2", "3"], "season": [2025] * 3, "game_pk": [1, 2, 3],
                      "n": [400, 400, 400], "s": [-400.0, 0.0, 400.0]})
    w = pd.Series({"1": 1.0, "2": 1.0, "3": 0.0})              # bootstrap drew none of pitcher 3's games
    c = stuff_rank.candidate_stuff(g, w, seasons=[2026, 2025, 2024]).set_index("pitcher_id").stuff_plus
    assert np.isclose(c.loc[3], 100.0)
    stuff_rank._R.clear()


# ------------------------------------------------------------------ amendment 19
def test_metric_ranking_bb_low_first_k_high_first():
    from src import metric_reports
    C = pd.DataFrame({"name": ["a", "b", "c"], "sign_tier": ["T1_free"] * 3, "marcel_k_pct": [0.20, 0.25, 0.22],
                      "marcel_bb_pct": [0.06, 0.10, 0.08], "marcel_score": [0.14, 0.15, 0.14], "combined_rank": [3, 1, 2]})
    assert metric_reports._ranking(C, "bb").球員.tolist() == ["a", "c", "b"]
    assert metric_reports._ranking(C, "k").球員.tolist() == ["b", "c", "a"]
    assert metric_reports._ranking(C, "k")["主排名（可簽者）"].tolist() == [1, 2, 3]


# ------------------------------------------------------------------ amendment 20
def _teams(rng, vals):
    """Synthetic team-seasons whose runs are exact linear weights of their events."""
    rows = []
    for s in range(2015, 2025):
        for t in range(5):
            p = {"1B": rng.uniform(.14, .18), "2B": rng.uniform(.03, .05), "3B": rng.uniform(.002, .006),
                 "HR": rng.uniform(.015, .035), "BB": rng.uniform(.07, .10), "HBP": rng.uniform(.008, .015)}
            pa = 5000
            r = {"season": s, "PA": pa, **{f"p_{e}": v for e, v in p.items()}}
            r["p_OUT"] = 1 - sum(p.values())
            r["rpa"] = sum(vals[e] * p[e] for e in p)
            rows.append(r)
    return pd.DataFrame(rows)


def test_importance_recovers_linear_weights():
    from src import kbb_weights
    vals = {"1B": .45, "2B": .75, "3B": 1.0, "HR": 1.4, "BB": .30, "HBP": .32}   # relative to an out (0)
    t = _teams(np.random.default_rng(0), vals)
    r = kbb_weights.importance(t)
    avg = float(np.average(sum(t[f"p_{e}"] * v for e, v in vals.items()), weights=t.PA))
    assert np.isclose(r["w_K"], avg, atol=1e-6)                      # a K = an out: minus the average PA
    assert np.isclose(r["w_BB"], vals["BB"] - avg, atol=1e-6)


def test_baseruns_calibrates_to_league_runs_and_k_equals_out():
    from src import kbb_weights
    t = pd.DataFrame({"R": [700.0], "H": [1400.0], "2B": [250.0], "3B": [25.0], "HR": [120.0], "BB": [500.0], "HBP": [60.0],
                      "AB": [5500.0], "SH": [40.0], "SF": [40.0]})
    t["1B"] = t.H - t["2B"] - t["3B"] - t.HR
    t["OUT"] = t.AB - t.H + t.SH + t.SF
    t["PA"] = t.AB + t.BB + t.HBP + t.SH + t.SF
    r = kbb_weights.baseruns(t)
    assert r["w_K"] > 0 and r["w_BB"] > 0 and 0.3 < r["B_multiplier"] < 3
