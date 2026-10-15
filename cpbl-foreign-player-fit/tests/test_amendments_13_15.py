"""Amendments 13-15: Stuff+ blend, two-strike battle flags, AAA Marcel helpers (synthetic data, no downloads)."""
import numpy as np
import pandas as pd

from src import battle, marcel, stuff_rank
from src.common import zscore


# ------------------------------------------------------------------ amendment 13
def test_combined_weights_and_missing_stuff_is_average():
    base = pd.Series([1.0, 2.0, 3.0, 4.0])
    stuff = pd.Series([110.0, np.nan, 90.0, 100.0])
    fit = pd.Series([0.5, -0.5, 0.0, 0.0])
    got = stuff_rank.combined(base, stuff, fit)
    zs = zscore(stuff, 2.5).fillna(0.0)
    want = 0.5 * zscore(base) + 0.2 * zs + 0.3 * fit
    assert np.allclose(got, want)
    assert zs.iat[1] == 0.0                                    # < 300 pitches: no Stuff+ = league average


def test_stuff_sensitivity_takes_weight_from_baseline():
    base, stuff, fit = pd.Series([1.0, 2.0, 3.0]), pd.Series([100.0, 105.0, 95.0]), pd.Series([0.0, 0.0, 0.0])
    got = stuff_rank.combined(base, stuff, fit, w_stuff=0.3)
    want = 0.4 * zscore(base) + 0.3 * zscore(stuff, 2.5)
    assert np.allclose(got, want)


def test_candidate_stuff_scale_and_minimum():
    # season 2025: three pitchers; lower predicted run value = better = higher Stuff+
    g = pd.DataFrame({"pitcher_id": ["1", "2", "3", "1"], "season": [2025, 2025, 2025, 2025],
                      "game_pk": [10, 11, 12, 13], "n": [400, 500, 100, 200], "s": [-4.0, 0.0, 1.0, -2.0]})
    c = stuff_rank.candidate_stuff_a13(g).set_index("pitcher_id")      # amendment 13's rule (18 replaces it)
    assert c.stuff_plus.loc[1] > 100 > c.stuff_plus.loc[2]
    assert np.isnan(c.stuff_plus.loc[3])                       # 100 pitches < 300 minimum
    assert c.stuff_pitches.loc[1] == 600


# ------------------------------------------------------------------ amendment 14
def _pa(pa_id, seq):
    """seq: list of (balls, strikes, AAA call code, pa_end, event)."""
    rows = []
    for i, (b, k, code, end, ev) in enumerate(seq, 1):
        rows.append({"pa_id": pa_id, "pitch_seq": i, "balls": b, "strikes": k, "pitch_code": code,
                     "swing": code in {"S", "W", "T", "F", "X", "Q", "R"}, "whiff": code in {"S", "W", "Q"},
                     "called_strike": code == "C", "pa_end": end, "pa_event": ev if end else None,
                     "is_bunt": code in {"L", "M", "O"}, "pitch_type_raw": "FF"})
    return rows


def test_flags_foul_tip_strikeout_is_not_a_continuation_foul():
    d = pd.DataFrame(_pa("p1", [(0, 0, "C", False, None), (0, 1, "T", False, None),     # one-strike foul tip
                                (0, 2, "F", False, None),                               # two-strike foul: continuation
                                (0, 2, "T", True, "SO")]))                             # foul-tip strikeout: terminal
    f = battle._flags(d)
    assert f.n2.sum() == 2 and f.sw2.sum() == 2
    assert f.foul2.sum() == 1 and f.contact2.sum() == 2 and f.so2.sum() == 1


def test_audit_keeps_clean_pas_and_drops_the_rest():
    ok = _pa("a", [(0, 0, "S", False, None), (0, 1, "F", False, None), (0, 2, "F", False, None),
                   (0, 2, "S", True, "SO")])
    early = _pa("b", [(0, 2, "S", False, None), (0, 2, "B", True, "uBB")])             # not 0-0 at the start
    jump = _pa("c", [(0, 0, "B", False, None), (2, 0, "B", False, None),               # 1-0 expected
                     (3, 0, "B", True, "uBB")])
    bunt = _pa("d", [(0, 0, "L", False, None), (0, 1, "X", True, "OUT")])
    pitchout = _pa("e", [(0, 0, "P", False, None), (1, 0, "X", True, "OUT")])
    tip2 = _pa("f", [(0, 0, "C", False, None), (0, 1, "C", False, None), (0, 2, "T", False, None),
                     (0, 2, "X", True, "OUT")])                                        # non-terminal two-strike tip
    walk = _pa("g", [(0, 0, "B", False, None), (1, 0, "B", False, None), (2, 0, "*B", False, None),
                     (3, 0, "B", True, "uBB")])
    k3 = _pa("h", [(0, 0, "C", False, None), (0, 1, "C", False, None), (0, 2, "S", False, None),
                   (0, 2, "X", True, "OUT")])                                          # strike three not terminal
    so_ball = _pa("i", [(0, 0, "C", False, None), (0, 1, "C", False, None), (0, 2, "B", True, "SO")])
    hbp = _pa("j", [(0, 0, "F", False, None), (0, 1, "H", True, "HBP")])
    a = battle._audit(pd.DataFrame(ok + early + jump + bunt + pitchout + tip2 + walk + k3 + so_ball + hbp))
    assert a.keep.to_dict() == {"a": True, "b": False, "c": False, "d": False, "e": False, "f": False, "g": True,
                                "h": False, "i": False, "j": True}


def test_partial_spearman_removes_the_control():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    e = pd.DataFrame({"aaa_kbb": x, "putaway": x + rng.normal(size=200), "cpbl_kbb_c": x + rng.normal(size=200)})
    assert abs(battle._partial(e)) < 0.2                      # linked only through the control


# ------------------------------------------------------------------ amendment 15
def test_age_factor_shape():
    f = marcel.age_factor(pd.Series([25.0, 29.0, 33.0, np.nan]))
    assert np.isclose(f.iat[0], 1 + 0.006 * 4) and f.iat[1] == 1.0
    assert np.isclose(f.iat[2], 1 / (1 + 0.003 * 4)) and f.iat[3] == 1.0


def test_log_scale_identity_and_ratio():
    sl = pd.DataFrame({"player_id": [1, 1], "team_id": [5, 5], "game_id": ["g1", "g2"], "game_date": ["2025-05-01"] * 2,
                       "pitching_BF": [3, 3], "pitching_IP": [1 / 3, 1.0], "outs": [1, 3]})     # statsapi: true thirds
    assert np.isclose(marcel.log_scale(sl, pd.Series({"g1": 2.0, "g2": 0.0})).pitching_IP.iat[0], 0.5)
    logs = pd.DataFrame({"player_id": [1, 1, 1], "team_id": [5, 5, 5], "game_id": ["g1", "g2", "g3"],
                         "game_date": ["2026-05-01", "2026-05-02", "2026-05-03"],
                         "pitching_BF": [20, 10, 30], "pitching_SO": [5, 0, 5], "pitching_IP": [5.0, 2.1, 6.0]})
    one = marcel.log_scale(logs, pd.Series(1.0, index=["g1", "g2", "g3"]))
    assert np.allclose(one.to_numpy(), 1.0)
    sc = marcel.log_scale(logs, pd.Series({"g1": 2.0, "g2": 0.0, "g3": 1.0}))
    r = sc.loc[(1, 2026, 5)]
    assert np.isclose(r.pitching_BF, 70 / 60) and np.isclose(r.pitching_SO, 15 / 10)
    assert np.isclose(r.pitching_IP, (2 * 15 + 18) / (15 + 7 + 18))   # outs, not x.1 / x.2 notation


def test_scaled_only_touches_matching_stints():
    s = pd.DataFrame({"player_id": [1, 2], "season": [2026, 2026], "team_id": [5, 5],
                      "pitching_BF": [100.0, 100.0], "pitching_SO": [20.0, 20.0]})
    sc = pd.DataFrame({"pitching_BF": [2.0], "pitching_SO": [0.5]},
                      index=pd.MultiIndex.from_tuples([(1, 2026, 5)], names=["player_id", "season", "team_id"]))
    out = marcel._scaled(s, sc)
    assert out.pitching_BF.tolist() == [200.0, 100.0] and out.pitching_SO.tolist() == [10.0, 20.0]


def test_standardisation_uses_full_target_on_common_support():
    S = lambda **kw: pd.Series(kw, dtype=float)
    blank = {"pitches": 1.0, "pas": 1.0, "pa2_all": 1.0, **{f"len_{lo}": 0.0 for lo in (1, 4, 6, 8)}}
    q = {k: S(A=1, B=1) for k in ("n2", "sw2", "foul2", "contact2", "so2", "pa2", "so_pa", "n2_pa")}
    cpbl = {**blank, **q, "n2": S(A=100, B=0), "foul2": S(A=30, B=0)}
    aaa = {**blank, **q, "n2": S(A=100, B=300), "foul2": S(A=20, B=30)}
    r = battle._season_stats(cpbl, aaa)["A1 continuation-foul rate"]
    # target: cpbl (1, 0), aaa (.25, .75) -> (.625, .375); only A is common
    assert np.isclose(r["excluded_weight"], 0.375)
    assert np.isclose(r["cpbl_std"], 0.30) and np.isclose(r["aaa_std"], 0.20)
