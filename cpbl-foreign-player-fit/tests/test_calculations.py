"""Metric definitions and Fit-index properties (synthetic fixtures only)."""
import numpy as np
import pandas as pd
import pytest

from src import fit_score
from src.common import config, shrink_rate, wilson
from src.milb_loader import classify_des
from src.pitch_schema import harmonise_pitch_type, woba_columns


def test_wilson_contains_point_estimate_and_handles_zero_n():
    lo, hi = wilson(np.array([20, 0]), np.array([100, 0]))
    assert lo[0] < 0.2 < hi[0]
    assert np.isnan(lo[1]) and np.isnan(hi[1])


def test_shrinkage_moves_small_samples_more():
    small = shrink_rate(3, 5, 0.25, 60)        # raw 0.60
    big = shrink_rate(300, 500, 0.25, 60)      # raw 0.60
    assert abs(small - 0.25) < abs(big - 0.25)


def test_woba_denominator_excludes_ibb_sh_ci():
    ev = pd.Series(["1B", "uBB", "IBB", "SH", "CI", "SO", "HR", "SF"])
    val, den = woba_columns(ev)
    assert list(den) == [1, 1, 0, 0, 0, 1, 1, 1]
    w = config()["woba_weights"]
    assert val.iloc[0] == w["1B"] and val.iloc[6] == w["HR"] and val.iloc[2] == 0


def test_pitch_type_harmonisation_and_yaml_strings():
    common, group = harmonise_pitch_type(pd.Series(["ST", "SV", "FO", "FS", "KC", "FT", "FA", "KN", None]))
    assert list(common[:7]) == ["SL", "SL", "SPL", "SPL", "CU", "SI", "FF"]
    assert pd.isna(common.iloc[7]) and pd.isna(common.iloc[8])
    # YAML 1.1 would parse a bare OFF as boolean False — the group must be the string
    assert group.iloc[2] == "OFF" and isinstance(group.iloc[2], str)


@pytest.mark.parametrize("des,event", [
    ("T.J. Rumfield doubles (2) on a sharp line drive to left fielder.", "2B"),
    ("Jorbit Vivas walks. Ismael Munguia to 2nd.", "uBB"),
    ("Joe Smith intentionally walks Bob Jones.", "IBB"),
    ("Mets challenged (pitch result), call on the field was overturned: Bo Bichette strikes out swinging.", "SO"),
    ("Tommy Sacco Jr. called out on strikes.", "SO"),
    ("Kyle Lewis hits a grand slam (3) to left field.", "HR"),
    ("Pat Doe out on a sacrifice fly to right fielder Al Roe.", "SF"),
    ("Pat Doe grounds into a double play, shortstop to first.", "OUT"),
    ("Pat Doe reaches on a fielding error by shortstop Al Roe.", "OUT"),
    ("Yonathan Perlaza steals (1) 3rd base.", "OTHER"),
    (np.nan, "OTHER"),
])
def test_aaa_pa_outcome_parser(des, event):
    assert classify_des(des) == event


def _env(cpbl_whiff, aaa_whiff, cpbl_woba=None, aaa_woba=None, swings=1000):
    """Synthetic build_tables() output with one game per league."""
    rows_c, rows_a = [], []
    for i, c in enumerate(fit_score.CELLS):
        rows_c.append({"season": 2025, "game_id": "c1", "cell": c, "pitches": 2 * swings, "swings": swings,
                       "whiffs": cpbl_whiff[i] * swings, "pa": 500,
                       "woba_num": 500 * (cpbl_woba[i] if cpbl_woba else 0.3)})
        rows_a.append({"season": 2025, "game_id": "a1", "cell": c, "pitches": 2 * swings, "swings": swings,
                       "whiffs": aaa_whiff[i] * swings, "pa": 500,
                       "woba_num": 500 * (aaa_woba[i] if aaa_woba else 0.33)})
    hand = pd.DataFrame([{"season": 2025, "game_id": g, "stand": s, "p_throws": p, "pa": 100, "woba_num": 33}
                         for g in ("c1", "a1") for s in "LR" for p in "LR"])
    return {"env_cpbl": pd.DataFrame(rows_c), "env_aaa": pd.DataFrame(rows_a),
            "hand_cpbl": hand[hand.game_id == "c1"], "hand_aaa": hand[hand.game_id == "a1"]}


def test_uniform_level_gap_gives_zero_contrast():
    aaa = np.linspace(0.15, 0.35, 9)
    lg = np.log(aaa / (1 - aaa)) - 0.3   # CPBL uniformly 0.3 logit lower on every cell
    cpbl = 1 / (1 + np.exp(-lg))
    t = _env(cpbl, aaa, swings=200_000)  # large n so the small-cell shrinkage is negligible
    e = fit_score.environment(t)
    # shrinkage toward the CPBL league rate perturbs slightly; contrasts must be ~0
    assert np.allclose(e.whiff_contrast, 0, atol=0.03)


def test_aaa_average_pitcher_scores_zero_fit():
    aaa = np.linspace(0.15, 0.35, 9)
    cpbl = np.array(aaa) * np.array([1.2, 1.1, 1.0, 0.9, 0.8, 0.9, 1.0, 1.1, 1.2])
    t = _env(cpbl, aaa)
    e = fit_score.environment(t)
    # a pitcher whose cell usage equals AAA swing shares and whose whiff% equals AAA rates
    pit = t["env_aaa"].rename(columns={}).assign(pitcher_id="p1")
    t["pit"] = pit
    f = fit_score.pitcher_fit(t, e)
    assert abs(f.loc["p1", "fit_mix_whiff"]) < 1e-6
    assert abs(f.loc["p1", "fit_mix_damage"]) < 1e-6


def test_hitter_fit_zero_when_no_distribution_shift():
    aaa = np.linspace(0.15, 0.35, 9)
    t = _env(aaa, aaa)
    e = fit_score.environment(t)
    hit = t["env_aaa"].assign(batter_id="b1")
    hit["woba_num"] = hit.pa * np.linspace(0.25, 0.45, 9)   # strong shape
    t["hit"] = hit
    t["hand_hit"] = t["hand_aaa"].assign(batter_id="b1")
    f = fit_score.hitter_fit(t, e)
    assert abs(f.loc["b1", "fit_type"]) < 1e-9 and abs(f.loc["b1", "fit_velo"]) < 1e-9


def test_recruitment_facts_never_enter_baseline_score():
    from src import baseline
    rng = np.random.default_rng(0)
    spec = baseline.config()["baseline"]["pitcher"]
    d = pd.DataFrame({m: rng.normal(size=30) for m in spec})
    d["name"] = [f"p{i}" for i in range(30)]
    a = baseline.score(d.assign(on_40man=True), "pitcher").sort_values("name").baseline_score.values
    b = baseline.score(d.assign(on_40man=rng.random(30) > 0.5), "pitcher").sort_values("name").baseline_score.values
    assert np.allclose(a, b)


def test_roster_tier_wording():
    from src import roster_status
    on = pd.Series({"on_40man": True, "roster_status": "Injured 60-Day", "roster_club": "San Francisco Giants"})
    off = pd.Series({"on_40man": False, "roster_status": np.nan, "roster_club": np.nan})
    assert roster_status.tier(on) == "on MLB 40-man roster (Injured 60-Day) — San Francisco Giants"
    assert roster_status.tier(off) == "not on any MLB 40-man roster"


def test_roster_mlb_line_does_not_double_count_traded_players():
    from src import roster_status
    stint = lambda pa, ip, g, team=True: {"sport": {"id": 1}, **({"team": {"id": 1}} if team else {}),
                                          "stat": {"plateAppearances": pa, "inningsPitched": ip, "gamesPlayed": g}}
    traded = [stint(40, "10.1", 12), stint(60, "5.2", 20), stint(100, "16.0", 32, team=False)]
    assert roster_status.mlb_line(traded)["plateAppearances"] == 100          # combined row, not 200
    assert roster_status.mlb_line(traded)["inningsPitched"] == "16.0"
    no_total = roster_status.mlb_line(traded[:2])                              # stints only: add them up
    assert no_total["plateAppearances"] == 100 and no_total["gamesPlayed"] == 32
    assert no_total["inningsPitched"] == "16.0"                                # 31 + 17 outs
    assert roster_status.mlb_line([{"sport": {"id": 11}, "stat": {}}]) is None
