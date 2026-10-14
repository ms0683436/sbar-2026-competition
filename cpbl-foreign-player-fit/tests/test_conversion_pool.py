"""Amendment 9: role effect, starter-equivalent counts, scoring against the starter pool."""
import numpy as np
import pandas as pd

from src import conversion_pool as cp


def _app(rows):
    return pd.DataFrame(rows, columns=["season", "pitcher_id", "game_id", "role", "pitches", "bf", "SO", "BB", "HR"])


def test_role_effect_is_within_pitcher_relief_minus_start():
    rows = []
    for p, (k_sp, k_rp) in enumerate([(0.20, 0.25), (0.10, 0.15)]):     # +5 points each, different levels
        rows.append([2025, str(p), 1, "SP", 1500, 100, 100 * k_sp, 8, 3])
        rows.append([2025, str(p), 2, "RP", 600, 40, 40 * k_rp, 4, 1])
    eff = cp.role_effect(_app(rows), np.random.default_rng(0))
    assert eff["n_pitcher_seasons"] == 2
    assert np.isclose(eff["SO"]["effect"], 0.05)
    assert np.isclose(eff["BB"]["effect"], 4 / 40 - 8 / 100)


def test_adjust_counts_scales_by_relief_share_and_never_goes_negative():
    df = pd.DataFrame({"relief_share": [1.0, 0.5, 0.0], "pitching_BF": [100, 100, 100],
                       "pitching_SO": [30, 30, 30], "pitching_BB": [10, 10, 10], "pitching_HR": [0, 2, 2]})
    eff = {"SO": {"effect": 0.02}, "BB": {"effect": 0.01}, "HR": {"effect": 0.05}}
    out = cp.adjust_counts(df, eff)
    assert list(out.pitching_SO) == [28, 29, 30]
    assert list(out.pitching_HR) == [0, 0, 2]


def test_starting_evidence_thresholds():
    s = cp.starting_evidence(pd.Series([90, 75, 74, 50, 49]))
    assert list(s) == ["stretched", "stretched", "partial", "partial", "none"]


def test_score_vs_starter_reference_matches_zscore_on_the_reference():
    from src import baseline
    rng = np.random.default_rng(1)
    d = pd.DataFrame({"k_pct_adj": rng.normal(0, .05, 50), "bb_pct_adj": rng.normal(0, .02, 50),
                      "fip": rng.normal(0, .8, 50)})
    own = baseline.score(d, "pitcher", baseline.judgment_weights("pitcher")).sort_index().baseline_score
    assert np.allclose(cp.score_vs(d, d), own)


def test_cohort_role_effect_uses_only_the_pre_arrival_season():
    import inspect
    src = inspect.getsource(cp.cohort_check)
    assert "_pitches([int(y)])" in src            # role effect and relief share: season Y only
