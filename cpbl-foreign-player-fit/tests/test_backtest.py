"""Backtest cohort rules (amendment 2) on synthetic data."""
import numpy as np
import pandas as pd

from src import backtest
from src.common import ROOT, config


def _pitches(pid, dates, hand="R", season=None, n=1):
    rows = []
    for d in dates:
        for k in range(n):
            rows.append({"pitcher_id": pid, "date": pd.Timestamp(d), "season": season or pd.Timestamp(d).year,
                         "p_throws": hand, "is_bunt": False, "swing": k % 2 == 0, "whiff": k % 4 == 0,
                         "pa_end": k % 4 == 3, "pa_event": "SO" if k % 8 == 3 else ("uBB" if k % 8 == 7 else None),
                         "woba_value": 0.0, "woba_denom": 1.0})
    return pd.DataFrame(rows)


def _ref(**kw):
    base = {"rebas_id": "abc", "rebas_name": "測試", "english_name": "Test Guy", "mlbam_id": 1}
    base.update(kw)
    return pd.DataFrame([base]).astype({"mlbam_id": "Int64"})


def test_rates_use_the_same_definitions_on_both_sides():
    r = backtest._rates(_pitches("1", ["2024-05-01"], n=16))
    assert r["pa"] == 4 and r["swings"] == 8
    assert r["k_minus_bb_pct"] == 0.0            # 2 SO, 2 uBB
    assert r["whiff_pct"] == 0.5


def test_screen_counts_only_aaa_pitches_before_the_first_cpbl_game():
    cpbl = _pitches("abc", ["2025-04-10", "2025-06-01"], n=40)
    aaa = pd.concat([_pitches("1", ["2024-04-11"], n=300),          # 364 days before -> inside
                     _pitches("1", ["2024-04-09"], n=300),          # 366 days before -> outside
                     _pitches("1", ["2025-04-10"], n=300)])         # on the debut day -> outside
    sc = backtest.screen(_ref(), cpbl, aaa)
    row = sc.iloc[0]
    assert row.aaa_pitches == 300
    assert row.excluded.startswith("AAA pitches in window")


def test_screen_rejects_hand_mismatch_and_counts_a_pitcher_once():
    cpbl = pd.concat([_pitches("abc", ["2024-08-01"], n=400, season=2024),
                      _pitches("abc", ["2025-04-01"], n=400, season=2025)])
    aaa = _pitches("1", ["2024-05-01"], n=500)
    sc = backtest.screen(_ref(), cpbl, aaa)
    assert sc.excluded.tolist() == ["", "counted in an earlier season"]
    sc = backtest.screen(_ref(), cpbl, aaa.assign(p_throws="L"))
    assert set(sc.excluded) == {"hand mismatch"}


def test_screen_requires_cpbl_workload_and_an_id():
    cpbl = _pitches("abc", ["2025-04-01"], n=200)                  # 50 PA < 60
    aaa = _pitches("1", ["2024-05-01"], n=500)
    assert backtest.screen(_ref(), cpbl, aaa).excluded.iat[0].startswith("CPBL BF")
    assert backtest.screen(_ref(mlbam_id=pd.NA), cpbl, aaa).excluded.iat[0] == "no MLBAM id"


def test_verdict_follows_the_registered_rule():
    n = config()["backtest"]["min_n_for_conclusion"]
    assert backtest.verdict(n - 1, 0.2, 0.6, 1).startswith("descriptive")
    assert backtest.verdict(n, 0.1, 0.6, 1) == "supports"
    assert backtest.verdict(n, -0.6, -0.1, 1) == "contradicts"
    assert backtest.verdict(n, -0.6, -0.1, -1) == "supports"
    assert backtest.verdict(n, -0.2, 0.4, 1) == "inconclusive"


def test_reference_table_ids_are_unique_and_sourced():
    ref = backtest.reference()
    assert ref.rebas_id.is_unique
    ids = ref.mlbam_id.dropna()
    assert ids.is_unique
    assert ref.source.notna().all()
    assert (ROOT / config()["backtest"]["reference"]).exists()


def test_spearman_bootstrap_is_reproducible():
    df = pd.DataFrame({"x": np.arange(20.0), "y": np.arange(20.0) ** 2})
    a = backtest.boot_ci(df, "x", "y", 200, 0.9, np.random.default_rng(1))
    b = backtest.boot_ci(df, "x", "y", 200, 0.9, np.random.default_rng(1))
    assert a == b and a[0] == 1.0
