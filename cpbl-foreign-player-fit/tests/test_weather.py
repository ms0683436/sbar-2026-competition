"""Amendment 10: centring and the hot / other split."""
import numpy as np
import pandas as pd

from src import weather


def test_centre_removes_club_and_month_means():
    d = pd.DataFrame({"season": 2025, "home_club": [1, 1, 2, 2], "month": [5, 7, 5, 7], "v": [1.0, 3.0, 2.0, 4.0]})
    c = weather._centre(d, "v")
    assert np.allclose(c, 0)            # purely additive club + month pattern leaves nothing


def test_split_needs_minimum_in_both_groups():
    rows = []
    for i in range(60):                 # 40 hot BF and 20 other BF: below the 40-BF minimum
        rows.append({"season": 2025, "pitcher_id": "1", "temp_f": 90 if i < 40 else 70, "home_club": 1, "month": 6,
                     "pa_end": True, "pa_event": "SO" if i % 3 == 0 else "OUT", "swing": False, "is_bunt": False,
                     "whiff": False, "pitch_type_raw": "SL", "velo_mph": 85.0})
    sp = weather.splits(pd.DataFrame(rows), 85)
    assert sp.empty or sp.k_pct.isna().all()


def test_reliability_handles_one_season_and_constant_splits():
    sp = pd.DataFrame({"season": [2025] * 4, "pitcher_id": list("abcd"), "k_pct": [.1, .2, .3, .4],
                       "whiff_pct": [.0] * 4, "velo": [1.0] * 4})
    out = weather.reliability(sp, np.random.default_rng(0))
    assert all(r["verdict"] == "too few" for r in out)
