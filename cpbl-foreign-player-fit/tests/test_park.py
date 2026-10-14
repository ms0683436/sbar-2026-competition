"""Amendment 5: AAA park factors and the half-home adjustment (synthetic data only)."""
import numpy as np
import pandas as pd
import pytest

from src import park


def _games(rng, n_clubs=10, games=300, hr_boost=None):
    rows = []
    for k in range(games * n_clubs // 2):
        h, a = rng.choice(n_clubs, 2, replace=False)
        pa = 80
        rate = {"SO": 0.22, "uBB": 0.09, "HBP": 0.01, "1B": 0.15, "2B": 0.05, "3B": 0.005, "HR": 0.03}
        if hr_boost and h in hr_boost:
            rate = dict(rate, HR=rate["HR"] * hr_boost[h])
        r = {e: rng.poisson(pa * v) for e, v in rate.items()}
        rows.append(dict(r, PA=pa, home_id=h, away_id=a))
    return pd.DataFrame(rows)


def test_estimate_recovers_a_home_run_park_and_averages_to_one():
    g = _games(np.random.default_rng(0), hr_boost={0: 1.5})
    pf = park.estimate(g)
    assert pf.HR.idxmax() == 0 and pf.loc[0, "HR"] > 1.2      # regressed and renormalised
    assert pf.drop(index=0).HR.between(0.85, 1.1).all()
    home_pa = g.groupby("home_id").PA.sum()
    assert np.average(pf.HR, weights=home_pa.reindex(pf.index)) == pytest.approx(1.0)
    assert pf["r_HBP"].iat[0] < 0.5          # pure noise is regressed away


def _patch(monkeypatch, hr=1.5, single=1.2):
    pf = pd.DataFrame({e: [1.0] for e in park.EVENTS}, index=pd.Index([7], name="club_id"))
    pf.loc[7, "HR"], pf.loc[7, "1B"] = hr, single
    monkeypatch.setattr(park, "factors", lambda: pf)
    monkeypatch.setattr(park, "enabled", lambda: True)


def test_season_stint_uses_half_the_home_factor(monkeypatch):
    _patch(monkeypatch)
    d = pd.DataFrame({"team_id": [7, 8], "pitching_HR": [25.0, 25.0], "pitching_H": [100.0, 100.0],
                      "pitching_SO": [80.0, 80.0]})
    out = park.adjust(d, "pitching")
    assert out.pitching_HR.tolist() == pytest.approx([25 / 1.25, 25.0])
    assert out.pitching_H.iat[0] == pytest.approx(75 / 1.1 + 25 / 1.25)
    assert out.pitching_SO.tolist() == [80.0, 80.0]


def test_hitter_hits_and_total_bases_are_rebuilt(monkeypatch):
    _patch(monkeypatch)
    d = pd.DataFrame({"team_id": [7], "batting_H": [100.0], "batting_2B": [20.0], "batting_3B": [2.0],
                      "batting_HR": [15.0], "batting_TB": [100 + 20 + 4 + 45.0], "batting_BB": [40.0]})
    out = park.adjust(d, "batting")
    one = 63 / 1.1
    hr = 15 / 1.25
    assert out.batting_HR.iat[0] == pytest.approx(hr)
    assert out.batting_H.iat[0] == pytest.approx(one + 20 + 2 + hr)
    assert out.batting_TB.iat[0] == pytest.approx(one + 40 + 6 + 4 * hr)


def test_game_logs_adjust_home_games_only(monkeypatch):
    _patch(monkeypatch)
    d = pd.DataFrame({"team_id": [7, 7], "pitching_HR": [3.0, 3.0], "pitching_H": [8.0, 8.0]})
    out = park.adjust(d, "pitching", home=pd.Series([True, False]))
    assert out.pitching_HR.tolist() == pytest.approx([2.0, 3.0])


def test_disabled_is_a_no_op(monkeypatch):
    monkeypatch.setattr(park, "enabled", lambda: False)
    d = pd.DataFrame({"team_id": [7], "pitching_HR": [3.0]})
    assert park.adjust(d, "pitching") is d
