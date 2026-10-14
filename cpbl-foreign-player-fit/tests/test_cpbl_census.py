"""Amendment 12 census helpers and the corrected wOBA weights (synthetic data, no downloads)."""
import numpy as np
import pandas as pd

from src import cpbl_census as cc
from src.common import config


def test_country_reads_the_trailing_register_code():
    assert cc._country("December 16, 1987 in Davie, FL us") == "US"
    assert cc._country("September 20, 1997 in Taiwan tw") == "TW"            # no comma before the code
    assert cc._country("May 1, 1990 in Santo Domingo, Distrito Nacional do") == "DO"
    assert cc._country("January 18, 1988") == ""                             # no birthplace: unknown


def test_category_keeps_mexican_league_out_of_aaa():
    assert cc._category("IL", "AAA") == "AAA" and cc._category("PCL", "AAA") == "AAA"
    assert cc._category("MEX", "AAA") == "other"                            # MLB-affiliated AAA only
    assert cc._category("AL", "Maj") == "MLB"
    assert cc._category("KBO", "Fgn") == "KBO" and cc._category("CL", "Fgn") == "NPB"
    assert cc._category("SL", "AA") == "AA or lower"
    assert cc._category("ATLL", "Ind") == "other"


def test_history_drops_multi_league_summary_rows():
    pitching = pd.DataFrame({"year_ID": ["2024", "2024", "2024", "2025"],
                             "lg_ID": ["2 Lgs", "IL", "MEX", "CPBL"],
                             "level": ["AAA-Ind", "AAA", "Ind", "Fgn"],
                             "batters_faced": ["300", "200", "100", "400"]})
    h = cc.history({"pitching": pitching, "batting": pd.DataFrame()})
    got = {(r.season, r.cat): r.batters_faced for r in h.itertuples()}
    assert got == {(2024, "AAA"): 200, (2024, "other"): 100, (2025, "CPBL"): 400}


def test_rolling_uses_earlier_arrivals_only():
    d = pd.DataFrame({"season": [2019] * 4 + [2020] * 2,
                      "x": [0.0, 1.0, 2.0, 3.0, 1.0, 2.0],
                      "y": [0.0, 0.1, 0.2, 0.3, 9.0, 9.0]})
    r = cc.rolling(d, "x", "y", np.random.default_rng(0))
    assert r["n_held_out"] == 2
    assert np.isclose(r["mae_m1"], np.mean([9.0 - 0.1, 9.0 - 0.2]))        # the 9.0 outcomes never leak in


def test_woba_weights_are_fangraphs_2024():
    w = config()["woba_weights"]
    want = {"uBB": 0.689131, "HBP": 0.720192, "1B": 0.881711, "2B": 1.25445, "3B": 1.58991, "HR": 2.04961}
    assert {k: w[k] for k in want} == want


def test_census_woba_uses_config_weights_and_drops_ibb():
    d = pd.DataFrame({"AB": [4], "H": [2], "2B": [1], "3B": [0], "HR": [1], "BB": [2], "IBB": [1],
                      "HBP": [0], "SF": [0], "PA": [6]})
    w = config()["woba_weights"]
    num = w["uBB"] * 1 + w["2B"] + w["HR"]
    assert np.isclose(cc._woba(d).iat[0], num / (4 + 1 + 0 + 0))           # AB + uBB + SF + HBP
