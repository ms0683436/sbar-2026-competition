"""Amendment 6: signability tiers, two-season pool and the recruitment backtest arms (synthetic data)."""
import numpy as np
import pandas as pd

from src import milb_candidates, recruitment_backtest, signability

AFF = {100, 200}          # affiliated clubs; 900 = a winter / Mexican league club


def _tx(rows, draft=None):
    return pd.DataFrame([{"mlbam_id": r[0], "date": r[1], "tx_id": i, "type": r[2], "to_team": r[3],
                          "from_team": None, "description": "", "draft_year": draft} for i, r in enumerate(rows)])


def _tier(rows, as_of="2026-10-13", on40=(), draft=None):
    return signability.classify(_tx(rows, draft), as_of, set(on40), AFF).set_index("mlbam_id").sign_tier


def test_free_agent_after_election_and_winter_league_ignored():
    rows = [(1, "2025-05-01", "SFA", 100), (1, "2025-11-06", "DFA", 100), (1, "2025-11-20", "ASG", 900),
            (1, "2025-12-01", "SFA", 900)]          # signs in Mexico: not an affiliated club
    assert _tier(rows)[1] == "T1_free"


def test_resigning_restores_control_and_veteran_minor_league_deal_is_expiring():
    rows = [(2, "2024-11-06", "DFA", 100), (2, "2025-01-10", "SFA", 200)]
    assert _tier(rows)[2] == "T2_expiring"


def test_international_first_contract_is_not_a_free_agent_signing():
    rows = [(3, "2022-02-03", "SFA", 100)]           # amateur signing, filed as SFA by statsapi
    assert _tier(rows)[3] == "T3_controlled"
    assert _tier(rows, as_of="2028-10-13")[3] == "T2_expiring"     # six seasons later


def test_draftee_is_controlled_until_six_seasons_and_40man_wins():
    rows = [(4, "2024-07-24", "SGN", 100)]
    assert _tier(rows, draft=2024)[4] == "T3_controlled"
    assert _tier(rows, draft=2024, on40={4})[4] == "T4_40man"


def test_events_after_as_of_are_ignored():
    rows = [(5, "2024-11-06", "DFA", 100), (5, "2026-11-06", "SFA", 100)]
    assert _tier(rows)[5] == "T1_free"


def _stints():
    rows = []
    for season, gs, ip in ((2025, 8, 50.0), (2026, 4, 25.0)):       # 12 GS / 75 IP over two seasons
        rows.append({"season": season, "player_id": 9, "player_full_name": "A", "player_position": "P",
                     "team_id": 100, "team_abv": "X", "team_league": f"{season}_INT", "pitching_GS": gs,
                     "pitching_IP": ip, "outs": ip * 3, "pitching_BF": ip * 4.3, "pitching_SO": ip, "pitching_BB": ip / 3,
                     "pitching_HBP": 1, "pitching_HR": 2, "pitching_ER": ip / 2, "pitching_H": ip})
    return pd.DataFrame(rows)


def test_two_season_pool_keeps_a_player_the_one_season_rule_drops():
    s = _stints()
    df = milb_candidates._combine(s, ["pitching_GS", "pitching_IP", "outs"], "pitching_BF")
    df = milb_candidates._latest(s, df, ["pitching_GS", "pitching_IP"])
    assert df.pitching_GS.iat[0] == 12 and df.latest_pitching_GS.iat[0] == 4
    assert {"share_2025_INT", "share_2026_INT"} <= set(df.columns)
    assert np.isclose(df.share_2025_INT.iat[0] + df.share_2026_INT.iat[0], 1.0)


def test_backtest_arm_counts_hits_and_random_expectation():
    d = pd.DataFrame({"mlbam_id": range(10), "name": list("abcdefghij"), "baseline_score": np.arange(10, 0, -1)})
    keep = pd.Series([True] * 8 + [False] * 2)
    out, hit = recruitment_backtest.evaluate_arm(d, keep, {0, 5, 9}, n_reachable=4, ks=[2])
    assert out["N"] == 8 and out["positives_in_list"] == 2 and out["hits@2"] == 1
    assert out["random@2"] == round(2 * 2 / 8, 2) and out["coverage"] == 0.5
    assert hit.list_rank.tolist() == [1, 6]


def test_trade_or_waiver_claim_keeps_the_free_agent_contract():
    rows = [(6, "2022-06-01", "SGN", 100), (6, "2024-07-01", "REL", 100), (6, "2025-01-05", "SFA", 100),
            (6, "2025-06-01", "TR", 200), (6, "2025-07-01", "CLW", 100)]
    assert _tier(rows)[6] == "T2_expiring"


def test_college_and_showcase_assignments_do_not_start_the_pro_clock():
    rows = [(7, "2019-07-20", "ASG", 900), (7, "2022-07-25", "SGN", 100)]     # 900 = college summer team
    assert _tier(rows, draft=2022)[7] == "T3_controlled"
