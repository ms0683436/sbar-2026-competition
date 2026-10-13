"""Eligibility rules: boundaries, per-season (not summed), thirds of innings, unknowns."""
import re

import pandas as pd
import pytest

from src.common import ip_to_outs, outs_to_ip_str
from src.eligibility import decide, regular_season_files


@pytest.mark.parametrize("ip,outs", [("40.0", 120), ("40", 120), ("39.2", 119), ("39.1", 118), ("0.1", 1),
                                     (12.333333, 37), (12.666667, 38), (40.0, 120), ("40.1", 121)])
def test_ip_to_outs_thirds(ip, outs):
    assert ip_to_outs(ip) == outs


def test_ip_invalid_fraction_rejected():
    with pytest.raises(ValueError):
        ip_to_outs("12.5")


def test_outs_round_trip():
    assert outs_to_ip_str(119) == "39.2" and outs_to_ip_str(120) == "40.0"


def test_hitter_boundary_inclusive():
    assert decide("hitter", {2024: 200, 2025: 200})[0] is True
    assert decide("hitter", {2024: 201, 2025: 0})[0] is False
    assert decide("hitter", {2024: 0, 2025: 201})[0] is False


def test_pitcher_boundary_inclusive_in_outs():
    assert decide("pitcher", {2024: ip_to_outs("40.0"), 2025: 0})[0] is True
    assert decide("pitcher", {2024: ip_to_outs("40.1"), 2025: 0})[0] is False
    assert decide("pitcher", {2024: ip_to_outs("39.2"), 2025: ip_to_outs("39.2")})[0] is True


def test_seasons_checked_separately_not_summed():
    # 150 + 150 = 300 > 200, but each season is <= 200 -> eligible
    assert decide("hitter", {2024: 150, 2025: 150})[0] is True
    # 35 + 35 IP = 70 > 40, but each season <= 40 -> eligible
    assert decide("pitcher", {2024: 105, 2025: 105})[0] is True


def test_unconfirmed_season_is_not_eligible():
    ok, reason = decide("hitter", {2024: 10, 2025: None})
    assert ok is None and "unconfirmed" in reason
    # but an over-limit season still makes the player ineligible even if the other is unknown
    assert decide("hitter", {2024: 500, 2025: None})[0] is False


def _have_retro():
    try:
        regular_season_files(2025)
        return True
    except FileNotFoundError:
        return False


@pytest.mark.skipif(not _have_retro(), reason="Retrosheet clone not present (run src.fetch_data)")
def test_regular_season_only_files():
    for season in (2024, 2025):
        files = regular_season_files(season)
        names = [f.rsplit("/", 1)[-1] for f in files]
        # regular-season team files only: <yyyy><TEAM>.EVA/.EVN (postseason/All-Star files are .EVE)
        assert len(names) == 30 and all(re.fullmatch(rf"{season}[A-Z]{{3}}\.EV[AN]", n) for n in names)


def test_mlb_season_lines_match_published_totals():
    """Cross-check the cached Retrosheet aggregation against well-known published lines."""
    from src.common import processed
    p = processed("mlb_season_lines.csv")
    if not p.exists():
        pytest.skip("processed/mlb_season_lines.csv not built")
    lines = pd.read_csv(p).set_index(["season", "retro_id"])
    assert lines.loc[(2024, "ohtas001"), "pa"] == 731      # Shohei Ohtani 2024: 731 PA
    assert lines.loc[(2024, "judga001"), "pa"] == 704      # Aaron Judge 2024: 704 PA
    assert lines.loc[(2024, "skubt001"), "p_outs"] == 576  # Tarik Skubal 2024: 192.0 IP
    assert lines.loc[(2024, "salec001"), "p_outs"] == 533  # Chris Sale 2024: 177.2 IP
    audit = pd.read_csv(processed("mlb_season_audit.csv")).set_index("season")
    assert audit.loc[2024, "games"] == 2429 and audit.loc[2025, "games"] == 2430
