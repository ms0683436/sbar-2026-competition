"""Cross-source consistency: de-duplication, identical swing/whiff definitions, units,
zone geometry, snapshot cut-off."""
import numpy as np
import pandas as pd
import pytest

from src import milb_loader, rebas_loader, statsapi_pbp
from src.common import processed, raw


def _aaa_fixture():
    base = {c: np.nan for c in milb_loader.USECOLS}
    rows = []
    # one PA: called strike, foul, swinging strike blocked (K) — then a 2nd PA single on first pitch
    seq = [(0, 1, "C", 0, 1, "Al Bo strikes out swinging."), (0, 2, "F", 0, 2, "Al Bo strikes out swinging."),
           (0, 3, "W", 0, 3, "Al Bo strikes out swinging."), (1, 1, "D", 0, 0, "Cy Do singles on a line drive.")]
    for ab, pn, code, b, s, des in seq:
        r = dict(base, game_pk=1, game_date="2025-04-01", game_type="R", league_level_name="Triple-A",
                 league_name="International League", at_bat_number=ab, pitch_number=pn, inning=1,
                 inning_top_bot="Top", home_team="H", away_team="A", pitcher=10, batter=20 + ab, p_throws="R",
                 stand="L", type=code, des=des, pitch_type="ST", release_speed=85.0, zone=5, balls=b, strikes=s)
        rows.append(r)
    late = dict(rows[-1], game_pk=2, game_date="2026-10-20", at_bat_number=0)   # after snapshot
    df = pd.DataFrame(rows + rows + [late])      # duplicated like the mirror's scraper output
    return df


def test_aaa_dedup_snapshot_and_flags():
    raw = _aaa_fixture()
    dedup = raw.drop_duplicates(["game_pk", "at_bat_number", "pitch_number"])
    s = milb_loader.to_schema(dedup)
    assert len(s) == 4                                  # duplicates gone, post-snapshot game dropped
    assert s.date.max() <= pd.Timestamp("2026-10-13")
    assert list(s.swing) == [False, True, True, True]
    assert list(s.whiff) == [False, False, True, False]  # blocked swinging strike is a whiff
    assert list(s.pa_event.dropna()) == ["SO", "1B"]
    assert list(s.strikes) == [0, 1, 2, 0]               # count BEFORE the pitch
    assert (s.pitch_type == "SL").all()                  # ST harmonised to SL


def test_cpbl_codes_map_to_same_definitions():
    df = pd.DataFrame({"pitch_code": ["S", "SW", "F", "FT", "H", "B", "BUNT", "TRY_BUNT"]})
    swing = df.pitch_code.isin(rebas_loader.SWING_CODES)
    whiff = df.pitch_code.eq("SW")
    bunt = df.pitch_code.isin(rebas_loader.BUNT_CODES)
    assert list(swing) == [False, True, True, True, True, False, False, False]
    assert list(whiff) == [False, True, False, False, False, False, False, False]
    assert list(bunt)[-2:] == [True, True]
    # AAA equivalents: S(swinging) W(blocked) -> whiff; T(foul tip) is contact in both leagues
    assert "T" in milb_loader.SWING_CODES and "T" not in milb_loader.WHIFF_CODES
    assert "FT" in rebas_loader.SWING_CODES


def test_zone_geometry_matches_rulebook_box():
    x = pd.Series([0, 50, 50.1, -50, 0, 0, np.nan])
    y = pd.Series([0, 60, 0, -60, 60.1, -61, 0])
    z = rebas_loader.in_zone(x, y)
    assert list(z[:6]) == [True, True, False, True, False, False]
    assert pd.isna(z.iloc[6])


def test_kmh_to_mph():
    assert abs(150 / 1.609344 - 93.2057) < 1e-3


@pytest.mark.skipif(not processed("cpbl_pitches.parquet").exists(), reason="CPBL pitches not built")
def test_rebas_seasons_complete_and_unique():
    d = pd.read_parquet(processed("cpbl_pitches.parquet"))
    g = d.groupby("season").game_id.nunique()
    assert g.loc[2024] == 360 and g.loc[2025] == 360
    assert not d.duplicated(["pa_id", "pitch_seq"]).any()
    assert d.pitcher_id.notna().all() and d.batter_id.notna().all()


@pytest.mark.skipif(not processed("aaa_pitches.parquet").exists(), reason="AAA pitches not built")
def test_aaa_pitches_unique_and_regular_season():
    d = pd.read_parquet(processed("aaa_pitches.parquet"), columns=["game_id", "pa_id", "pitch_seq", "date", "season"])
    assert not d.duplicated(["pa_id", "pitch_seq"]).any()
    assert d.date.max() <= pd.Timestamp("2026-10-13")
    assert set(d.season.unique()) <= {2024, 2025, 2026}


def _feed_fixture():
    def pitch(n, code, b, s, speed=95.0):
        return {"isPitch": True, "type": "pitch", "pitchNumber": n, "count": {"balls": b, "strikes": s},
                "details": {"call": {"code": code}, "type": {"code": "FF"}},
                "pitchData": {"startSpeed": speed, "extension": 6.5, "zone": 5,
                              "coordinates": {"pfxX": -6.0, "pfxZ": 9.0, "pX": 0.1, "pZ": 2.5},
                              "breaks": {"spinRate": 2300}}}

    def action(kind, who, b, s, replaced=None):
        e = {"isPitch": False, "type": "action", "isSubstitution": True, "player": {"id": who},
             "details": {"eventType": kind, "description": kind}, "count": {"balls": b, "strikes": s}}
        return e | ({"replacedPlayer": {"id": replaced}} if replaced else {})
    pickoff = {"isPitch": False, "type": "pickoff", "details": {"description": "Pickoff Attempt 1B"},
               "count": {"balls": 1, "strikes": 0}}
    auto_strike = {"isPitch": False, "type": "no_pitch", "count": {"balls": 0, "strikes": 1},
                   "details": {"call": {"code": "AC"}, "description": "Automatic Strike - Batter Pitch Timer Violation"}}
    mu = lambda p, b, ph="L", bs="R": {"pitcher": {"id": p}, "batter": {"id": b}, "pitchHand": {"code": ph},
                                       "batSide": {"code": bs}}
    plays = [
        # PA 0: pickoff between pitches
        {"about": {"atBatIndex": 0, "inning": 1, "isTopInning": False}, "matchup": mu(10, 20),
         "result": {"description": "Al Bo strikes out swinging."},
         "playEvents": [pitch(1, "*B", 1, 0), pickoff, pitch(2, "F", 1, 1), pitch(3, "S", 1, 2)]},
        # PA 1: automatic strike before the first pitch, then a mid-PA pitching change (no
        # replacedPlayer in the feed) and a pinch-hitter (with replacedPlayer)
        {"about": {"atBatIndex": 1, "inning": 1, "isTopInning": False}, "matchup": mu(11, 22, ph="R", bs="L"),
         "result": {"description": "Ed Fo walks."},
         "playEvents": [auto_strike, pitch(1, "B", 1, 1), action("pitching_substitution", 11, 1, 1),
                        pitch(2, "B", 2, 1), action("offensive_substitution", 22, 2, 1, replaced=21),
                        pitch(3, "B", 3, 1), pitch(4, "B", 4, 1)]},
    ]
    team = lambda ab: {"abbreviation": ab, "league": {"name": "Pacific Coast League"}, "sport": {"name": "Triple-A"}}
    players = {"ID10": {"pitchHand": {"code": "L"}}, "ID11": {"pitchHand": {"code": "R"}},
               "ID21": {"batSide": {"code": "R"}}, "ID22": {"batSide": {"code": "L"}}}
    return {"gameData": {"game": {"pk": 99, "type": "R"}, "datetime": {"officialDate": "2026-06-01"},
                         "teams": {"home": team("ABQ"), "away": team("SL")}, "players": players},
            "liveData": {"plays": {"allPlays": plays}}}


def test_statsapi_feed_rows_use_mirror_layout():
    rows = pd.DataFrame(statsapi_pbp.feed_rows(_feed_fixture()), columns=statsapi_pbp.COLUMNS + statsapi_pbp.PRE_COUNT)
    assert statsapi_pbp.COLUMNS == milb_loader.USECOLS          # same layout as the mirror (+ pre-pitch count)
    pa0 = rows[rows.at_bat_number == 0]
    assert len(pa0) == 3                                         # pickoff is not a pitch
    assert list(pa0.pitch_number) == [1, 2, 3]                   # 0-based PA index like the mirror
    assert list(pa0.type) == ["*B", "F", "S"]                    # call codes verbatim
    assert list(pa0.strikes) == [0, 1, 2]                        # count AFTER the pitch, as in the mirror
    assert (rows.inning_top_bot == "Bot").all() and (rows.home_team == "ABQ").all()
    s = milb_loader.to_schema(rows.assign(source="statsapi"))
    s0 = s[s.pa_id == "99-0"]
    assert list(s0.strikes) == [0, 0, 1]                         # loader gives the count BEFORE the pitch
    assert list(s0.whiff) == [False, False, True] and list(s0.pa_event.dropna()) == ["SO"]
    assert (s.pitcher_team == "SL").all()                        # bottom half: away team pitches


def test_statsapi_automatic_strike_moves_the_count():
    rows = pd.DataFrame(statsapi_pbp.feed_rows(_feed_fixture()), columns=statsapi_pbp.COLUMNS + statsapi_pbp.PRE_COUNT)
    s = milb_loader.to_schema(rows.assign(source="statsapi"))
    s1 = s[s.pa_id == "99-1"]
    assert list(s1.strikes) == [1, 1, 1, 1]                      # the timer violation's strike counts
    assert list(s1.balls) == [0, 1, 2, 3]


def test_statsapi_mid_pa_substitutions_credit_the_right_players():
    rows = pd.DataFrame(statsapi_pbp.feed_rows(_feed_fixture()))
    pa1 = rows[rows.at_bat_number == 1].set_index("pitch_number")
    assert list(pa1.pitcher) == [10, 11, 11, 11]                 # pitch 1 thrown before the change
    assert list(pa1.p_throws) == ["L", "R", "R", "R"]
    assert list(pa1.batter) == [21, 21, 22, 22]                  # pinch-hitter enters after pitch 2
    assert list(pa1.stand) == ["R", "R", "L", "L"]


def test_statsapi_starter_replaced_in_first_pa():
    """The box score names the starter, so a change during the team's first PA is undone."""
    feed = _feed_fixture()
    first = feed["liveData"]["plays"]["allPlays"][0]
    pitch = first["playEvents"][0]
    change = {"isPitch": False, "type": "action", "isSubstitution": True, "player": {"id": 11},
              "details": {"eventType": "pitching_substitution"}, "count": {"balls": 1, "strikes": 0}}
    first["matchup"]["pitcher"], first["matchup"]["pitchHand"] = {"id": 11}, {"code": "R"}
    first["playEvents"] = [pitch, change, dict(pitch, pitchNumber=2), dict(pitch, pitchNumber=3)]
    feed["liveData"]["boxscore"] = {"teams": {"away": {"pitchers": [10, 11]}, "home": {"pitchers": [30]}}}
    rows = pd.DataFrame(statsapi_pbp.feed_rows(feed))
    pa0 = rows[rows.at_bat_number == 0]
    assert list(pa0.pitcher) == [10, 11, 11]
    assert list(pa0.p_throws) == ["L", "R", "R"]


def test_statsapi_two_pitching_changes_in_one_pa():
    """12 replaces 10 before the first pitch, then 11 replaces 12 after it (no replacedPlayer)."""
    feed = _feed_fixture()
    pitch = feed["liveData"]["plays"]["allPlays"][0]["playEvents"][0]
    change = lambda who: {"isPitch": False, "type": "action", "isSubstitution": True, "player": {"id": who},
                          "details": {"eventType": "pitching_substitution"}, "count": {"balls": 0, "strikes": 0}}
    play = {"about": {"atBatIndex": 2, "inning": 2, "isTopInning": False},
            "matchup": {"pitcher": {"id": 11}, "batter": {"id": 23}, "pitchHand": {"code": "R"}, "batSide": {"code": "R"}},
            "result": {"description": "Gi Ho walks."},
            "playEvents": [change(12), dict(pitch, pitchNumber=1), change(11), dict(pitch, pitchNumber=2),
                           dict(pitch, pitchNumber=3)]}
    feed["liveData"]["plays"]["allPlays"].append(play)
    feed["gameData"]["players"]["ID12"] = {"pitchHand": {"code": "L"}}
    rows = pd.DataFrame(statsapi_pbp.feed_rows(feed))
    pa2 = rows[rows.at_bat_number == 2]
    assert list(pa2.pitcher) == [12, 11, 11]
    assert list(pa2.p_throws) == ["L", "R", "R"]


def _overlap_games():
    mirror = raw("milb_pbp", "2025_5_aaa_pbp.csv")
    games = raw("statsapi_pbp", "games")
    if not mirror.exists() or not games.exists():
        return None, []
    m = pd.read_csv(mirror, usecols=milb_loader.USECOLS, low_memory=False)
    m = m[m.game_date >= "2025-05-20"].drop_duplicates(["game_pk", "at_bat_number", "pitch_number"])
    pks = [pk for pk in sorted(m.game_pk.unique()) if (games / f"{pk}.csv.gz").exists()][:20]
    return m, pks


@pytest.mark.skipif(not _overlap_games()[1], reason="mirror/statsapi overlap not downloaded")
def test_statsapi_matches_mirror_on_overlapping_games():
    m, pks = _overlap_games()
    key = ["game_pk", "at_bat_number", "pitch_number"]
    f = pd.concat(pd.read_csv(raw("statsapi_pbp", "games", f"{pk}.csv.gz")) for pk in pks)
    a = m[m.game_pk.isin(pks)].set_index(key).sort_index()
    b = f.set_index(key).sort_index()
    assert a.index.equals(b.index)                             # identical pitch rows
    for c in ("release_speed", "pfx_x", "pfx_z", "plate_x", "plate_z", "launch_speed"):
        assert np.allclose(a[c].astype(float), b[c].astype(float), equal_nan=True)
    for c in ("stand", "balls", "strikes"):
        assert (a[c] == b[c]).all(), c
    # same PA outcome (description wording can change later: name edits, ABS-challenge preamble)
    assert (a.des.map(milb_loader.classify_des) == b.des.map(milb_loader.classify_des)).mean() > 0.995
    # The mirror credits every pitch of a PA to the final pitcher (wrong before a mid-PA pitching
    # change: game 779632 PA 56, the first 3 pitches) and takes the batter from the PA's first
    # event (a pinch-runner when the PA opens with that substitution: games 779708, 779714).
    # The feed parser credits the players on the field for each pitch, so these few rows differ.
    for c in ("pitcher", "p_throws", "batter"):
        assert (a[c] == b[c]).mean() > 0.995, c
    # MLBAM re-classifies a few pitches after the game; the mirror captured them earlier
    assert (a.pitch_type == b.pitch_type).mean() > 0.9
