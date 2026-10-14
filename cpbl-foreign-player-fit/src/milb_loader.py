"""AAA pitch-level feed -> common pitch schema. Two sources, same upstream (MLB Stats API):
  * armstjc/milb-data-repository mirror: 2024 and 2025-03-28..05-25
  * statsapi.mlb.com game feed (src.statsapi_pbp, amendment 1): rest of 2025 and 2026
Mirror rows are kept first where the two overlap.

COUNT NOTE: both sources store the count AFTER each pitch. statsapi rows also carry the
count BEFORE the pitch (balls_before/strikes_before), read from the feed's previous event,
so automatic balls/strikes from pitch-timer violations (not pitches) are respected. The
mirror dropped those events, so for mirror rows (2024, 2025 to 05-25) the pre-pitch count
is reconstructed by shifting and misses them.

DATA-QUALITY NOTE: the mirror's scraper concatenates every successfully parsed
game twice (and re-appends the previous game when a game fails), so raw files
contain 2-3 copies of most pitches. Rows are de-duplicated on
(game_pk, at_bat_number, pitch_number) — see tests/test_data_consistency.py.
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import park, statsapi_pbp
from .common import config, processed, raw, snapshot
from .pitch_schema import finalize

USECOLS = [
    "game_pk", "game_date", "game_type", "league_level_name", "league_name", "at_bat_number", "pitch_number",
    "inning", "inning_top_bot", "home_team", "away_team", "pitcher", "batter", "p_throws", "stand",
    "type", "des", "pitch_type", "release_speed", "release_spin_rate", "release_extension",
    "pfx_x", "pfx_z", "plate_x", "plate_z", "zone", "launch_speed", "launch_angle", "balls", "strikes",
]

SWING_CODES = {"S", "W", "T", "F", "X", "D", "E", "Q", "R"}
WHIFF_CODES = {"S", "W", "Q"}
BUNT_CODES = {"L", "M", "O"}
NO_PITCH_CODES = {"V", "A", "N"}     # automatic balls/strikes (pitch clock), no-pitch

_EVENT_RE = re.compile(
    r"(intentionally walks|walks|hit by pitch|strikes out|called out on strikes|ground-rule double|"
    r"hits a grand slam|homers|triples|doubles|singles|out on a sacrifice fly|sacrifice fly|"
    r"out on a sacrifice bunt|sacrifice bunt|catcher interference|"
    r"grounds out|flies out|lines out|pops out|grounds into|lines into|flies into|pops into|"
    r"reaches on|fielder's choice|fielders choice|double play|triple play|"
    r"steals|caught stealing|picks off|picked off|wild pitch|passed ball|balk)"
)
_EVENT_MAP = {
    "intentionally walks": "IBB", "walks": "uBB", "hit by pitch": "HBP", "strikes out": "SO",
    "called out on strikes": "SO", "ground-rule double": "2B", "hits a grand slam": "HR", "homers": "HR",
    "triples": "3B", "doubles": "2B", "singles": "1B", "out on a sacrifice fly": "SF", "sacrifice fly": "SF",
    "out on a sacrifice bunt": "SH", "sacrifice bunt": "SH", "catcher interference": "CI",
}


def classify_des(des: str | float) -> str:
    """Batter's PA outcome from the play description (leftmost recognised verb)."""
    if not isinstance(des, str):
        return "OTHER"
    m = _EVENT_RE.search(des)
    if not m:
        return "OTHER"
    v = m.group(1)
    if v in _EVENT_MAP:
        return _EVENT_MAP[v]
    if v in ("steals", "caught stealing", "picks off", "picked off", "wild pitch", "passed ball", "balk"):
        return "OTHER"
    return "OUT"


def read_raw(pattern: str, statsapi: pd.DataFrame | None = None) -> pd.DataFrame:
    frames = []
    for f in sorted(glob.glob(str(raw("milb_pbp", pattern)))):
        frames.append(pd.read_csv(f, usecols=USECOLS, low_memory=False).assign(source="mirror"))
    if statsapi is not None and len(statsapi):
        frames.append(statsapi.assign(source="statsapi"))
    df = pd.concat(frames, ignore_index=True)
    n0 = len(df)
    df = df.drop_duplicates(["game_pk", "at_bat_number", "pitch_number"]).copy()
    df.attrs["n_raw"], df.attrs["n_dedup"] = n0, len(df)
    return df


def to_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df[(df.game_type == "R") & (df.league_level_name == "Triple-A")].copy()
    df["date"] = pd.to_datetime(df.game_date)
    df = df[df.date <= snapshot()]
    df = df[~df["type"].isin(NO_PITCH_CODES)]
    df = df.sort_values(["game_pk", "at_bat_number", "pitch_number"])
    df["league"] = "AAA"
    df["season"] = df.date.dt.year
    df["game_id"] = df.game_pk.astype(str)
    df["pa_id"] = df.game_pk.astype(str) + "-" + df.at_bat_number.astype(str)
    df["pitch_seq"] = df.pitch_number
    df["pitcher_id"] = df.pitcher.astype("Int64").astype(str)
    df["batter_id"] = df.batter.astype("Int64").astype(str)
    top = df.inning_top_bot.eq("Top")
    df["pitcher_team"] = np.where(top, df.home_team, df.away_team)
    df["batter_team"] = np.where(top, df.away_team, df.home_team)
    first = (df.sort_values(["game_pk", "inning", "at_bat_number"])
             .groupby(["game_pk", "pitcher_team"]).pitcher.first().rename("sp"))
    df = df.join(first, on=["game_pk", "pitcher_team"])
    df["pitcher_role"] = np.where(df.pitcher == df.sp, "SP", "RP")
    # the feed's balls/strikes are the count AFTER the pitch; shift to count before,
    # unless the source recorded the pre-pitch count itself (statsapi rows)
    g = df.groupby("pa_id")
    for c in ("balls", "strikes"):
        shifted = g[c].shift(1).fillna(0)
        if f"{c}_before" in df:
            shifted = pd.to_numeric(df[f"{c}_before"], errors="coerce").fillna(shifted)
        df[c] = shifted.clip(upper=3 if c == "balls" else 2).astype(int)
    code = df["type"].astype(str)
    bunt_in_play = code.isin(["X", "D", "E"]) & df.des.str.contains("bunt", case=False, na=False) & \
        df.groupby("pa_id").pitch_number.transform("max").eq(df.pitch_number)
    df["is_bunt"] = code.isin(BUNT_CODES) | bunt_in_play
    df["swing"] = code.isin(SWING_CODES)
    df["whiff"] = code.isin(WHIFF_CODES)
    df["called_strike"] = code.eq("C")
    df["in_zone"] = df.zone.between(1, 9).where(df.zone.notna())
    df["pitch_type_raw"] = df.pitch_type
    df["velo_mph"] = df.release_speed
    last = df.groupby("pa_id").pitch_number.transform("max").eq(df.pitch_number)
    event = df.des.map(classify_des)
    df["pa_end"] = last & event.ne("OTHER")
    df["pa_event"] = np.where(df.pa_end, event, None)
    extra = df[["release_spin_rate", "release_extension", "pfx_x", "pfx_z", "plate_x", "plate_z",
                "launch_angle", "pitch_type"]].rename(columns={"pitch_type": "_pt"})
    out = finalize(df)
    out = pd.concat([out, extra.drop(columns="_pt")], axis=1)
    out["launch_speed"] = df["launch_speed"]
    return out


def _inputs_newer_than(out) -> bool:
    """True if any raw input (mirror CSV, statsapi schedule or scheduled game file) is newer than the
    cache. Game files outside schedule.csv (the backtest's AAA 2023) are not inputs of this cache."""
    t = out.stat().st_mtime
    files = glob.glob(str(raw("milb_pbp", "*_aaa_pbp.csv")))
    sched = raw("statsapi_pbp", "schedule.csv")
    if sched.exists():
        files.append(str(sched))
        files += [str(statsapi_pbp._game_path(pk)) for pk in pd.read_csv(sched).game_pk]
    return any(Path(f).exists() and Path(f).stat().st_mtime > t for f in files)


def load(force: bool = False) -> pd.DataFrame:
    out = processed("aaa_pitches.parquet")
    if out.exists() and not force and not _inputs_newer_than(out):
        return pd.read_parquet(out)
    audits, frames = [], []
    api = statsapi_pbp.read_all(USECOLS + statsapi_pbp.PRE_COUNT)
    api_year = pd.to_datetime(api.game_date).dt.year if len(api) else pd.Series(dtype=int)
    years = sorted(set(config()["sources"]["milb_pbp_months"]) | set(api_year.unique().tolist()))
    for year in years:
        r = read_raw(f"{year}_*_aaa_pbp.csv", api[api_year == year] if len(api) else None)
        s = to_schema(r)
        src = r.loc[s.index, "source"] if len(s) else pd.Series(dtype=object)
        audits.append({"season": year, "rows_raw": r.attrs["n_raw"], "rows_dedup": r.attrs["n_dedup"],
                       "rows_regular_aaa": len(s), "games": s.game_id.nunique(),
                       "games_mirror": s.game_id[src.eq("mirror").values].nunique(),
                       "games_statsapi": s.game_id[src.eq("statsapi").values].nunique(),
                       "first_date": s.date.min(), "last_date": s.date.max()})
        frames.append(s)
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(out, index=False)
    pd.DataFrame(audits).to_csv(processed("aaa_pitches_audit.csv"), index=False)
    return df


# ------------------------------------------------------------ season / game stats
def season_stats(kind: str, year: int) -> pd.DataFrame:
    """AAA season totals per player x club stint; park-adjusted (amendment 5) when enabled."""
    df = pd.read_csv(raw("milb_season", f"{year}_aaa_season_{kind}_stats.csv"), low_memory=False)
    return park.adjust(df[df.team_level_abv == "AAA"].copy(), kind)


def amendment6() -> dict | None:
    """Amendment-6 settings when the two-season pool is registered, else None."""
    return config().get("amendment6")


def candidate_stats(kind: str) -> pd.DataFrame:
    """Season stints behind the candidate pool and the baseline's league table.

    Before amendment 6: the 2026 mirror season file. Amendment 6: the statsapi per-club tables
    of every pool season (park-adjusted, src.baseline_weights.fetch), with `team_league` set to
    "<season>_<league>" so a player's league mix (and the league table) is per season x league.
    """
    a6 = amendment6()
    if not a6:
        return season_stats(kind, config()["candidate_filters"]["pitcher" if kind == "pitching" else "hitter"]["season"])
    from .baseline_weights import fetch
    group = "pitching" if kind == "pitching" else "hitting"
    df = pd.concat([fetch(group, y) for y in a6["pool"]["seasons"]], ignore_index=True)
    df["league"] = df.team_league
    df["team_league"] = df.season.astype(str) + "_" + df.league
    return df


GAMELOG_PITCHING = {"gamesStarted": "pitching_GS", "battersFaced": "pitching_BF", "outs": "outs",
                    "hits": "pitching_H", "homeRuns": "pitching_HR", "baseOnBalls": "pitching_BB",
                    "intentionalWalks": "pitching_IBB", "strikeOuts": "pitching_SO", "hitByPitch": "pitching_HBP",
                    "earnedRuns": "pitching_ER"}
GAMELOG_HITTING = {"plateAppearances": "batting_PA", "atBats": "batting_AB", "hits": "batting_H",
                   "doubles": "batting_2B", "triples": "batting_3B", "homeRuns": "batting_HR",
                   "baseOnBalls": "batting_BB", "intentionalWalks": "batting_IBB", "strikeOuts": "batting_SO",
                   "hitByPitch": "batting_HBP", "sacFlies": "batting_SF", "totalBases": "batting_TB"}


def statsapi_game_logs(year: int, ids, group: str, workers: int = 4) -> pd.DataFrame:
    """Amendment 6: AAA per-game lines from the statsapi gameLog for seasons the mirror's game
    files do not cover. One request per player, cached per player under
    data/raw/statsapi_season/gamelog_<group>_<year>/. Same column names as the mirror logs."""
    from concurrent.futures import ThreadPoolExecutor
    d = raw("statsapi_season", f"gamelog_{group}_{year}")
    d.mkdir(parents=True, exist_ok=True)
    fields = GAMELOG_PITCHING if group == "pitching" else GAMELOG_HITTING

    def one(pid: int) -> pd.DataFrame:
        f = d / f"{pid}.csv"
        if f.exists():
            return pd.read_csv(f)
        js = statsapi_pbp._get(f"{statsapi_pbp.API}/v1/people/{pid}/stats",
                               {"stats": "gameLog", "group": group, "season": year, "sportId": 11, "gameType": "R"})
        rows = []
        for st in js.get("stats", []):
            for sp in st.get("splits", []):
                r = {"player_id": pid, "season": year, "game_id": sp["game"]["gamePk"], "game_date": sp["date"],
                     "team_id": sp.get("team", {}).get("id"), "loc": "H" if sp.get("isHome") else "A"}
                r.update({v: sp["stat"].get(k, 0) for k, v in fields.items()})
                rows.append(r)
        out = pd.DataFrame(rows, columns=["player_id", "season", "game_id", "game_date", "team_id", "loc", *fields.values()])
        out.to_csv(f, index=False)
        return out

    with ThreadPoolExecutor(workers) as ex:
        parts = list(ex.map(one, sorted(set(int(i) for i in ids))))
    df = pd.concat([p for p in parts if len(p)], ignore_index=True)
    if group == "pitching":
        df["pitching_IP"] = df.outs / 3
    df["game_date"] = pd.to_datetime(df.game_date)
    df = df[df.game_date <= snapshot()].drop_duplicates(["game_id", "player_id", "team_id"])
    if park.enabled():
        home = df["loc"].eq("H")
        df = park.adjust(df, "pitching" if group == "pitching" else "batting", home=home)
    return df


def game_logs(year: int) -> pd.DataFrame:
    files = sorted(glob.glob(str(raw("milb_game", f"{year}_*_aaa_player_game_stats.csv"))))
    df = pd.concat((pd.read_csv(f, low_memory=False) for f in files), ignore_index=True)
    n0 = len(df)
    df = df.drop_duplicates(["game_id", "player_id", "team_id"])
    df = df[(df.game_type == "R") & (df.league_level_name == "Triple-A")]
    df["game_date"] = pd.to_datetime(df.game_date)
    df = df[df.game_date <= snapshot()]
    if park.enabled():          # amendment 5: home games divided by the club's park factor
        home = df["loc"].eq("H")
        df = park.adjust(park.adjust(df, "pitching", home=home), "batting", home=home)
    df.attrs["n_raw"] = n0
    return df


if __name__ == "__main__":
    d = load(force=True)
    print(pd.read_csv(processed("aaa_pitches_audit.csv")))
    print(d.pa_event.value_counts())
