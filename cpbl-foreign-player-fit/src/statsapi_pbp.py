"""AAA pitch-level data straight from the MLB Stats API game feed (statsapi.mlb.com).

The GitHub mirror (milb_loader) stops publishing pitch-level files after 2025-05-25.
This module fills the gap — the rest of 2025 and all of 2026 — from the same upstream
feed, written in the mirror's column layout so milb_loader.to_schema() treats both
sources identically:

  at_bat_number = about.atBatIndex (0-based, as in the mirror)
  pitch_number  = playEvents[].pitchNumber
  type          = details.call.code (kept verbatim, e.g. '*B' ball in dirt, as in the mirror)
  balls/strikes = count AFTER the pitch (as in the mirror; the loader shifts it)
  balls_before/strikes_before = count BEFORE the pitch, taken from the previous event of the
                  plate appearance, so automatic balls/strikes (pitch-timer violations, which
                  are not pitches) still move the count. Not in the mirror; the loader uses
                  these where present and the shifted count otherwise.
  pitcher/batter (and p_throws/stand) = the players on the mound and at the plate for THAT
                  pitch: a mid-PA pitching change or pinch-hitter is undone for the earlier
                  pitches (the feed's matchup only names the final pair). The mirror credits the
                  final pitcher to every pitch and takes the batter from the PA's first event,
                  a pinch-runner when the PA opens with that substitution (3 PAs, 9 of 6,294
                  pitches in the 20-game overlap test).
  des           = result.description of the plate appearance
  pfx_x/pfx_z, plate_x/plate_z = pitchData.coordinates pfxX/pfxZ (inches), pX/pZ (feet)

Checked against the mirror on overlapping games (tests/test_data_consistency.py): identical
rows and physics; MLBAM later reclassifies some pitch types and strike-zone cells, so a few
% of `pitch_type` / `zone` values differ from the mirror's earlier capture.

Only isPitch events are kept (pickoffs, step-offs and pitch-clock no-pitches are not pitches).
Games are cached one file per game under data/raw/statsapi_pbp/games/ so the download
resumes; requests are throttled (a few at a time) to stay polite to the public API.

Usage:  python -m src.statsapi_pbp
"""
from __future__ import annotations

import gzip
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from .common import config, raw, snapshot

API = "https://statsapi.mlb.com/api"
COLUMNS = [
    "game_pk", "game_date", "game_type", "league_level_name", "league_name", "at_bat_number", "pitch_number",
    "inning", "inning_top_bot", "home_team", "away_team", "pitcher", "batter", "p_throws", "stand",
    "type", "des", "pitch_type", "release_speed", "release_spin_rate", "release_extension",
    "pfx_x", "pfx_z", "plate_x", "plate_z", "zone", "launch_speed", "launch_angle", "balls", "strikes",
]
PRE_COUNT = ["balls_before", "strikes_before"]
_session = requests.Session()


def _get(url: str, params: dict | None = None, tries: int = 4) -> dict:
    for i in range(tries):
        try:
            r = _session.get(url, params=params, timeout=60)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (404,):
                r.raise_for_status()
        except requests.RequestException:
            if i == tries - 1:
                raise
        time.sleep(2 ** (i + 1))
    raise RuntimeError(f"failed: {url}")


def schedule(start: str, end: str) -> pd.DataFrame:
    """Completed AAA regular-season games between start and end (inclusive)."""
    d = _get(f"{API}/v1/schedule", {"sportId": 11, "gameType": "R", "startDate": start, "endDate": end})
    rows = []
    for day in d.get("dates", []):
        for g in day["games"]:
            rows.append({"game_pk": g["gamePk"], "date": g.get("officialDate", day["date"]),
                         "state": g["status"]["abstractGameState"], "detailed": g["status"]["detailedState"]})
    df = pd.DataFrame(rows).drop_duplicates("game_pk")
    return df[df.state.eq("Final") & ~df.detailed.str.contains("Cancel|Postpon", na=False)]


def _hand(players: dict, pid: int, key: str) -> str | None:
    return ((players.get(f"ID{pid}") or {}).get(key) or {}).get("code")


def _identities(play: dict, players: dict, prev_pitcher: int | None) -> list[tuple]:
    """(pitcher, batter, p_throws, stand) for every play event, undoing mid-PA substitutions.

    Walks the events backwards from the final matchup. A pitching change carries no
    replacedPlayer, so the pitcher it replaced is the fielding team's previous pitcher
    (prev_pitcher); a pinch-hitter's replacedPlayer names the original batter.
    """
    mu = play["matchup"]
    pit, bat = mu["pitcher"]["id"], mu["batter"]["id"]
    thr, side = mu.get("pitchHand", {}).get("code"), mu.get("batSide", {}).get("code")
    out = []
    for e in reversed(play["playEvents"]):
        out.append((pit, bat, thr, side))
        det, who = e.get("details") or {}, (e.get("player") or {}).get("id")
        if not e.get("isSubstitution"):
            continue
        if det.get("eventType") == "pitching_substitution" and who == pit:
            old = (e.get("replacedPlayer") or {}).get("id") or prev_pitcher
            if old and old != pit:
                pit, thr = old, _hand(players, old, "pitchHand") or thr
                if _hand(players, bat, "batSide") == "S":
                    side = "L" if thr == "R" else "R"
        elif det.get("eventType") == "offensive_substitution" and who == bat:
            old = (e.get("replacedPlayer") or {}).get("id")
            if old:
                bat = old
                side = _hand(players, old, "batSide") or side
                if side == "S":
                    side = "L" if thr == "R" else "R"
    return out[::-1]


def feed_rows(feed: dict) -> list[dict]:
    gd = feed["gameData"]
    home, away = gd["teams"]["home"], gd["teams"]["away"]
    players = gd.get("players", {})
    base = {
        "game_pk": gd["game"]["pk"], "game_date": gd["datetime"]["officialDate"], "game_type": gd["game"]["type"],
        "league_level_name": home.get("sport", {}).get("name"), "league_name": home.get("league", {}).get("name"),
        "home_team": home.get("abbreviation"), "away_team": away.get("abbreviation"),
    }
    out = []
    last_pitcher = {}                       # fielding half ("Top"/"Bot") -> pitcher of its previous PA
    for play in feed["liveData"]["plays"]["allPlays"]:
        ab, res = play["about"], play["result"]
        half = "Top" if ab["isTopInning"] else "Bot"
        ids = _identities(play, players, last_pitcher.get(half))
        last_pitcher[half] = play["matchup"]["pitcher"]["id"]
        pa = {"at_bat_number": ab["atBatIndex"], "inning": ab["inning"],
              "inning_top_bot": "Top" if ab["isTopInning"] else "Bot", "des": res.get("description")}
        count = {"balls": 0, "strikes": 0}
        for e, (pit, bat, thr, side) in zip(play["playEvents"], ids):
            before, count = count, e.get("count") or count
            if not e.get("isPitch"):
                continue
            det, pdat, hd = e.get("details", {}), e.get("pitchData", {}) or {}, e.get("hitData", {}) or {}
            co, br = pdat.get("coordinates", {}) or {}, pdat.get("breaks", {}) or {}
            code = (det.get("call") or {}).get("code")
            out.append({**base, **pa,
                        "pitcher": pit, "batter": bat, "p_throws": thr, "stand": side,
                        "pitch_number": e.get("pitchNumber"),
                        "type": code,
                        "pitch_type": (det.get("type") or {}).get("code"),
                        "release_speed": pdat.get("startSpeed"), "release_spin_rate": br.get("spinRate"),
                        "release_extension": pdat.get("extension"),
                        "pfx_x": co.get("pfxX"), "pfx_z": co.get("pfxZ"), "plate_x": co.get("pX"),
                        "plate_z": co.get("pZ"), "zone": pdat.get("zone"),
                        "launch_speed": hd.get("launchSpeed"), "launch_angle": hd.get("launchAngle"),
                        "balls": count.get("balls"), "strikes": count.get("strikes"),
                        "balls_before": before.get("balls"), "strikes_before": before.get("strikes")})
    return out


def _game_path(pk: int):
    return raw("statsapi_pbp", "games", f"{pk}.csv.gz")


def _current(p) -> bool:
    """Cached file exists and was written by this parser version (has the pre-pitch count)."""
    if not p.exists():
        return False
    with gzip.open(p, "rt", encoding="utf-8") as f:
        return PRE_COUNT[0] in f.readline()


def fetch_game(pk: int) -> int:
    p = _game_path(pk)
    if _current(p):
        return -1
    feed = _get(f"{API}/v1.1/game/{pk}/feed/live")
    df = pd.DataFrame(feed_rows(feed), columns=COLUMNS + PRE_COUNT)
    tmp = p.with_suffix(".part")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        df.to_csv(f, index=False)
    tmp.rename(p)
    return len(df)


def fetch(workers: int = 4) -> pd.DataFrame:
    """Download every configured window; returns the schedule that was requested."""
    windows = config()["sources"]["statsapi_pbp_windows"]
    snap = snapshot().strftime("%Y-%m-%d")
    games = []
    for year, (start, end) in windows.items():
        games.append(schedule(start, min(end, snap)).assign(season=int(year)))
    sched = pd.concat(games, ignore_index=True)
    raw("statsapi_pbp", "games").mkdir(parents=True, exist_ok=True)
    sched.to_csv(raw("statsapi_pbp", "schedule.csv"), index=False)
    todo = [pk for pk in sched.game_pk if not _current(_game_path(pk))]
    print(f"statsapi: {len(sched)} final AAA games, {len(todo)} to download")
    done = 0
    with ThreadPoolExecutor(workers) as ex:
        for n in ex.map(fetch_game, todo):
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(todo)}", flush=True)
    return sched


def read_all(usecols: list[str] | None = None) -> pd.DataFrame:
    sched_path = raw("statsapi_pbp", "schedule.csv")
    if not sched_path.exists():
        return pd.DataFrame(columns=usecols or COLUMNS + PRE_COUNT)
    sched = pd.read_csv(sched_path)
    frames = []
    for pk in sched.game_pk:
        p = _game_path(pk)
        if p.exists():
            frames.append(pd.read_csv(p, usecols=usecols, low_memory=False))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=usecols or COLUMNS + PRE_COUNT)


if __name__ == "__main__":
    s = fetch()
    missing = [pk for pk in s.game_pk if not _game_path(pk).exists()]
    print("missing:", len(missing))
