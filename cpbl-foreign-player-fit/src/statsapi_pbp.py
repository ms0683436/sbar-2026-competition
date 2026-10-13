"""AAA pitch-level data straight from the MLB Stats API game feed (statsapi.mlb.com).

The GitHub mirror (milb_loader) stops publishing pitch-level files after 2025-05-25.
This module fills the gap — the rest of 2025 and all of 2026 — from the same upstream
feed, written in the mirror's column layout so milb_loader.to_schema() treats both
sources identically:

  at_bat_number = about.atBatIndex (0-based, as in the mirror)
  pitch_number  = playEvents[].pitchNumber
  type          = details.call.code (kept verbatim, e.g. '*B' ball in dirt, as in the mirror)
  balls/strikes = count AFTER the pitch (as in the mirror; the loader shifts it)
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


def feed_rows(feed: dict) -> list[dict]:
    gd = feed["gameData"]
    home, away = gd["teams"]["home"], gd["teams"]["away"]
    base = {
        "game_pk": gd["game"]["pk"], "game_date": gd["datetime"]["officialDate"], "game_type": gd["game"]["type"],
        "league_level_name": home.get("sport", {}).get("name"), "league_name": home.get("league", {}).get("name"),
        "home_team": home.get("abbreviation"), "away_team": away.get("abbreviation"),
    }
    out = []
    for play in feed["liveData"]["plays"]["allPlays"]:
        ab, mu, res = play["about"], play["matchup"], play["result"]
        pa = {"at_bat_number": ab["atBatIndex"], "inning": ab["inning"],
              "inning_top_bot": "Top" if ab["isTopInning"] else "Bot",
              "pitcher": mu["pitcher"]["id"], "batter": mu["batter"]["id"],
              "p_throws": mu.get("pitchHand", {}).get("code"), "stand": mu.get("batSide", {}).get("code"),
              "des": res.get("description")}
        for e in play["playEvents"]:
            if not e.get("isPitch"):
                continue
            det, pdat, hd = e.get("details", {}), e.get("pitchData", {}) or {}, e.get("hitData", {}) or {}
            co, br = pdat.get("coordinates", {}) or {}, pdat.get("breaks", {}) or {}
            code = (det.get("call") or {}).get("code")
            out.append({**base, **pa,
                        "pitch_number": e.get("pitchNumber"),
                        "type": code,
                        "pitch_type": (det.get("type") or {}).get("code"),
                        "release_speed": pdat.get("startSpeed"), "release_spin_rate": br.get("spinRate"),
                        "release_extension": pdat.get("extension"),
                        "pfx_x": co.get("pfxX"), "pfx_z": co.get("pfxZ"), "plate_x": co.get("pX"),
                        "plate_z": co.get("pZ"), "zone": pdat.get("zone"),
                        "launch_speed": hd.get("launchSpeed"), "launch_angle": hd.get("launchAngle"),
                        "balls": (e.get("count") or {}).get("balls"), "strikes": (e.get("count") or {}).get("strikes")})
    return out


def _game_path(pk: int):
    return raw("statsapi_pbp", "games", f"{pk}.csv.gz")


def fetch_game(pk: int) -> int:
    p = _game_path(pk)
    if p.exists():
        return -1
    feed = _get(f"{API}/v1.1/game/{pk}/feed/live")
    df = pd.DataFrame(feed_rows(feed), columns=COLUMNS)
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
    todo = [pk for pk in sched.game_pk if not _game_path(pk).exists()]
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
        return pd.DataFrame(columns=usecols or COLUMNS)
    sched = pd.read_csv(sched_path)
    frames = []
    for pk in sched.game_pk:
        p = _game_path(pk)
        if p.exists():
            frames.append(pd.read_csv(p, usecols=usecols, low_memory=False))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=usecols or COLUMNS)


if __name__ == "__main__":
    s = fetch()
    missing = [pk for pk in s.game_pk if not _game_path(pk).exists()]
    print("missing:", len(missing))
