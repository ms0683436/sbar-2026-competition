"""Amendment 8 — release point and spin axis for AAA pitches (statsapi game feed).

The cached feed extracts (src/statsapi_pbp.py) and the GitHub mirror do not carry the same
release-point fields, so the Stuff+ inputs are re-read from one source for every season:
pitchData.coordinates x0 / z0 (horizontal and vertical position at y0 = 50 ft, the feed's
release proxy), pitchData.breaks.spinDirection, plus speed, spin rate, extension and pfx for a
consistency check. One small CSV per game under data/raw/statsapi_release/ (git-ignored, ©
MLBAM, not redistributed).

Usage:  python -m src.statsapi_release 2023 2024 2025
"""
from __future__ import annotations

import gzip
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from . import statsapi_pbp
from .common import raw

COLS = ["game_pk", "at_bat_number", "pitch_number", "pitcher", "release_speed", "release_spin_rate",
        "spin_axis", "release_extension", "x0", "z0", "pfx_x", "pfx_z"]
SEASONS = {2023: ("2023-03-28", "2023-09-30"), 2024: ("2024-03-26", "2024-09-29"),
           2025: ("2025-03-25", "2025-09-28"), 2026: ("2026-03-24", "2026-09-27")}


def _path(pk: int):
    return raw("statsapi_release", "games", f"{pk}.csv.gz")


def rows(feed: dict) -> list[dict]:
    pk = feed.get("gamePk")
    out = []
    for play in feed["liveData"]["plays"]["allPlays"]:
        ab = play["about"]["atBatIndex"]
        for e in play["playEvents"]:
            if not e.get("isPitch"):
                continue
            pd_, co = e.get("pitchData", {}) or {}, (e.get("pitchData", {}) or {}).get("coordinates", {}) or {}
            br = pd_.get("breaks", {}) or {}
            out.append({"game_pk": pk, "at_bat_number": ab, "pitch_number": e.get("pitchNumber"),
                        "pitcher": play["matchup"]["pitcher"]["id"], "release_speed": pd_.get("startSpeed"),
                        "release_spin_rate": br.get("spinRate"), "spin_axis": br.get("spinDirection"),
                        "release_extension": pd_.get("extension"), "x0": co.get("x0"), "z0": co.get("z0"),
                        "pfx_x": co.get("pfxX"), "pfx_z": co.get("pfxZ")})
    return out


def fetch_game(pk: int) -> int:
    p = _path(pk)
    if p.exists():
        return -1
    feed = statsapi_pbp._get(f"{statsapi_pbp.API}/v1.1/game/{pk}/feed/live")
    df = pd.DataFrame(rows(feed), columns=COLS)
    tmp = p.with_suffix(".part")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        df.to_csv(f, index=False)
    tmp.rename(p)
    return len(df)


def schedule(season: int) -> pd.DataFrame:
    f = raw("statsapi_release", f"schedule_{season}.csv")
    if f.exists():
        return pd.read_csv(f)
    s = statsapi_pbp.schedule(*SEASONS[season]).assign(season=season)
    f.parent.mkdir(parents=True, exist_ok=True)
    s.to_csv(f, index=False)
    return s


def fetch(seasons, workers: int = 4) -> None:
    raw("statsapi_release", "games").mkdir(parents=True, exist_ok=True)
    for y in seasons:
        todo = [pk for pk in schedule(y).game_pk if not _path(pk).exists()]
        print(f"release {y}: {len(todo)} games to download", flush=True)
        with ThreadPoolExecutor(workers) as ex:
            for i, _ in enumerate(ex.map(fetch_game, todo)):
                if i % 250 == 0:
                    print(f"  {y}: {i}/{len(todo)}", flush=True)


def load(seasons) -> pd.DataFrame:
    frames = []
    for y in seasons:
        pks = schedule(y).game_pk
        frames += [pd.read_csv(_path(pk)).assign(season=y) for pk in pks if _path(pk).exists()]
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    fetch([int(a) for a in sys.argv[1:]] or [2025])
