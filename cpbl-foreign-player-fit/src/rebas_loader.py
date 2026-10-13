"""REBAS 野球革命 Open Data (ODC-By 1.0) -> common pitch schema.

Each release zip holds one season-wide JSON plus one JSON per game; the two
overlap completely, so games are de-duplicated on (seasonId, seq).
REBAS PA records only carry player NAMES; ids are recovered from the box
scores of the same game (name + team -> playerId).
"""
from __future__ import annotations

import json
import zipfile

import numpy as np
import pandas as pd

from .common import processed, raw
from .pitch_schema import finalize

SWING_CODES = {"SW", "F", "FT", "H"}
BUNT_CODES = {"BUNT", "FOUL_BUNT", "TRY_BUNT"}

RESULT_MAP = {
    "1B": "1B", "2B": "2B", "3B": "3B", "HR": "HR", "IHR": "HR",
    "SO": "SO", "uBB": "uBB", "IBB": "IBB", "HBP": "HBP",
    "GO": "OUT", "FO": "OUT", "FC": "OUT", "E": "OUT", "GIDP": "OUT", "DP": "OUT", "TP": "OUT",
    "SF": "SF", "SH": "SH", "IH": "CI",
}


def read_games(season: int) -> list[dict]:
    z = zipfile.ZipFile(raw("rebas", f"CPBL-{season}-OpenData.zip"))
    games: dict[tuple, dict] = {}
    for info in z.infolist():
        if not info.filename.endswith(".json"):
            continue
        obj = json.loads(z.read(info))
        for g in obj if isinstance(obj, list) else [obj]:
            games.setdefault((g["seasonId"], g["seq"]), g)
    return [games[k] for k in sorted(games, key=lambda k: k[1])]


def _num(v) -> float:
    """Parse REBAS numeric strings; malformed manual entries (e.g. '12+') -> NaN."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _id_map(g: dict) -> dict[tuple[str, str], str]:
    m = {}
    for side, team in (("away", g["awayTeam"]), ("home", g["homeTeam"])):
        for box in ("BatterBox", "PitcherBox"):
            for p in g.get(f"{side}{box}", []) or []:
                m[(team, p["playerName"])] = p["playerId"]
    return m


def pitches(season: int) -> pd.DataFrame:
    rows = []
    for g in read_games(season):
        ids = _id_map(g)
        grows = []
        gid = f"CPBL-{season}-{g['seq']}"
        for side, bat_team, fld_team in (("away", g["awayTeam"], g["homeTeam"]),
                                         ("home", g["homeTeam"], g["awayTeam"])):
            first_pitcher = None
            for pa_i, pa in enumerate(g[f"{side}PAList"]):
                pitcher = pa.get("pitcherName")
                if first_pitcher is None and pitcher:
                    first_pitcher = pitcher
                result = pa.get("result") or ""
                evs = [e for e in pa.get("events", []) if e.get("type") == "PITCH"]
                balls = strikes = 0
                for k, e in enumerate(evs):
                    code = e.get("pitchCode") or ""
                    last = k == len(evs) - 1
                    velo = e.get("velocity")
                    cx, cy = e.get("coordX"), e.get("coordY")
                    grows.append({
                        "season": season, "game_id": gid, "date": g["date"][:10],
                        "pa_id": f"{gid}-{side}-{pa_i}", "pitch_seq": k + 1,
                        "pitcher_name": e.get("pitcherName") or pitcher, "pitcher_team": fld_team,
                        "batter_name": e.get("batterName") or pa.get("batterName"), "batter_team": bat_team,
                        "p_throws": pa.get("pitcherHand") or None, "stand": pa.get("batterHand") or None,
                        "pitcher_role": "SP" if pitcher == first_pitcher else "RP",
                        "balls": balls, "strikes": strikes,
                        "pitch_code": code, "pitch_type_raw": e.get("pitchType") or None,
                        "velo_kmh": _num(velo),
                        "coord_x": _num(cx),
                        "coord_y": _num(cy),
                        "pa_end": bool(last and result not in ("", "IGNORE", "IR", "ID")),
                        "pa_result_raw": result if last else None,
                        "hardness": (pa.get("hardness") or None) if last and code == "H" else None,
                        "trajectory": (pa.get("trajectory") or None) if last and code == "H" else None,
                    })
                    if code == "B":
                        balls += 1
                    elif code in ("S", "SW", "FT", "TRY_BUNT", "FOUL_BUNT") or (code == "F" and strikes < 2):
                        strikes = min(strikes + 1, 2)
        for r in grows:
            r["pitcher_id"] = ids.get((r["pitcher_team"], r["pitcher_name"]))
            r["batter_id"] = ids.get((r["batter_team"], r["batter_name"]))
        rows.extend(grows)
    df = pd.DataFrame(rows)
    df["league"] = "CPBL"
    # manual radar entries outside a physically plausible range are typos (e.g. 463, 12)
    df.loc[~df.velo_kmh.between(90, 170), "velo_kmh"] = np.nan
    df["velo_mph"] = df["velo_kmh"] / 1.609344
    df["is_bunt"] = df.pitch_code.isin(BUNT_CODES)
    df["swing"] = df.pitch_code.isin(SWING_CODES)
    df["whiff"] = df.pitch_code.eq("SW")
    df["called_strike"] = df.pitch_code.eq("S")
    df["pa_event"] = np.where(df.pa_end, df.pa_result_raw.map(RESULT_MAP).fillna("OTHER"), None)
    df["in_zone"] = in_zone(df.coord_x, df.coord_y)
    extra = df[["coord_x", "coord_y", "velo_kmh", "pitch_code", "hardness", "trajectory"]]
    out = finalize(df)
    return pd.concat([out, extra], axis=1)


# Zone: REBAS coordinates are centred on the strike-zone middle (pitcher's view);
# the schema doc defines zone width = 100 % and height = 120 % of width, i.e. an
# in-zone box of |x| <= 50, |y| <= 60 in REBAS units if 100 units = full width.
# This is verified empirically in cpbl_profile.zone_calibration() (called-strike
# rate on taken pitches) and the bounds below are the calibrated ones.
ZONE_HALF_X = 50.0
ZONE_HALF_Y = 60.0


def in_zone(x: pd.Series, y: pd.Series) -> pd.Series:
    z = (x.abs() <= ZONE_HALF_X) & (y.abs() <= ZONE_HALF_Y)
    return z.where(x.notna() & y.notna())


def load(seasons=(2024, 2025), force: bool = False) -> pd.DataFrame:
    out = processed("cpbl_pitches.parquet")
    if out.exists() and not force:
        return pd.read_parquet(out)
    df = pd.concat([pitches(s) for s in seasons], ignore_index=True)
    df.to_parquet(out, index=False)
    return df
