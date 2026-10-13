"""MLB playing-time eligibility for the 2026 Taiwan baseball analytics contest.

Rules (from the brief):
  * hitters : 2024 MLB PA <= 200  AND  2025 MLB PA <= 200   (each season separately)
  * pitchers: 2024 MLB IP <= 40   AND  2025 MLB IP <= 40    (IP counted in outs: <= 120)
  * exactly 200 PA / 40.0 IP is eligible
  * MLB regular season only (no MiLB, spring training, postseason)
  * a season with no record that cannot be confirmed as zero => NOT eligible ("unknown")

Data: Retrosheet regular-season event files (all 2,429/2,430 games of 2024/2025)
-> chadwick `cwdaily` player-game lines -> season totals; MLBAM ids are linked
through the Chadwick register. Retrosheet is used because statsapi.mlb.com was
unreachable from the analysis environment; it is the standard reference
compilation of MLB play-by-play and is cross-checked against game counts and
published season totals in tests.
"""
from __future__ import annotations

import glob
import io
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from .common import config, ip_to_outs, outs_to_ip_str, path, processed

CWDAILY_FIELDS = "0,1,4,5,11,12,36,37,44,45"
CWDAILY_NAMES = ["game_id", "date", "team", "retro_id", "b_g", "pa", "p_g", "p_gs", "p_outs", "p_bf"]


def _ext() -> Path:
    return path("external")


def regular_season_files(season: int) -> list[str]:
    """Regular-season event files only: <yyyy><TEAM>.EVA / .EVN.

    Postseason (.EVE: WS, LCS, DS, WC) and All-Star files are excluded, as are
    the .EB* box-score duplicates.
    """
    d = _ext() / "retrosheet" / "seasons" / str(season)
    files = sorted(glob.glob(str(d / f"{season}???.EVA")) + glob.glob(str(d / f"{season}???.EVN")))
    if not files:
        raise FileNotFoundError(f"No Retrosheet event files in {d}; run python -m src.fetch_data --only retro")
    return files


def run_cwdaily(season: int) -> pd.DataFrame:
    bin_dir = _ext() / "cw" / "install" / "bin"
    env = dict(os.environ, LD_LIBRARY_PATH=str(_ext() / "cw" / "install" / "lib"))
    files = regular_season_files(season)
    d = Path(files[0]).parent
    out = subprocess.run(
        [str(bin_dir / "cwdaily"), "-q", "-y", str(season), "-f", CWDAILY_FIELDS, *[Path(f).name for f in files]],
        cwd=d, env=env, capture_output=True, text=True, check=True,
    ).stdout
    df = pd.read_csv(io.StringIO(out), header=None, names=CWDAILY_NAMES)
    df["season"] = season
    return df


def build_mlb_season_lines(force: bool = False) -> pd.DataFrame:
    """Per (season, retro_id) MLB regular-season totals. Cached to processed/."""
    out = processed("mlb_season_lines.csv")
    if out.exists() and not force:
        return pd.read_csv(out)
    frames, audit = [], []
    for season in config()["eligibility"]["seasons"]:
        g = run_cwdaily(season)
        audit.append({"season": season, "games": g.game_id.nunique(), "player_game_rows": len(g),
                      "total_pa": int(g.pa.sum()), "total_outs": int(g.p_outs.sum())})
        frames.append(g)
    lines = pd.concat(frames)
    agg = (lines.groupby(["season", "retro_id"])
           .agg(games=("b_g", "sum"), pa=("pa", "sum"), p_games=("p_g", "sum"),
                p_gs=("p_gs", "sum"), p_outs=("p_outs", "sum"), p_bf=("p_bf", "sum"))
           .reset_index())
    agg.to_csv(out, index=False)
    pd.DataFrame(audit).to_csv(processed("mlb_season_audit.csv"), index=False)
    return agg


def load_register() -> pd.DataFrame:
    files = sorted(glob.glob(str(_ext() / "register" / "data" / "people-*.csv")))
    if not files:
        raise FileNotFoundError("Chadwick register missing; run python -m src.fetch_data --only retro")
    cols = ["key_mlbam", "key_retro", "key_bbref", "name_first", "name_last", "birth_year",
            "birth_month", "birth_day", "mlb_played_first", "mlb_played_last"]
    reg = pd.concat(pd.read_csv(f, usecols=cols, dtype=str, low_memory=False) for f in files)
    reg = reg[reg.key_mlbam.notna()].copy()
    reg["key_mlbam"] = reg.key_mlbam.astype(float).astype("int64")
    return reg.drop_duplicates("key_mlbam")


def roster_names(season: int) -> set[str]:
    """Lower-cased 'first last' of everyone on a 2024/2025 MLB Retrosheet roster."""
    d = _ext() / "retrosheet" / "seasons" / str(season)
    names = set()
    for f in glob.glob(str(d / "*.ROS")):
        if "ALS" in Path(f).name or "NLS" in Path(f).name:   # All-Star rosters
            continue
        for row in open(f, encoding="latin-1"):
            parts = row.strip().split(",")
            if len(parts) >= 3:
                names.add(f"{parts[2]} {parts[1]}".lower())
    return names


def decide(role: str, values: dict[int, float | None]) -> tuple[bool | None, str]:
    """Pure rule. values: season -> PA (hitter) or outs (pitcher); None = unconfirmed."""
    cfg = config()["eligibility"]
    limit = cfg["hitter_max_pa"] if role == "hitter" else cfg["pitcher_max_ip"] * 3
    unit = "PA" if role == "hitter" else "outs"
    reasons = []
    unknown = False
    for season in cfg["seasons"]:
        v = values.get(season)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            unknown = True
            reasons.append(f"{season}: unconfirmed")
        elif v > limit:
            reasons.append(f"{season}: {v:g} {unit} > {limit}")
            return False, "; ".join(reasons)
        else:
            reasons.append(f"{season}: {v:g} {unit} <= {limit}")
    if unknown:
        return None, "; ".join(reasons)
    return True, "; ".join(reasons)


def check(players: pd.DataFrame, role: str) -> pd.DataFrame:
    """players: DataFrame with columns mlbam_id, name. Returns one row per player."""
    seasons = config()["eligibility"]["seasons"]
    lines = build_mlb_season_lines()
    reg = load_register().set_index("key_mlbam")
    rosters = {s: roster_names(s) for s in seasons}
    rows = []
    for _, p in players.iterrows():
        pid = int(p.mlbam_id)
        r = {"mlbam_id": pid, "name": p["name"], "role": role}
        if pid not in reg.index:
            r.update(retro_id=None, evidence="MLBAM id not in Chadwick register")
            vals = {s: None for s in seasons}
        else:
            rr = reg.loc[pid]
            retro = rr.key_retro if isinstance(rr.key_retro, str) else None
            r.update(retro_id=retro, birth_year=rr.birth_year,
                     mlb_played_first=rr.mlb_played_first, mlb_played_last=rr.mlb_played_last)
            vals = {}
            if retro:
                pl = lines[lines.retro_id == retro].set_index("season")
                for s in seasons:
                    if s in pl.index:
                        vals[s] = float(pl.loc[s, "pa"] if role == "hitter" else pl.loc[s, "p_outs"])
                    else:
                        vals[s] = 0.0   # complete-season event files: absence == no appearance
                r["evidence"] = "Retrosheet regular-season event files (complete seasons) via retro id"
            else:
                last = rr.mlb_played_last
                full = f"{rr.name_first} {rr.name_last}".lower()
                name_hit = [s for s in seasons if full in rosters[s]]
                if (not isinstance(last, str) or int(float(last)) < min(seasons)) and not name_hit:
                    vals = {s: 0.0 for s in seasons}
                    r["evidence"] = ("No Retrosheet id; register shows no MLB play in 2024-25 "
                                     "and name absent from all 2024/2025 MLB rosters")
                else:
                    vals = {s: None for s in seasons}
                    r["evidence"] = f"No Retrosheet id but register/roster hint of MLB play ({last}; roster {name_hit})"
        for s in seasons:
            v = vals.get(s)
            if role == "hitter":
                r[f"mlb_pa_{s}"] = v
            else:
                r[f"mlb_outs_{s}"] = v
                r[f"mlb_ip_{s}"] = None if v is None else outs_to_ip_str(int(v))
        ok, reason = decide(role, vals)
        r["eligible"] = ok
        r["eligibility_status"] = {True: "eligible", False: "ineligible", None: "unknown"}[ok]
        r["eligibility_reason"] = reason
        rows.append(r)
    return pd.DataFrame(rows)


def ip_string_outs(ip) -> int:
    """Public helper kept for tests: '39.2' -> 119 outs."""
    return ip_to_outs(ip)


if __name__ == "__main__":
    agg = build_mlb_season_lines(force=True)
    print(pd.read_csv(processed("mlb_season_audit.csv")))
    print(agg.sort_values("pa", ascending=False).head())
