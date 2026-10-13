"""Phase 2 — AAA candidate pools with an auditable screening log.

Funnel (every AAA player in the 2026 season file enters; each exit is logged):
  1. 2026 AAA regular-season workload filter   (pitchers: GS>=10 & IP>=60; hitters: PA>=300)
  2. MLB eligibility (eligibility.check)        -> 'ineligible' and 'unknown' both exit
  3. tracked-pitch availability (AAA pitch data in samples.player_seasons) -> does NOT
     exit; Fit is reported as NA ("insufficient tracking data") but the baseline still ranks them.

Recruitment feasibility: factual MLB roster status from statsapi (src.roster_status,
amendment 1) — 40-man membership, 2026 MLB playing time, latest transaction. Contract
terms, salary and willingness are still NOT assessed. Never mixed into ability scores.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import eligibility, milb_loader, roster_status
from .common import config, processed, snapshot

PITCH_SUM = ["pitching_G", "pitching_GS", "pitching_BF", "pitching_IP", "pitching_H", "pitching_HR",
             "pitching_BB", "pitching_IBB", "pitching_SO", "pitching_HBP", "pitching_ER", "pitching_R",
             "pitching_PI", "pitching_total_swings", "pitching_swing_and_misses"]
BAT_SUM = ["G", "batting_PA", "batting_AB", "batting_H", "batting_2B", "batting_3B", "batting_HR",
           "batting_BB", "batting_IBB", "batting_SO", "batting_HBP", "batting_SF", "batting_SH", "batting_TB",
           "batting_swings", "batting_whiffs", "batting_pitches_faced"]


def _combine(df: pd.DataFrame, cols: list[str], weight: str) -> pd.DataFrame:
    """Sum counting stats over a player's AAA team stints; keep weighted league mix."""
    df = df.copy()
    agg = df.groupby("player_id")[cols].sum()
    info = df.sort_values(weight, ascending=False).groupby("player_id").agg(
        name=("player_full_name", "first"), position=("player_position", "first"),
        teams=("team_abv", lambda s: "/".join(sorted(set(s)))),
        leagues=("team_league", lambda s: "/".join(sorted(set(s)))))
    w = df.pivot_table(index="player_id", columns="team_league", values=weight, aggfunc="sum").fillna(0)
    w = w.div(w.sum(axis=1), axis=0).add_prefix("share_")
    return agg.join(info).join(w).reset_index()


def pitcher_pool() -> pd.DataFrame:
    year = config()["candidate_filters"]["pitcher"]["season"]
    raw = milb_loader.season_stats("pitching", year)
    df = _combine(raw, PITCH_SUM, "pitching_BF")
    df["outs"] = (df.pitching_IP * 3).round().astype(int)
    df["IP"] = df.outs / 3
    return df


def hitter_pool() -> pd.DataFrame:
    year = config()["candidate_filters"]["hitter"]["season"]
    raw = milb_loader.season_stats("batting", year)
    raw = raw[raw.player_position != "P"]
    return _combine(raw, BAT_SUM, "batting_PA")


def tracked_counts(aaa: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    aaa = aaa[aaa.season.isin(config()["samples"]["player_seasons"])]
    p = aaa.groupby("pitcher_id").size()
    b = aaa[aaa.pa_end].groupby("batter_id").size()
    return p, b


def screen(role: str, aaa: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (screening_log for all players, candidate table for survivors)."""
    f = config()["candidate_filters"][role]
    pool = pitcher_pool() if role == "pitcher" else hitter_pool()
    if role == "pitcher":
        pool["pass_workload"] = (pool.pitching_GS >= f["min_gs"]) & (pool.IP >= f["min_ip"])
        wl = "GS>={} & IP>={}".format(f["min_gs"], f["min_ip"])
    else:
        pool["pass_workload"] = pool.batting_PA >= f["min_pa"]
        wl = "PA>={}".format(f["min_pa"])
    pool["mlbam_id"] = pool.player_id.astype(int)
    stage = pool[pool.pass_workload]
    el = eligibility.check(stage[["mlbam_id", "name"]], role)
    pool = pool.merge(el.drop(columns=["name"]), on="mlbam_id", how="left")
    off = eligibility.official_crosscheck(stage[["mlbam_id"]], role)          # amendment 1
    pool = pool.merge(off, on="mlbam_id", how="left")
    pool["official_agrees"] = np.where(pool.pass_workload, pool.official_eligibility == pool.eligibility_status, np.nan)
    if aaa is not None:
        pc, bc = tracked_counts(aaa)
        key = pool.mlbam_id.astype(str)
        pool["tracked_pitches"] = key.map(pc).fillna(0).astype(int) if role == "pitcher" else np.nan
        pool["tracked_pa"] = key.map(bc).fillna(0).astype(int) if role == "hitter" else np.nan
    reason = np.where(~pool.pass_workload, f"excluded: 2026 AAA workload below {wl}",
                      np.where(pool.eligibility_status == "ineligible", "excluded: MLB playing time over limit",
                               np.where(pool.eligibility_status == "unknown", "excluded: MLB playing time unconfirmed",
                                        "candidate")))
    pool["screening_result"] = reason
    by = pool.birth_year.astype(float)
    pool["age_2026"] = snapshot().year - by
    pool["data_year_baseline"] = config()["candidate_filters"][role]["season"]
    log = pool.copy()
    cand = pool[pool.screening_result == "candidate"].copy()
    return log, cand


REC_COLS = ["on_40man", "roster_club", "roster_status", "mlb_2026_g", "mlb_2026_pa", "mlb_2026_ip",
            "last_transaction_date", "last_transaction", "recruitment_feasibility"]


def recruitment(ids) -> pd.DataFrame:
    """Cached roster facts; rebuilt from statsapi when a candidate is missing from the cache."""
    ids = set(int(i) for i in ids)
    rs = roster_status.load() if processed("recruitment_status.csv").exists() else None
    if rs is None or not ids <= set(rs.mlbam_id):
        rs = roster_status.load(sorted(ids), force=True)
    return rs[["mlbam_id", *REC_COLS]]


def run(aaa: pd.DataFrame | None = None) -> dict:
    out = {}
    screened = {role: screen(role, aaa) for role in ("pitcher", "hitter")}
    rec = recruitment(pd.concat([c.mlbam_id for _, c in screened.values()]))
    for role in ("pitcher", "hitter"):
        log, cand = screened[role]
        cand = cand.merge(rec, on="mlbam_id", how="left")
        cols = ["mlbam_id", "name", "position", "teams", "leagues", "age_2026", "pass_workload",
                "eligibility_status", "eligibility_reason", "evidence", "retro_id",
                *[c for c in log.columns if c.startswith("mlb_")],
                *[c for c in log.columns if c.startswith("official_")], "screening_result"]
        log = log.merge(rec, on="mlbam_id", how="left")
        cols += REC_COLS
        log[[c for c in cols if c in log.columns]].to_csv(processed(f"screening_log_{role}s.csv"), index=False)
        out[role] = (log, cand)
    return out


if __name__ == "__main__":
    r = run(milb_loader.load())
    for role, (log, cand) in r.items():
        print(role, log.screening_result.value_counts().to_dict(), "candidates:", len(cand))
