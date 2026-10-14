"""Phase 3 — traditional AAA baseline ranking (weights pre-registered in config).

Pitchers (2026 AAA regular season, all AAA stints combined):
  K-BB%        = (SO - BB) / BF                       league-adjusted
  FIP          = (13 HR + 3 (BB + HBP) - 2 SO) / IP + cFIP(league)   (cFIP makes lg FIP == lg ERA)
                 expressed as FIP - league FIP (league-adjusted)
  recent K-BB% = same, game logs 2026-08-01..09-30, shrunk toward own season rate (k BF)
  IP/GS, GS    = starter workload
Hitters:
  wOBA (generic weights, denominator AB+BB-IBB+SF+HBP), ISO, BB% (BB/PA), K% (SO/PA) — league-adjusted
  recent wOBA  = game logs 2026-08-01..09-30, shrunk toward own season wOBA (k PA)

Scoring: z-score within the eligible candidate pool, clipped to +-winsor_z, sign-aligned,
weighted sum. Higher = better. Weights: amendment 4 estimates them from 2025 -> 2026 AAA data
(src.baseline_weights); the original judgment weights stay in config as the fallback and are
reported next to it (baseline_score_judgment / baseline_rank_judgment).
xwOBA / HardHit% / Barrel% are not in the hitter baseline: the weights were pre-registered
when 2026 AAA pitch data was unavailable, and adding stats after seeing rankings is not allowed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import milb_candidates, milb_loader
from .common import config, processed, zscore


def _league_table(kind: str, year: int, s: pd.DataFrame | None = None) -> pd.DataFrame:
    """League (PCL / INT) totals and rates; `s` overrides the mirror season file (amendment 4)."""
    s = milb_loader.candidate_stats(kind) if s is None else s      # amendment 6: two seasons
    if kind == "pitching":
        g = s.groupby("team_league")[["pitching_BF", "pitching_SO", "pitching_BB", "pitching_HBP", "pitching_HR",
                                      "pitching_ER", "pitching_IP"]].sum()
        g["k_minus_bb_pct"] = (g.pitching_SO - g.pitching_BB) / g.pitching_BF
        g["k_pct"] = g.pitching_SO / g.pitching_BF
        g["bb_pct"] = g.pitching_BB / g.pitching_BF
        g["era"] = 9 * g.pitching_ER / g.pitching_IP
        raw = (13 * g.pitching_HR + 3 * (g.pitching_BB + g.pitching_HBP) - 2 * g.pitching_SO) / g.pitching_IP
        g["cfip"] = g.era - raw
        g["fip"] = g.era
        return g
    g = s.groupby("team_league")[["batting_PA", "batting_AB", "batting_H", "batting_2B", "batting_3B", "batting_HR",
                                  "batting_BB", "batting_IBB", "batting_SO", "batting_HBP", "batting_SF",
                                  "batting_TB"]].sum()
    return _hitter_rates(g)


def _hitter_rates(g: pd.DataFrame) -> pd.DataFrame:
    w = config()["woba_weights"]
    single = g.batting_H - g.batting_2B - g.batting_3B - g.batting_HR
    ubb = g.batting_BB - g.batting_IBB
    num = (w["uBB"] * ubb + w["HBP"] * g.batting_HBP + w["1B"] * single + w["2B"] * g.batting_2B
           + w["3B"] * g.batting_3B + w["HR"] * g.batting_HR)
    den = g.batting_AB + ubb + g.batting_SF + g.batting_HBP
    g = g.copy()
    g["woba_num"], g["woba_den"] = num, den
    g["woba"] = num / den
    g["iso"] = (g.batting_TB - g.batting_H) / g.batting_AB
    g["bb_pct"] = g.batting_BB / g.batting_PA
    g["k_pct"] = g.batting_SO / g.batting_PA
    return g


def _league_mix(df: pd.DataFrame, lg: pd.DataFrame, col: str) -> pd.Series:
    shares = [c for c in df.columns if c.startswith("share_")]
    val = 0
    for c in shares:
        val = val + df[c] * lg.loc[c.replace("share_", ""), col]
    return val


def _recent(year: int, role: str, g: pd.DataFrame | None = None) -> pd.DataFrame:
    """Recent-window counting stats per player; `g` = rows already limited to the window
    (amendment 4 passes statsapi date-range stats), else the 2026 game logs are filtered."""
    if g is None:
        lo, hi = config()["baseline"]["recent_window"]
        g = milb_loader.game_logs(year)
        g = g[(g.game_date >= lo) & (g.game_date <= hi)]
    if role == "pitcher":
        r = g.groupby("player_id")[["pitching_BF", "pitching_SO", "pitching_BB"]].sum()
        return r.rename(columns=lambda c: "recent_" + c)
    r = g.groupby("player_id")[["batting_PA", "batting_AB", "batting_H", "batting_2B", "batting_3B", "batting_HR",
                                "batting_BB", "batting_IBB", "batting_SO", "batting_HBP", "batting_SF",
                                "batting_TB"]].sum()
    r = _hitter_rates(r)
    return r[["batting_PA", "woba_num", "woba_den"]].rename(columns=lambda c: "recent_" + c)


def pitcher_metrics(cand: pd.DataFrame, year: int | None = None, season: pd.DataFrame | None = None,
                    recent: pd.DataFrame | None = None) -> pd.DataFrame:
    """Baseline metrics. Defaults: the 2026 candidates from the mirror files; amendment 4
    passes another season's statsapi tables to build the weight-estimation sample."""
    year = year or config()["candidate_filters"]["pitcher"]["season"]
    lg = _league_table("pitching", year, season)
    k = config()["samples"]["recent_shrink_bf"]
    d = cand.copy()
    d["k_pct"] = d.pitching_SO / d.pitching_BF
    d["bb_pct"] = d.pitching_BB / d.pitching_BF
    d["k_minus_bb_pct_raw"] = d.k_pct - d.bb_pct
    d["era"] = 9 * d.pitching_ER / d.IP
    d["whip"] = (d.pitching_H + d.pitching_BB) / d.IP
    cfip = _league_mix(d, lg, "cfip")
    d["fip_raw"] = (13 * d.pitching_HR + 3 * (d.pitching_BB + d.pitching_HBP) - 2 * d.pitching_SO) / d.IP + cfip
    d["lg_k_minus_bb"] = _league_mix(d, lg, "k_minus_bb_pct")
    d["lg_fip"] = _league_mix(d, lg, "fip")
    adj = config().get("league_adjust", True)
    d["k_minus_bb_pct"] = d.k_minus_bb_pct_raw - (d.lg_k_minus_bb if adj else 0)
    d["lg_k_pct"] = _league_mix(d, lg, "k_pct")
    d["lg_bb_pct"] = _league_mix(d, lg, "bb_pct")
    d["k_pct_adj"] = d.k_pct - (d.lg_k_pct if adj else 0)           # amendment 5
    d["bb_pct_adj"] = d.bb_pct - (d.lg_bb_pct if adj else 0)
    d["fip"] = d.fip_raw - (d.lg_fip if adj else 0)
    d["ip_per_gs"] = d.IP / d.pitching_GS
    d["gs"] = d.pitching_GS
    d["swstr_pct_aaa"] = d.pitching_swing_and_misses / d.pitching_PI if "pitching_PI" in d else np.nan
    rec = _recent(year, "pitcher", recent)
    d = d.join(rec, on="player_id")
    for c in rec.columns:
        d[c] = d[c].fillna(0)
    own = d.k_minus_bb_pct_raw
    d["recent_k_minus_bb_pct_raw"] = ((d.recent_pitching_SO - d.recent_pitching_BB) + own * k) / (d.recent_pitching_BF + k)
    d["recent_k_minus_bb_pct"] = d.recent_k_minus_bb_pct_raw - (d.lg_k_minus_bb if adj else 0)
    return d


def hitter_metrics(cand: pd.DataFrame, year: int | None = None, season: pd.DataFrame | None = None,
                   recent: pd.DataFrame | None = None) -> pd.DataFrame:
    year = year or config()["candidate_filters"]["hitter"]["season"]
    lg = _league_table("batting", year, season)
    k = config()["samples"]["recent_shrink_bf"]
    d = _hitter_rates(cand.set_index("player_id")).reset_index()
    for c in ("woba", "iso", "bb_pct", "k_pct"):
        d[f"{c}_raw"] = d[c]
        d[f"lg_{c}"] = _league_mix(d, lg, c)
        if config().get("league_adjust", True):
            d[c] = d[f"{c}_raw"] - d[f"lg_{c}"]
    d["obp"] = (d.batting_H + d.batting_BB + d.batting_HBP) / (d.batting_AB + d.batting_BB + d.batting_HBP + d.batting_SF)
    d["slg"] = d.batting_TB / d.batting_AB
    d["ops"] = d.obp + d.slg
    d["whiff_pct_aaa"] = d.batting_whiffs / d.batting_swings if "batting_swings" in d else np.nan
    rec = _recent(year, "hitter", recent)
    d = d.join(rec, on="player_id")
    for c in rec.columns:
        d[c] = d[c].fillna(0)
    d["recent_woba_raw"] = (d.recent_woba_num + d.woba_raw * k) / (d.recent_woba_den + k)
    d["recent_woba"] = d.recent_woba_raw - (d.lg_woba if config().get("league_adjust", True) else 0)
    return d


def judgment_weights(role: str) -> dict:
    """The weights pre-registered by judgment before any ranking (kept as amendment 4's fallback)."""
    return {m: s["weight"] for m, s in config()["baseline"][role].items()}


def active_weights(role: str) -> dict:
    """Weights the baseline uses. Amendment 4: estimated from 2025 -> 2026 AAA data
    (src.baseline_weights) unless the estimate fell back to the judgment weights."""
    src = config()["baseline"].get("weights_source", "judgment")
    src = src.get(role, "judgment") if isinstance(src, dict) else src      # amendment 5: per role
    if src != "estimated":
        return judgment_weights(role)
    from . import baseline_weights
    return baseline_weights.weights(role)


def score(d: pd.DataFrame, role: str, weights: dict | None = None) -> pd.DataFrame:
    """Weighted sum of sign-aligned, clipped z-scores. `weights` defaults to active_weights(role)."""
    spec = config()["baseline"][role]
    clip = config()["baseline"]["winsor_z"]
    weights = weights or active_weights(role)
    d = d.copy()
    total = 0
    for metric, s in spec.items():
        w = weights.get(metric, s["weight"])
        z = s["sign"] * zscore(d[metric], clip)
        d[f"z_{metric}"] = z
        total = total + w * z
    d["baseline_score"] = total
    d["baseline_rank"] = d.baseline_score.rank(ascending=False, method="first").astype(int)
    return d.sort_values("baseline_rank")


def _with_judgment(d: pd.DataFrame, role: str) -> pd.DataFrame:
    """Adds the judgment-weight score and rank next to the active one (amendment 4 sensitivity row)."""
    j = score(d, role, judgment_weights(role))
    d["baseline_score_judgment"] = j.baseline_score.reindex(d.index)
    d["baseline_rank_judgment"] = j.baseline_rank.reindex(d.index)
    return d


def run(cands: dict) -> dict:
    out = {}
    p = _with_judgment(score(pitcher_metrics(cands["pitcher"]), "pitcher"), "pitcher")
    h = _with_judgment(score(hitter_metrics(cands["hitter"]), "hitter"), "hitter")
    pcols = ["baseline_rank", "mlbam_id", "name", "teams", "leagues", "age_2026", "pitching_GS", "IP",
             "pitching_BF", "k_pct", "bb_pct", "k_minus_bb_pct_raw", "k_minus_bb_pct", "era", "fip_raw", "fip",
             "whip", "swstr_pct_aaa", "recent_pitching_BF", "recent_k_minus_bb_pct_raw", "ip_per_gs",
             *[c for c in p.columns if c.startswith("z_")], "baseline_score", "baseline_score_judgment",
             "baseline_rank_judgment", "mlb_ip_2024", "mlb_ip_2025",
             "eligibility_status", "tracked_pitches", *milb_candidates.REC_COLS,
             *milb_candidates.SIGN_COLS]
    hcols = ["baseline_rank", "mlbam_id", "name", "position", "teams", "leagues", "age_2026", "batting_PA",
             "woba_raw", "woba", "obp", "slg", "ops", "iso_raw", "bb_pct_raw", "k_pct_raw", "whiff_pct_aaa",
             "recent_batting_PA", "recent_woba_raw", *[c for c in h.columns if c.startswith("z_")],
             "baseline_score", "baseline_score_judgment", "baseline_rank_judgment", "mlb_pa_2024", "mlb_pa_2025", "eligibility_status", "tracked_pa",
             *milb_candidates.REC_COLS, *milb_candidates.SIGN_COLS]
    p[[c for c in pcols if c in p.columns]].to_csv(processed("baseline_pitchers.csv"), index=False)
    h[[c for c in hcols if c in h.columns]].to_csv(processed("baseline_hitters.csv"), index=False)
    out["pitcher"], out["hitter"] = p, h
    return out


if __name__ == "__main__":
    aaa = milb_loader.load()
    c = milb_candidates.run(aaa)
    r = run({k: v[1] for k, v in c.items()})
    pd.set_option("display.width", 250)
    print(r["pitcher"][["baseline_rank", "name", "age_2026", "pitching_GS", "IP", "k_minus_bb_pct_raw", "fip_raw",
                        "era", "recent_k_minus_bb_pct_raw", "baseline_score", "tracked_pitches"]].head(15).round(3).to_string())
    print(r["hitter"][["baseline_rank", "name", "position", "age_2026", "batting_PA", "woba_raw", "iso_raw", "bb_pct_raw",
                       "k_pct_raw", "recent_woba_raw", "baseline_score", "tracked_pa", "leagues"]].head(15).round(3).to_string())
