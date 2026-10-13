"""Amendment 4 — baseline weights estimated from data instead of judgment.

Question asked of the data: which mix of a pitcher's (hitter's) AAA baseline metrics in one
season best predicts how good he is in AAA the NEXT season?

  predictors  the baseline metrics exactly as src.baseline computes them, for 2025 AAA
              (league-adjusted, recent form = 2025-08-01..09-30 shrunk toward the season rate),
              z-scored within the estimation sample, clipped at +-winsor_z, sign-aligned
  sample      2025 AAA players passing the candidate workload filter (pitchers GS>=10 & IP>=60,
              hitters PA>=300), whatever their MLB status, who reach the target minimum in 2026
  target      2026 AAA, league-adjusted: pitchers -(FIP - league FIP), hitters wOBA - league wOBA
  fit         non-negative least squares on the centred data, each player weighted by his
              2026 BF / PA; coefficients rescaled to sum to 1 (the score is z-scored later, so
              only the proportions matter)
  uncertainty players resampled with replacement (bootstrap), percentile CI of each weight and
              the share of draws in which it is exactly 0
  check       10-fold cross-validated Spearman of the estimated-weight score with the target,
              next to the judgment-weight score (which is not fitted, so needs no folds)
  fallback    fewer than min_n players, or every coefficient 0 -> the judgment weights

Both seasons come from the official MLB Stats API season / date-range statistics
(statsapi.mlb.com, sportId 11 = AAA, regular season), so both sides of the regression use one
source. The 2026 candidates themselves keep the mirror season file used since the start.

Known bias (reported, not corrected): players promoted to MLB for most of 2026 drop out of the
target sample, so it under-represents the best 2025 players.
"""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from . import baseline, milb_candidates
from .common import config, processed, raw

API = "https://statsapi.mlb.com/api/v1/stats"
LEAGUES = {112: "PCL", 117: "INT"}
PITCHING = {"gamesPitched": "pitching_G", "gamesStarted": "pitching_GS", "battersFaced": "pitching_BF",
            "outs": "outs", "hits": "pitching_H", "homeRuns": "pitching_HR", "baseOnBalls": "pitching_BB",
            "intentionalWalks": "pitching_IBB", "strikeOuts": "pitching_SO", "hitByPitch": "pitching_HBP",
            "earnedRuns": "pitching_ER", "runs": "pitching_R"}
HITTING = {"gamesPlayed": "G", "plateAppearances": "batting_PA", "atBats": "batting_AB", "hits": "batting_H",
           "doubles": "batting_2B", "triples": "batting_3B", "homeRuns": "batting_HR", "baseOnBalls": "batting_BB",
           "intentionalWalks": "batting_IBB", "strikeOuts": "batting_SO", "hitByPitch": "batting_HBP",
           "sacFlies": "batting_SF", "sacBunts": "batting_SH", "totalBases": "batting_TB"}


def _est() -> dict:
    return config()["baseline"]["estimation"]


# ------------------------------------------------------------------ data
def to_frame(splits: list[dict], group: str) -> pd.DataFrame:
    """statsapi stat splits -> one row per player x AAA team, in the mirror's column names."""
    fields = PITCHING if group == "pitching" else HITTING
    rows = []
    for s in splits:
        lg = LEAGUES.get(s.get("league", {}).get("id"))
        if lg is None or "team" not in s:
            continue
        r = {"season": int(s["season"]), "player_id": s["player"]["id"], "player_full_name": s["player"]["fullName"],
             "team_abv": s["team"]["name"], "team_league": lg,
             "player_position": s.get("position", {}).get("abbreviation", "")}
        r.update({v: s["stat"].get(k, 0) for k, v in fields.items()})
        rows.append(r)
    df = pd.DataFrame(rows)
    if group == "pitching" and len(df):
        df["pitching_IP"] = df.outs / 3
    return df


def fetch(group: str, season: int, start: str | None = None, end: str | None = None,
          force: bool = False) -> pd.DataFrame:
    """Season (or date-range) AAA stats for every player, cached under data/raw/statsapi_season."""
    tag = f"{season}_{group}" + (f"_{start}_{end}" if start else "")
    out = raw("statsapi_season", f"{tag}.csv")
    if out.exists() and not force:
        return pd.read_csv(out)
    from .statsapi_pbp import _get
    params = {"stats": "byDateRange" if start else "season", "group": group, "sportId": 11, "season": season,
              "gameType": "R", "playerPool": "ALL", "limit": 10000}
    if start:
        params.update(startDate=start, endDate=end)
    d = _get(API, params)
    splits = d["stats"][0]["splits"]
    assert len(splits) == d["stats"][0]["totalSplits"], "statsapi returned a partial page"
    df = to_frame(splits, group)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def _pool(group: str, season: int) -> pd.DataFrame:
    s = fetch(group, season)
    if group == "pitching":
        df = milb_candidates._combine(s, [c for c in milb_candidates.PITCH_SUM if c in s] + ["outs"], "pitching_BF")
        df["IP"] = df.outs / 3
        return df
    s = s[s.player_position != "P"]
    return milb_candidates._combine(s, [c for c in milb_candidates.BAT_SUM if c in s], "batting_PA")


def sample(role: str) -> pd.DataFrame:
    """Estimation sample: 2025 predictors (as baseline computes them) + 2026 target."""
    e, f = _est(), config()["candidate_filters"][role]
    p_year, t_year = e["predictor_season"], e["target_season"]
    group = "pitching" if role == "pitcher" else "hitting"
    lo, hi = e["recent_window"]
    season, recent = fetch(group, p_year), fetch(group, p_year, lo, hi)
    pool = _pool(group, p_year)
    if role == "pitcher":
        pool = pool[(pool.pitching_GS >= f["min_gs"]) & (pool.IP >= f["min_ip"])]
        d = baseline.pitcher_metrics(pool, p_year, season, recent)
    else:
        pool = pool[pool.batting_PA >= f["min_pa"]]
        d = baseline.hitter_metrics(pool, p_year, season, recent)
    d = d.assign(n_predictor_pool=len(d))
    tgt = target(role, t_year)
    d = d.merge(tgt, on="player_id", how="left")
    d["target_n"] = d.target_n.fillna(0)
    d["in_sample"] = d.target_n >= e["target_min"][role]
    return d


def target(role: str, year: int) -> pd.DataFrame:
    """Next-season AAA quality, league-adjusted, sign so that higher = better."""
    group = "pitching" if role == "pitcher" else "hitting"
    s, pool = fetch(group, year), _pool(group, year)
    if role == "pitcher":
        lg = baseline._league_table("pitching", year, s)
        cfip = baseline._league_mix(pool, lg, "cfip")
        fip = (13 * pool.pitching_HR + 3 * (pool.pitching_BB + pool.pitching_HBP) - 2 * pool.pitching_SO) / pool.IP + cfip
        t = -(fip - baseline._league_mix(pool, lg, "fip"))
        n = pool.pitching_BF
    else:
        lg = baseline._league_table("batting", year, s)
        r = baseline._hitter_rates(pool.set_index("player_id")).reset_index()
        t = r.woba - baseline._league_mix(pool, lg, "woba")
        n = pool.batting_PA
    return pd.DataFrame({"player_id": pool.player_id, "target": t.values, "target_n": n.values})


# ------------------------------------------------------------------ estimation
def nnls_weights(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray | None:
    """Weighted NNLS on centred data; coefficients rescaled to sum to 1 (None if all are 0)."""
    w = np.asarray(w, float) / np.sum(w)
    Xc = X - w @ X
    yc = y - w @ y
    s = np.sqrt(w)
    coef, _ = nnls(Xc * s[:, None], yc * s)
    return coef / coef.sum() if coef.sum() > 0 else None


def _spearman(a, b) -> float:
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


def cv_spearman(X: np.ndarray, y: np.ndarray, w: np.ndarray, folds: int, rng) -> float:
    """Out-of-fold predictions from weights fitted on the other folds."""
    idx = rng.permutation(len(y))
    pred = np.empty(len(y))
    for k in range(folds):
        test = idx[k::folds]
        train = np.setdiff1d(idx, test)
        c = nnls_weights(X[train], y[train], w[train])
        pred[test] = X[test] @ (c if c is not None else np.ones(X.shape[1]) / X.shape[1])
    return _spearman(pred, y)


def estimate(role: str, d: pd.DataFrame | None = None) -> pd.DataFrame:
    e = _est()
    d = sample(role) if d is None else d
    s = d[d.in_sample].copy()
    judg = baseline.judgment_weights(role)
    names = list(judg)
    z = baseline.score(s, role, judg)
    X = z[[f"z_{m}" for m in names]].to_numpy(float)
    y = z.target.to_numpy(float)
    w = z.target_n.to_numpy(float)
    n = len(y)
    coef = nnls_weights(X, y, w) if n >= e["min_n"] else None
    rng = np.random.default_rng(e["bootstrap"]["seed"])
    draws = []
    for _ in range(e["bootstrap"]["draws"] if coef is not None else 0):
        i = rng.integers(0, n, n)
        c = nnls_weights(X[i], y[i], w[i])
        draws.append(c if c is not None else np.full(len(names), np.nan))
    B = np.array(draws) if draws else np.full((1, len(names)), np.nan)
    a = (1 - e["bootstrap"]["ci"]) / 2
    used = "estimated" if coef is not None else "judgment (fallback)"
    jw = np.array([judg[m] for m in names])
    cv = cv_spearman(X, y, w, e["cv_folds"], np.random.default_rng(e["bootstrap"]["seed"])) if coef is not None else np.nan
    return pd.DataFrame({
        "role": role, "metric": names, "judgment_weight": jw,
        "estimated_weight": coef if coef is not None else np.nan,
        "used_weight": coef if coef is not None else jw,
        "ci_lo": np.nanquantile(B, a, axis=0), "ci_hi": np.nanquantile(B, 1 - a, axis=0),
        "share_zero": np.nanmean(B == 0, axis=0) if draws else np.nan,
        "n": n, "n_predictor_pool": int(d.n_predictor_pool.iat[0]) if len(d) else 0,
        "n_without_target": int((~d.in_sample).sum()), "used": used,
        "cv_spearman_estimated": cv, "spearman_judgment": _spearman(X @ jw, y),
        "spearman_estimated_in_sample": _spearman(X @ coef, y) if coef is not None else np.nan,
    })


def run() -> pd.DataFrame:
    out = pd.concat([estimate(r) for r in ("pitcher", "hitter")], ignore_index=True)
    out.to_csv(processed("baseline_weights.csv"), index=False)
    _table.cache_clear()
    weights.cache_clear()
    return out


@lru_cache(maxsize=1)
def _table() -> pd.DataFrame:
    f = processed("baseline_weights.csv")
    return pd.read_csv(f) if f.exists() else run()


@lru_cache(maxsize=None)
def weights(role: str) -> dict:
    t = _table()
    t = t[t.role == role]
    return dict(zip(t.metric, t.used_weight.astype(float)))



if __name__ == "__main__":
    t = run()
    pd.set_option("display.width", 200)
    print(t.round(3).to_string())
    print(json.dumps({r: weights(r) for r in ("pitcher", "hitter")}, indent=1))
