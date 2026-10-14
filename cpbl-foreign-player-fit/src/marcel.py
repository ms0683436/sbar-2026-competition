"""Amendment 15 — "AAA Marcel" projections (rules in config/scoring.yaml, amendment15).

Tango's Marcel adapted to AAA: the last three AAA seasons weighted (hitters 5/4/3, pitchers 3/2/1),
each team stint's deviation from its season x league (PCL / INT) average added up, regressed toward
the pooled AAA average of the latest season by R weighted PA / BF of league-average play, then an
age factor. Only AAA lines are used (MLB, foreign and independent seasons add nothing).

  pitchers   SO, BB, HR, HBP per BF and IP per BF; score = projected K% - BB%; projected FIP shown
  hitters    wOBA (config woba_weights), ISO, K%, BB%; score = projected wOBA

Park: each stint is park-adjusted with that season's own single-year factors (2023-2026, from the
AAA pitch data); earlier seasons use PF = 1. The census validation (V2) uses PF = 1 throughout.

Usage:  python -m src.marcel      (validation V1-V3, reports/marcel.md)
"""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd

from . import backtest, baseline, baseline_weights, eligibility, milb_candidates, park
from .common import config, md_table, path, processed

PITCH_COUNTS = ["pitching_BF", "pitching_SO", "pitching_BB", "pitching_HR", "pitching_HBP", "pitching_IP",
                "pitching_ER", "pitching_GS"]
BAT_COUNTS = ["batting_PA", "batting_AB", "batting_H", "batting_2B", "batting_3B", "batting_HR", "batting_BB",
              "batting_IBB", "batting_SO", "batting_HBP", "batting_SF", "batting_TB"]
PF_SEASONS = (2023, 2024, 2025, 2026)          # seasons with AAA pitch data for single-year factors


def cfg() -> dict:
    return config()["amendment15"]


# ------------------------------------------------------------------ inputs
@lru_cache(maxsize=None)
def pf_for(season: int, neutral: bool = False) -> pd.DataFrame:
    """Single-season park factors, or an empty table (= PF 1) when unavailable or `neutral`."""
    if neutral or season not in PF_SEASONS:
        return pd.DataFrame(columns=park.EVENTS)
    from . import recruitment_backtest
    return park.factors_for([season], recruitment_backtest._pitches([season]))


@lru_cache(maxsize=None)
def stints(group: str, season: int, neutral: bool = False) -> pd.DataFrame:
    """Park-adjusted AAA team stints of one season (hitters: non-pitchers only)."""
    if season == 2020:
        return pd.DataFrame()
    s = baseline_weights.fetch(group, season, pf=pf_for(season, neutral))
    if group == "hitting":
        s = s[s.player_position != "P"]
    if group == "pitching":
        s = s.assign(pitching_IP=s.outs / 3)
    return s.assign(season=season)


@lru_cache(maxsize=1)
def birth_years() -> pd.Series:
    r = eligibility.load_register()
    return pd.to_numeric(r.set_index("key_mlbam").birth_year, errors="coerce")


def age_factor(a: pd.Series) -> pd.Series:
    g = cfg()["age"]
    f = pd.Series(1.0, index=a.index)
    young, old = a < g["peak"], a > g["peak"]
    f[young] = 1 + g["young"] * (g["peak"] - a[young])
    f[old] = 1 / (1 + g["old"] * (a[old] - g["peak"]))
    return f.fillna(1.0)


# ------------------------------------------------------------------ projection
def _stats(role: str) -> dict:
    """statistic -> (numerator function, denominator column, age direction)."""
    w = config()["woba_weights"]
    if role == "pitcher":
        return {"k": (lambda d: d.pitching_SO, "pitching_BF", 1), "bb": (lambda d: d.pitching_BB, "pitching_BF", -1),
                "hr": (lambda d: d.pitching_HR, "pitching_BF", -1), "hbp": (lambda d: d.pitching_HBP, "pitching_BF", 0),
                "ipbf": (lambda d: d.pitching_IP, "pitching_BF", 0)}
    def woba(d):
        single = d.batting_H - d.batting_2B - d.batting_3B - d.batting_HR
        return (w["uBB"] * (d.batting_BB - d.batting_IBB) + w["HBP"] * d.batting_HBP + w["1B"] * single
                + w["2B"] * d.batting_2B + w["3B"] * d.batting_3B + w["HR"] * d.batting_HR)
    return {"woba": (woba, "_woba_den", 1), "iso": (lambda d: d.batting_TB - d.batting_H, "batting_AB", 1),
            "k": (lambda d: d.batting_SO, "batting_PA", -1), "bb": (lambda d: d.batting_BB, "batting_PA", 1)}


def _prep(s: pd.DataFrame, role: str) -> pd.DataFrame:
    s = s.copy()
    if role == "hitter":
        s["_woba_den"] = s.batting_AB + s.batting_BB - s.batting_IBB + s.batting_SF + s.batting_HBP
    return s


def project(role: str, year: int, mult: float = 1.0, age: bool = True, neutral: bool = False,
            seasons: list[int] | None = None, scale: pd.DataFrame | None = None) -> pd.DataFrame:
    """Marcel projection for season `year` from AAA seasons year-1..year-3 (one row per player).
    `scale` (bootstrap): multipliers per (player_id, season, team_id) for the counting columns of
    those stints; league averages stay those of the unscaled data."""
    c = cfg()["pitcher" if role == "pitcher" else "hitter"]
    group = "pitching" if role == "pitcher" else "hitting"
    expo = "pitching_BF" if role == "pitcher" else "batting_PA"
    R = (c["regress_bf"] if role == "pitcher" else c["regress_pa"]) * mult
    seasons = seasons or [year - 1, year - 2, year - 3]
    stats = _stats(role)
    frames = []
    for wt, y in zip(c["weights"], seasons):
        s = stints(group, y, neutral)
        if s.empty:
            continue
        lg = _prep(s, role).groupby("team_league")
        if scale is not None:
            s = _scaled(s, scale)
        s = _prep(s, role)
        rows = pd.DataFrame({"player_id": s.player_id, "season": y, "w": wt, expo: s[expo]})
        for k, (num, den, _) in stats.items():
            L = (lg.apply(lambda g, num=num: num(g).sum(), include_groups=False) / lg[den].sum())
            rows[f"dev_{k}"] = wt * (num(s) - s[den] * s.team_league.map(L))
            rows[f"den_{k}"] = wt * s[den]
        frames.append(rows)
    allr = pd.concat(frames, ignore_index=True)
    latest = next(y for y in seasons if not stints(group, y, neutral).empty)   # 2021 forecasts: 2019
    last = _prep(stints(group, latest, neutral), role)
    out = allr.groupby("player_id").sum(numeric_only=True).drop(columns=["season", "w"])
    exp_last = allr[allr.season == latest].groupby("player_id")[expo].sum()
    out["exposure_last"] = exp_last.reindex(out.index).fillna(0)
    out["seasons_used"] = allr[allr[expo] > 0].groupby("player_id").season.nunique()
    by = birth_years()
    a = pd.Series(year - by.reindex(out.index.astype(int)).to_numpy(), index=out.index)
    out["age"] = a
    A = age_factor(a) if age else pd.Series(1.0, index=out.index)
    L0p = last[expo].sum()
    for k, (num, den, direction) in stats.items():
        L0 = num(last).sum() / last[den].sum()
        Rk = R * last[den].sum() / L0p if den != expo else R
        p = L0 + out[f"dev_{k}"] / (out[f"den_{k}"] + Rk)
        p = p * A if direction > 0 else p / A if direction < 0 else p
        m = out.exposure_last > 0                  # re-centre over the latest season's population
        p = p + (L0 - np.average(p[m], weights=out.exposure_last[m]))
        out[k] = p
        out[f"share_data_{k}"] = out[f"den_{k}"] / (out[f"den_{k}"] + Rk)
        out[f"lg_{k}"] = L0
    if role == "pitcher":
        lg_er = last.pitching_ER.sum() / last.pitching_IP.sum() * 9
        raw = (13 * last.pitching_HR.sum() + 3 * (last.pitching_BB.sum() + last.pitching_HBP.sum())
               - 2 * last.pitching_SO.sum()) / last.pitching_IP.sum()
        out["fip"] = (13 * out.hr + 3 * (out.bb + out.hbp) - 2 * out.k) / out.ipbf + (lg_er - raw)
        out["score"] = out.k - out.bb
        out["score_c"] = out.score - (out.lg_k - out.lg_bb)
    else:
        out["score"] = out.woba
        out["score_c"] = out.woba - out.lg_woba
    keep = [c_ for c_ in out.columns if not c_.startswith(("dev_", "den_"))]
    out = out[keep].reset_index()
    out["player_id"] = out.player_id.astype(int)
    return out


def _scaled(s: pd.DataFrame, scale: pd.DataFrame) -> pd.DataFrame:
    key = pd.MultiIndex.from_arrays([s.player_id.astype(int), s.season.astype(int), s.team_id.astype(int)])
    f = scale.reindex(key)
    s = s.copy()
    for c in scale.columns.intersection(s.columns):
        s[c] = s[c] * f[c].fillna(1.0).to_numpy()
    return s


def log_scale(logs: pd.DataFrame, w: pd.Series) -> pd.DataFrame:
    """Bootstrap multipliers per (player, season, team): re-weighted / original game-log totals."""
    cols = [c for c in logs.columns if c.startswith(("pitching_", "batting_"))]
    g = logs.assign(season=pd.to_datetime(logs.game_date).dt.year)
    if "pitching_IP" in g:              # log IP is in x.1 / x.2 notation: scale by outs
        ip = pd.to_numeric(g.pitching_IP, errors="coerce").fillna(0)
        g["pitching_IP"] = np.floor(ip) * 3 + np.round((ip - np.floor(ip)) * 10)
    f = g.game_id.astype(str).map(w).fillna(0).to_numpy()
    keys = [g.player_id.astype(int), g.season.astype(int), g.team_id.astype(int)]
    base = g[cols].groupby(keys).sum()
    rw = g[cols].mul(f, axis=0).groupby(keys).sum()
    out = (rw / base.replace(0, np.nan)).fillna(1.0)
    out.index.names = ["player_id", "season", "team_id"]
    return out


def active(role: str) -> bool:
    """Amendment 15 switch for one role: on unless V1 found Marcel "clearly worse" on the primary
    outcome (then that role keeps the amendment-13 ranking and Marcel is reported only)."""
    if "amendment15" not in config():
        return False
    f = processed("marcel_v1.json")
    key = json.loads(json.dumps({"amendment15": cfg(), "woba_weights": config()["woba_weights"]}))
    v = json.loads(f.read_text()) if f.exists() else {}
    if v.get("key") != key:
        v = {"key": key, "v1": v1()}
        f.write_text(json.dumps(v, indent=2, default=float))
    return v["v1"][role]["verdict"] != "clearly worse"


def candidate_scores(role: str, ids, scale: pd.DataFrame | None = None) -> pd.DataFrame:
    """Production projection (forecast season 2027 from AAA 2024-26) for the candidate ids."""
    p = project(role, cfg()["forecast_season"], scale=scale).set_index("player_id")
    return p.reindex(pd.Index(ids).astype(int))


# ------------------------------------------------------------------ V1: AAA next season
def _pool2(group: str, years: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Production-style two-season pool (amendment 6) from single-year park-adjusted stints."""
    raw = pd.concat([stints(group, y) for y in years], ignore_index=True)
    raw["league"] = raw.team_league
    raw["team_league"] = raw.season.astype(str) + "_" + raw.league
    if group == "pitching":
        df = milb_candidates._combine(raw, [c for c in milb_candidates.PITCH_SUM if c in raw] + ["outs"], "pitching_BF")
        df["IP"] = df.outs / 3
        last = raw[raw.season == years[-1]].groupby("player_id")[["pitching_IP"]].sum()
    else:
        df = milb_candidates._combine(raw, [c for c in milb_candidates.BAT_SUM if c in raw], "batting_PA")
        last = raw[raw.season == years[-1]].groupby("player_id")[["batting_PA"]].sum()
    return df.join(last.add_prefix("last_"), on="player_id").fillna({"last_pitching_IP": 0, "last_batting_PA": 0}), raw


def outcome(role: str, year: int) -> pd.DataFrame:
    """Season `year` AAA quality, league-relative (single-year park factors)."""
    group = "pitching" if role == "pitcher" else "hitting"
    s = stints(group, year)
    df = milb_candidates._combine(s, [c for c in (milb_candidates.PITCH_SUM if role == "pitcher" else milb_candidates.BAT_SUM)
                                      if c in s] + (["outs"] if role == "pitcher" else []),
                                  "pitching_BF" if role == "pitcher" else "batting_PA")
    if role == "pitcher":
        df["IP"] = df.outs / 3
        lg = baseline._league_table("pitching", year, s)
        kbb = (df.pitching_SO - df.pitching_BB) / df.pitching_BF - baseline._league_mix(df, lg, "k_minus_bb_pct")
        cf = baseline._league_mix(df, lg, "cfip")
        fip = (13 * df.pitching_HR + 3 * (df.pitching_BB + df.pitching_HBP) - 2 * df.pitching_SO) / df.IP + cf
        return pd.DataFrame({"player_id": df.player_id, "n": df.pitching_BF, "y_kbb": kbb,
                             "y_neg_fip": -(fip - baseline._league_mix(df, lg, "fip"))})
    lg = baseline._league_table("batting", year, s)
    r = baseline._hitter_rates(df.set_index("player_id")).reset_index()
    return pd.DataFrame({"player_id": df.player_id, "n": df.batting_PA,
                         "y_woba": (r.woba - baseline._league_mix(df, lg, "woba")).values})


def _metrics(role: str, years: list[int]) -> pd.DataFrame:
    """The production baseline metrics on the two-season pool `years` (recent = Aug-Sep of the last)."""
    group = "pitching" if role == "pitcher" else "hitting"
    pool, raw = _pool2(group, years)
    y = years[-1]
    if role == "pitcher":
        empty = pd.DataFrame(columns=["player_id", "pitching_BF", "pitching_SO", "pitching_BB"])
        return baseline.pitcher_metrics(pool, y, raw, empty)
    rec = baseline_weights.fetch("hitting", y, f"{y}-08-01", f"{y}-09-30", pf=pf_for(y))
    return baseline.hitter_metrics(pool, y, raw, rec)


def _eligible(role: str, d: pd.DataFrame) -> pd.Series:
    v = cfg()["validation"][role]
    if role == "pitcher":
        return (d.pitching_GS >= v["min_gs_2y"]) & (d.IP >= v["min_ip_2y"]) & (d.last_pitching_IP >= v["min_ip_last"])
    return (d.batting_PA >= v["min_pa_2y"]) & (d.last_batting_PA >= v["min_pa_last"])


def _estimated_weights(role: str, target_year: int) -> dict:
    """Amendment-4 NNLS weights fit on the transition ending in target_year - 1 (never on the
    target season); judgment weights below min_n or when every coefficient is 0."""
    e = config()["baseline"]["estimation"]
    t = target_year - 1
    d = _metrics(role, [t - 2, t - 1])
    d = d[_eligible(role, d)]
    o = outcome(role, t)
    d = d.merge(o, on="player_id")
    d = d[d.n >= cfg()["validation"][role]["min_bf_target" if role == "pitcher" else "min_pa_target"]]
    y = d["y_neg_fip" if role == "pitcher" else "y_woba"].to_numpy(float)
    coef = baseline_weights.nnls_weights(baseline_weights.design(role, d), y, d.n.to_numpy(float)) if len(d) >= e["min_n"] else None
    names = list(config()["baseline"][role])
    return dict(zip(names, coef)) if coef is not None else baseline.judgment_weights(role)


def v1_fold(role: str, target_year: int) -> pd.DataFrame:
    years = cfg()["validation"]["folds"][target_year]
    d = _metrics(role, [years[1], years[2]])                 # the two seasons before the target
    d = d[_eligible(role, d)].copy()
    o = outcome(role, target_year)
    d = d.merge(o, on="player_id")
    d = d[d.n >= cfg()["validation"][role]["min_bf_target" if role == "pitcher" else "min_pa_target"]].copy()
    d["judgment"] = baseline.score(d, role, baseline.judgment_weights(role)).baseline_score.reindex(d.index)
    d["estimated"] = baseline.score(d, role, _estimated_weights(role, target_year)).baseline_score.reindex(d.index)
    m = project(role, target_year).set_index("player_id")
    d["marcel"] = d.player_id.astype(int).map(m.score)
    if role == "pitcher":                        # secondary outcome -FIP: Marcel's own projected FIP
        d["marcel_neg_fip"] = -d.player_id.astype(int).map(m.fip)
    for mult in cfg()["sensitivity_regress"]:    # registered sensitivities (reported, never select the model)
        d[f"marcel_R{mult:g}"] = d.player_id.astype(int).map(project(role, target_year, mult=mult).set_index("player_id").score)
    d["marcel_no_age"] = d.player_id.astype(int).map(project(role, target_year, age=False).set_index("player_id").score)
    d["fold"] = target_year
    return d


def _pred(method: str, outcome_col: str) -> str:
    return "marcel_neg_fip" if method == "marcel" and outcome_col == "y_neg_fip" else method


def _paired(folds: list[pd.DataFrame], a: str, b: str, y: str, rng) -> dict:
    v = cfg()["validation"]["bootstrap"]
    mats = [f[[_pred(a, y), b, y]].astype(float).dropna().to_numpy() for f in folds]

    def diff(x):
        return backtest.spearman(x[:, 0], x[:, 2]) - backtest.spearman(x[:, 1], x[:, 2])
    est = [diff(m) for m in mats]
    draws = np.array([[diff(m[rng.integers(0, len(m), len(m))]) for m in mats] for _ in range(v["draws"])])
    q = (1 - v["ci"]) / 2
    out = {"a": a, "b": b, "outcome": y}
    for i, f in enumerate(folds):
        out[f"diff_{int(f.fold.iat[0])}"] = est[i]
        out[f"ci_{int(f.fold.iat[0])}"] = [float(np.nanquantile(draws[:, i], q)), float(np.nanquantile(draws[:, i], 1 - q))]
    mean = draws.mean(axis=1)
    out["diff_mean"] = float(np.mean(est))
    out["ci_mean"] = [float(np.nanquantile(mean, q)), float(np.nanquantile(mean, 1 - q))]
    return out


def v1() -> dict:
    rng = np.random.default_rng(cfg()["validation"]["bootstrap"]["seed"])
    ni = cfg()["validation"]["noninferiority"]
    res = {}
    for role, prim, outs in (("pitcher", "y_kbb", ["y_kbb", "y_neg_fip"]), ("hitter", "y_woba", ["y_woba"])):
        folds = [v1_fold(role, y) for y in sorted(cfg()["validation"]["folds"])]
        rhos = [{"fold": int(f.fold.iat[0]), "n": len(f), "outcome": y_,
                 **{m: backtest.spearman(f[_pred(m, y_)], f[y_]) for m in ("marcel", "judgment", "estimated")}}
                for f in folds for y_ in outs]
        sens = [{"fold": int(f.fold.iat[0]), "outcome": prim,
                 **{c: backtest.spearman(f[c], f[prim]) for c in f.columns if c.startswith("marcel_R") or c == "marcel_no_age"}}
                for f in folds]
        pairs = [_paired(folds, "marcel", b, y_, rng) for y_ in outs for b in ("judgment", "estimated")]
        prim_pairs = [p for p in pairs if p["outcome"] == prim]
        verdict = ("clearly worse" if any(p["ci_mean"][1] < ni for p in prim_pairs)
                   else "non-inferior" if all(p["ci_mean"][0] > ni for p in prim_pairs) else "not shown non-inferior")
        res[role] = {"rho": rhos, "paired": pairs, "verdict": verdict, "sensitivity": sens,
                     "pf1_seasons": sorted({y for f in cfg()["validation"]["folds"].values() for y in f} - set(PF_SEASONS))}
        pd.concat(folds).to_csv(processed(f"marcel_v1_{role}.csv"), index=False)
    return res


# ------------------------------------------------------------------ V2 / V3: CPBL
def _cpbl_marcel(coh: pd.DataFrame, neutral: bool, year_col: str = "season") -> pd.DataFrame:
    """Centred Marcel K-BB%, K%, BB% (projection minus that season's AAA average) per cohort row."""
    out = pd.DataFrame(np.nan, index=coh.index, columns=["kbb", "k", "bb"])
    for y, g in coh.groupby(year_col):
        y = int(y)
        seasons = [y - 1, y - 2, y - 3]
        m = project("pitcher", y, neutral=neutral, seasons=seasons).set_index("player_id")
        ids = g.mlbam_id.astype(int)
        out.loc[g.index, "kbb"] = ids.map(m.score_c).values
        out.loc[g.index, "k"] = ids.map(m.k - m.lg_k).values
        out.loc[g.index, "bb"] = ids.map(m.bb - m.lg_bb).values
    return out


def v2_v3() -> dict:
    from . import backtest_chrono, cpbl_census
    rng = np.random.default_rng(cfg()["validation"]["bootstrap"]["seed"] + 1)
    res = {}
    a11 = pd.read_csv(processed("backtest_chrono_cohort.csv"))
    a11[["marcel", "marcel_k", "marcel_bb"]] = _cpbl_marcel(a11, neutral=False).to_numpy()
    res["a11"] = {"n": int(a11.marcel.notna().sum()),
                  "marcel": backtest_chrono._corr_test("Marcel K-BB% vs CPBL K-BB%", a11, "marcel", "cpbl_kbb_c", rng),
                  "baseline": backtest_chrono._corr_test("baseline vs CPBL K-BB%", a11, "baseline_score", "cpbl_kbb_c", rng),
                  "paired": backtest_chrono.paired_diff(a11, "marcel", "baseline_score", "cpbl_kbb_c", rng)}
    cen = pd.read_csv(processed("cpbl_census_pitchers.csv"))
    cen[["marcel", "marcel_k", "marcel_bb"]] = _cpbl_marcel(cen, neutral=True).to_numpy()
    res["census"] = {"n": int(cen.marcel.notna().sum()),
                     "marcel": cpbl_census._test("Marcel K-BB% vs CPBL K-BB%", cen, "marcel", "cpbl_kbb_c", rng),
                     "single_season": cpbl_census._test("CP1 AAA K-BB% vs CPBL K-BB%", cen, "aaa_kbb_c", "cpbl_kbb_c", rng),
                     "paired": backtest_chrono.paired_diff(cen.dropna(subset=["marcel"]), "marcel", "aaa_kbb_c", "cpbl_kbb_c", rng)}
    # V3: rolling translation on the Marcel projection (>= 15 earlier arrivals, else the mean)
    v3 = {}
    for name, d in (("census", cen), ("a11", a11)):
        for comp, x, ycol in (("k-bb", "marcel", "cpbl_kbb_c"), ("k", "marcel_k", "cpbl_k_c"), ("bb", "marcel_bb", "cpbl_bb_c")):
            v3[f"{name}_{comp}"] = _rolling(d, x, ycol, rng)
    res["v3"] = v3
    a11.to_csv(processed("marcel_v2_a11.csv"), index=False)
    cen.to_csv(processed("marcel_v2_census.csv"), index=False)
    return res


def _rolling(d: pd.DataFrame, x: str, y: str, rng, min_train: int = 15) -> dict:
    rows = []
    for yr in sorted(d.season.unique()):
        tr, te = d[d.season < yr].dropna(subset=[x, y]), d[d.season == yr].dropna(subset=[x, y])
        if not len(te) or not len(tr):
            continue
        m0 = tr[y].mean()
        if len(tr) >= min_train and np.ptp(tr[x]) > 0:
            k, a = np.polyfit(tr[x], tr[y], 1)
            m1 = a + k * te[x]
        else:
            m1 = pd.Series(m0, index=te.index)
        rows += [{"season": yr, "actual": v, "m0": m0, "m1": p} for v, p in zip(te[y], m1)]
    f = pd.DataFrame(rows)
    if f.empty:
        return {"n_held_out": 0}
    e0, e1 = (f.actual - f.m0).abs().to_numpy(), (f.actual - f.m1).abs().to_numpy()
    b = cfg()["validation"]["bootstrap"]
    diffs = [e1[i].mean() - e0[i].mean() for i in (rng.integers(0, len(f), len(f)) for _ in range(b["draws"]))]
    q = (1 - b["ci"]) / 2
    return {"n_held_out": len(f), "mae_mean": float(e0.mean()), "mae_marcel": float(e1.mean()),
            "diff": float(e1.mean() - e0.mean()), "ci": [float(np.quantile(diffs, q)), float(np.quantile(diffs, 1 - q))]}


# ------------------------------------------------------------------ run
def ranking_effect() -> dict:
    """Signable top 3 under the old baseline vs Marcel (after evaluate has written the results)."""
    out = {}
    sh = pd.read_csv(processed("recommendation_shortlist.csv"))
    for role, f, old, new in (("pitcher", "pitcher_fit_results.csv", "combined_rank_a13", "combined_rank"),
                              ("hitter", "hitter_fit_results.csv", "baseline_rank_tracked", "marcel_rank_tracked")):
        D = pd.read_csv(processed(f))
        if new not in D or old not in D:
            continue
        D = D[D.fit_available & D.signable.fillna(False).astype(bool)]
        a, b = D.sort_values(old), D.sort_values(new)
        pk = sh[(sh.role == role) & (sh.rule_rank_signable == 1)]
        out[role] = {"before": a.name.head(3).tolist(), "after": b.name.head(3).tolist(),
                     "rank_corr_signable": float(D[old].corr(D[new], method="spearman")),
                     "p_top1_signable_pick": float(pk.boot_p_top1_signable.iat[0]) if len(pk) else None}
    return out


def run() -> dict:
    v = v1()
    key = json.loads(json.dumps({"amendment15": cfg(), "woba_weights": config()["woba_weights"]}))
    processed("marcel_v1.json").write_text(json.dumps({"key": key, "v1": v}, indent=2, default=float))
    out = {"v1": v, "v2_v3": v2_v3(), "ranking": ranking_effect()}
    processed("marcel_summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float))
    report(out)
    return out


def report(out: dict) -> None:
    L = ["# AAA Marcel projections (amendment 15; generated by `python -m src.marcel`)", "",
         "Marcel adapted to AAA: last three AAA seasons (hitters 5/4/3, pitchers 3/2/1), regressed by 1,200 PA / "
         "600 BF of league-average play, age factor; pitchers ranked by projected K% - BB%, hitters by projected wOBA.", "",
         "## V1: next-season AAA, Spearman rho by method", ""]
    for role in ("pitcher", "hitter"):
        r = out["v1"][role]
        L += [f"### {role.capitalize()}s (verdict on the primary outcome: **{r['verdict']}**)", "",
              md_table(pd.DataFrame(r["rho"]), ".3f"), "", "Paired difference Marcel minus comparator (90% CI):", ""]
        rows = []
        for p in r["paired"]:
            row = {"outcome": p["outcome"], "vs": p["b"]}
            for k, v in p.items():
                if k.startswith("diff_"):
                    yr = k[5:]
                    ci = p[f"ci_{yr}"]
                    row[yr] = f"{v:+.3f} ({ci[0]:+.3f} to {ci[1]:+.3f})"
            rows.append(row)
        L += [md_table(pd.DataFrame(rows)), ""]
        if r.get("sensitivity"):
            L += ["Registered sensitivities, Marcel rho on the primary outcome (reported only; they never choose the "
                  "model):", "", md_table(pd.DataFrame(r["sensitivity"]), ".3f"), ""]
        if r.get("pf1_seasons"):
            L += [f"Seasons without single-year park factors (PF = 1, flagged): {', '.join(map(str, r['pf1_seasons']))}.", ""]
    L += ["For pitchers, the -FIP rows use Marcel's projected FIP; the judgment and estimated baselines are their "
          "composite scores for both outcomes.", ""]
    v = out["v2_v3"]
    L += ["## V2: CPBL K-BB% (centred)", ""]
    for name, lab in (("a11", "Amendment-11 cohort (pre-arrival, park-adjusted)"), ("census", "Census 2018-2026 (PF = 1)")):
        r = v[name]
        L += [f"**{lab}**, n {r['n']}:", "", md_table(pd.DataFrame([r["marcel"], r.get("baseline", r.get("single_season"))]), ".2f"),
              "", f"Paired difference Marcel minus the comparator above: {r['paired'].get('rho', float('nan')):+.2f} "
              f"(90% CI {r['paired'].get('ci_lo', float('nan')):+.2f} to {r['paired'].get('ci_hi', float('nan')):+.2f}).", ""]
    L += ["## V3: rolling translation on the Marcel projection (>= 15 earlier arrivals, else the training mean)", "",
          md_table(pd.DataFrame(v["v3"]).T.reset_index(names="cohort").assign(
              ci=lambda d: d["ci"].map(lambda c: f"{c[0]:+.4f} to {c[1]:+.4f}")), ".4f"), ""]
    if out.get("ranking"):
        L += ["## Use in the ranking: signable top 3 before / after Marcel", "",
              "Pitchers: amendment 13 (old baseline in the 0.5 slot) vs amendment 15 (Marcel K-BB%); hitters: old "
              "baseline rank vs Marcel wOBA rank.", "",
              md_table(pd.DataFrame([{"role": k, "before": ", ".join(v["before"]), "after": ", ".join(v["after"]),
                                      "rank corr (signable)": v["rank_corr_signable"],
                                      "P(top 1 among signable), pick": v["p_top1_signable_pick"]}
                                     for k, v in out["ranking"].items()]), ".2f"), ""]
    (path("reports") / "marcel.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    print(json.dumps(run(), indent=1, ensure_ascii=False, default=float))
