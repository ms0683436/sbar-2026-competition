"""Amendment 7 — backtest with 2026 CPBL outcomes of import players (stats.cpbl.com.tw).

Johnny decided (2026-10-14) to use the 2026 first-team regular-season stats of import players
from CPBL's advanced-stats site as backtest OUTCOMES only; rules in config `amendment7`,
committed before any outcome of the cohort was parsed. The site's terms ask for written
authorization, which we do not have (reports/limitations.md L10). Nothing here touches the
2026 ranking or picks.

Pitchers: the amendment-2 cohort (2024-25 arrivals, REBAS outcomes) plus the 2026 cohort
(>= 400 tracked AAA pitches in 2025, >= 60 CPBL 2026 BF). Predictors come from AAA before
arrival: the amendment-5 baseline on the preceding AAA season, league-adjusted K% and BB%,
and F1 / F2 / Fit as in amendment 2. Outcomes are centred on their season's league value.
Hitters: 2026 only (>= 150 AAA PA in 2025, >= 100 CPBL PA), predictors from 2025 AAA.

Inputs: data/reference/cpbl_imports_2026_site.csv (identities) and the site's export of the
2026 一軍例行賽 PR table under data/raw/cpbl_site/ (git-ignored; not redistributed).

Usage:  python -m src.backtest_2026
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import backtest, baseline, baseline_weights, fit_score, milb_candidates, milb_loader, park, rebas_loader
from . import recruitment_backtest
from .common import ROOT, config, md_table, path, processed, raw, zscore


def cfg() -> dict:
    return config()["amendment7"]


# ------------------------------------------------------------------ AAA predictors
def pitcher_baseline(year: int) -> pd.DataFrame:
    """Amendment-5 pitcher baseline on one AAA season, z against every pitcher of that season
    with >= baseline_min_bf BF; park factors from (year, year + 1)."""
    seasons = [year, year + 1]
    pf = park.factors_for(seasons, recruitment_backtest._pitches(seasons))
    s = baseline_weights.fetch("pitching", year, pf=pf)
    df = milb_candidates._combine(s, [c for c in milb_candidates.PITCH_SUM if c in s] + ["outs"], "pitching_BF")
    df["IP"] = df.outs / 3
    df = df[df.pitching_BF >= cfg()["pitcher"]["baseline_min_bf"]]
    empty = pd.DataFrame(columns=["player_id", "pitching_BF", "pitching_SO", "pitching_BB"])
    d = baseline.pitcher_metrics(df, year, s, empty)
    d = baseline.score(d, "pitcher", baseline.judgment_weights("pitcher"))
    out = d[["player_id", "pitching_BF", "baseline_score", "k_pct_adj", "bb_pct_adj", "fip", "k_pct", "bb_pct"]].copy()
    out["mlbam_id"] = out.player_id.astype(int)
    out["aaa_season"] = year
    return out.drop(columns="player_id").rename(columns={"pitching_BF": "aaa_bf", "k_pct": "aaa_k_pct",
                                                         "bb_pct": "aaa_bb_pct"})


def aaa_league(year: int) -> dict:
    s = baseline_weights.fetch("pitching", year)
    return {"k_pct": s.pitching_SO.sum() / s.pitching_BF.sum(), "bb_pct": s.pitching_BB.sum() / s.pitching_BF.sum()}


def translation(coh: pd.DataFrame, aaa_whiff: dict) -> dict:
    """Amendment 7b item 3: AAA -> CPBL translation of K%, BB% and whiff% (descriptive)."""
    c = config()["amendment7b"]
    b = c["bootstrap"]
    rng = np.random.default_rng(b["seed"])
    out = {"league_ratio": [], "ols": []}
    for s, g in coh.groupby("season"):
        y = int(s) - 1
        al = aaa_league(y)
        out["league_ratio"].append({"cpbl_season": int(s), "aaa_season": y,
                                    "k_pct": g.lg_k_pct.iat[0] / al["k_pct"], "bb_pct": g.lg_bb_pct.iat[0] / al["bb_pct"],
                                    "whiff_pct": g.lg_whiff_pct.iat[0] / aaa_whiff[y]})
    a_ = (1 - b["ci"]) / 2
    for m, x, y, at in (("K%", "aaa_k_pct", "cpbl_k_pct", c["translation"]["k_pct_at"]),
                        ("BB%", "aaa_bb_pct", "cpbl_bb_pct", c["translation"]["bb_pct_at"]),
                        ("whiff%", "aaa_whiff_pct", "cpbl_whiff_pct", c["translation"]["whiff_pct_at"])):
        d = coh[[x, y]].astype(float).dropna().to_numpy()
        bb, aa = np.polyfit(d[:, 0], d[:, 1], 1)
        boots = []
        for _ in range(b["draws"]):
            i = rng.integers(0, len(d), len(d))
            if np.ptp(d[i, 0]) > 0:
                boots.append(np.polyfit(d[i, 0], d[i, 1], 1))
        boots = np.array(boots)
        row = {"metric": m, "n": len(d), "a": aa, "b": bb,
               "b_ci": [float(np.quantile(boots[:, 0], a_)), float(np.quantile(boots[:, 0], 1 - a_))],
               "aaa_mean": float(d[:, 0].mean()), "cpbl_mean": float(d[:, 1].mean())}
        for v in at:
            pred = boots[:, 1] + boots[:, 0] * v
            row[f"at_{v:.2f}"] = [float(aa + bb * v), float(np.quantile(pred, a_)), float(np.quantile(pred, 1 - a_))]
        out["ols"].append(row)
    return out


def hitter_predictors(year: int) -> pd.DataFrame:
    """Hitter baseline (estimated weights in use and judgment weights) on one AAA season."""
    h = cfg()["hitter"]
    lo, hi = h["recent_window"]
    pf = park.factors_for([year, year + 1], recruitment_backtest._pitches([year, year + 1]))
    s = baseline_weights.fetch("hitting", year, pf=pf)
    rec = baseline_weights.fetch("hitting", year, lo, hi, pf=pf)
    s = s[s.player_position != "P"]
    df = milb_candidates._combine(s, [c for c in milb_candidates.BAT_SUM if c in s], "batting_PA")
    df = df[df.batting_PA >= h["min_aaa_pa"]]
    d = baseline.hitter_metrics(df, year, s, rec)
    est = baseline.score(d, "hitter", baseline.active_weights("hitter"))
    jud = baseline.score(d, "hitter", baseline.judgment_weights("hitter"))
    out = est[["player_id", "batting_PA", "baseline_score", "k_pct_raw", "iso_raw", "woba_raw"]].copy()
    out["baseline_score_judgment"] = jud.baseline_score.reindex(est.index)
    out["mlbam_id"] = out.player_id.astype(int)
    return out.drop(columns="player_id").rename(columns={"batting_PA": "aaa_pa", "k_pct_raw": "aaa_k_pct",
                                                         "iso_raw": "aaa_iso", "woba_raw": "aaa_woba"})


# ------------------------------------------------------------------ identities and outcomes
def identities() -> pd.DataFrame:
    return pd.read_csv(ROOT / cfg()["identities"], comment="#", dtype={"mlbam_id": "Int64"})


EXPORT_COLS = {  # site column -> ours (rates as fractions)
    "球員": "site_name", "姓名": "site_name", "球隊": "team", "打席": "pa", "PA": "pa",
    "三振%": "k_pct", "保送%": "bb_pct", "揮空%": "whiff_pct", "追打%": "chase_pct",
    "加權上壘率": "woba", "wOBA": "woba", "純長打率": "iso", "ISO": "iso",
}


def load_export(role: str) -> pd.DataFrame:
    """The site's PR-table export for 2026 一軍例行賽 (role = 'pitcher' or 'batter')."""
    d = raw("cpbl_site")
    files = sorted(p for p in d.glob(f"*{role}*") if p.suffix.lower() in (".csv", ".xlsx", ".xls"))
    if not files:
        raise FileNotFoundError(f"no {role} export in {d}")
    f = files[-1]
    df = pd.read_excel(f) if f.suffix.lower() != ".csv" else pd.read_csv(f, encoding="utf-8-sig")
    df = df.rename(columns={c: EXPORT_COLS[c.strip()] for c in df.columns if c.strip() in EXPORT_COLS})
    for c in ("k_pct", "bb_pct", "whiff_pct", "chase_pct"):
        if c not in df:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c].astype(str).str.strip().str.rstrip("%"), errors="coerce") / 100
        elif df[c].max() > 1:
            df[c] = df[c] / 100
    for c in ("pa", "woba", "iso"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _league(df: pd.DataFrame, col: str, w: str = "pa") -> float:
    d = df[[col, w]].dropna()
    return float((d[col] * d[w]).sum() / d[w].sum())


def rebas_league() -> dict:
    """League K%, BB%, K-BB%, whiff% of each REBAS season (all pitchers)."""
    c = rebas_loader.load()
    return {int(s): backtest._rates(g) | {"k_pct": _kbb(g)[0], "bb_pct": _kbb(g)[1]} for s, g in c.groupby("season")}


def _kbb(df: pd.DataFrame) -> tuple[float, float]:
    pa = df[df.pa_end.astype(bool) & df.pa_event.ne("OTHER")]
    return pa.pa_event.eq("SO").mean(), pa.pa_event.eq("uBB").mean()


# ------------------------------------------------------------------ pitchers
def _aaa_league_whiff(aaa: pd.DataFrame) -> dict:
    d = aaa[~aaa.is_bunt.astype(bool)]
    g = d.groupby("season")[["whiff", "swing"]].sum()
    return (g.whiff / g.swing).to_dict()


def pitcher_cohort(exp: pd.DataFrame, ids: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Screening log and pooled cohort (amendment-2 cohort + 2026 cohort) with predictors and
    centred outcomes. The 2026 screen reads only BF and hand from the export."""
    p = cfg()["pitcher"]
    cpbl = rebas_loader.load()
    aaa_env = milb_loader.load()
    aaa_all = pd.concat([backtest.aaa_extra(), aaa_env], ignore_index=True)
    # 2026 screen
    e = exp[["site_name", "pa"]].merge(ids[ids.role.eq("pitcher")], on="site_name", how="inner")
    a25 = aaa_env[aaa_env.season == p["aaa_season"]]
    rows = []
    a2 = pd.read_csv(processed("backtest_screening.csv"), dtype={"rebas_id": str})
    a2_ids = set(a2.loc[a2.excluded.fillna("").eq(""), "mlbam_id"].dropna().astype(int))
    for r in e.itertuples(index=False):
        w = a25[a25.pitcher_id == str(r.mlbam_id)] if pd.notna(r.mlbam_id) else a25.iloc[0:0]
        hand = w.p_throws.mode().iat[0] if len(w) else None
        row = {"site_name": r.site_name, "english_name": r.english_name, "mlbam_id": r.mlbam_id,
               "season": p["outcome_season"], "cpbl_bf": r.pa, "aaa_pitches": len(w), "aaa_hand": hand,
               "site_hand": r.throws}
        if pd.isna(r.mlbam_id):
            reason = "no MLBAM id"
        elif int(r.mlbam_id) in a2_ids:
            reason = "counted in an earlier season"
        elif len(w) < p["min_aaa_tracked_pitches"]:
            reason = f"AAA {p['aaa_season']} pitches < {p['min_aaa_tracked_pitches']}"
        elif not r.pa >= p["min_cpbl_bf"]:
            reason = f"CPBL BF < {p['min_cpbl_bf']}"
        elif hand != r.throws:
            reason = "hand mismatch"
        else:
            reason = ""
        rows.append(row | {"excluded": reason})
    sc = pd.DataFrame(rows)
    new = sc[sc.excluded.eq("")].copy()
    # Fit for the 2026 cohort: AAA 2025 window, environment CPBL 2024-25 vs AAA 2024-25
    base = fit_score.build_tables(cpbl, aaa_env, set(), set())
    w = a25[a25.pitcher_id.isin(new.mlbam_id.astype(int).astype(str))]
    w = w.assign(cell=fit_score.assign_cell(w))
    t = dict(base, pit=fit_score._agg(w, ["pitcher_id", "season", "game_id", "cell"]))
    env = fit_score.environment(t, cpbl_seasons=p["env_seasons"], aaa_seasons=p["env_seasons"])
    f = fit_score.pitcher_fit(t, env)[["fit_mix_whiff", "fit_mix_damage"]]
    aaa_r = pd.DataFrame({k: backtest._rates(g) for k, g in w.groupby("pitcher_id")}).T.add_prefix("aaa_")
    new["key"] = new.mlbam_id.astype(int).astype(str)
    new = new.join(f.join(aaa_r), on="key")
    ex = exp.set_index("site_name")
    lg26 = {c: _league(exp, c) for c in ("k_pct", "bb_pct", "whiff_pct")}
    new["cpbl_k_pct"] = new.site_name.map(ex.k_pct)
    new["cpbl_bb_pct"] = new.site_name.map(ex.bb_pct)
    new["cpbl_whiff_pct"] = new.site_name.map(ex.whiff_pct)
    new = new.assign(lg_k_pct=lg26["k_pct"], lg_bb_pct=lg26["bb_pct"], lg_whiff_pct=lg26["whiff_pct"])
    # amendment-2 cohort with REBAS outcomes
    old = pd.read_csv(processed("backtest_cohort.csv"), dtype={"rebas_id": str})
    lg = rebas_league()
    kb = {}
    for r in old.itertuples(index=False):
        kb[r.rebas_id] = _kbb(cpbl[(cpbl.season == r.season) & (cpbl.pitcher_id == r.rebas_id)])
    old["cpbl_k_pct"] = old.rebas_id.map(lambda i: kb[i][0])
    old["cpbl_bb_pct"] = old.rebas_id.map(lambda i: kb[i][1])
    old["lg_k_pct"] = old.season.map(lambda s: lg[s]["k_pct"])
    old["lg_bb_pct"] = old.season.map(lambda s: lg[s]["bb_pct"])
    old["lg_whiff_pct"] = old.season.map(lambda s: lg[s]["whiff_pct"])
    keep = ["season", "english_name", "mlbam_id", "fit_mix_whiff", "fit_mix_damage", "aaa_whiff_pct",
            "cpbl_k_pct", "cpbl_bb_pct", "cpbl_whiff_pct", "lg_k_pct", "lg_bb_pct", "lg_whiff_pct"]
    coh = pd.concat([old[keep].assign(source="REBAS", rebas_name=old.rebas_name),
                     new[keep].assign(source="stats.cpbl.com.tw", rebas_name=new.site_name)], ignore_index=True)
    coh["mlbam_id"] = coh.mlbam_id.astype(int)
    coh["aaa_season"] = coh.season - 1
    bl = pd.concat([pitcher_baseline(y) for y in sorted(coh.aaa_season.unique())], ignore_index=True)
    coh = coh.merge(bl, on=["mlbam_id", "aaa_season"], how="left")
    aw = _aaa_league_whiff(aaa_all)
    coh["fit"] = 0.6 * zscore(coh.fit_mix_whiff) + 0.4 * zscore(coh.fit_mix_damage)
    coh["cpbl_k_c"] = coh.cpbl_k_pct - coh.lg_k_pct
    coh["cpbl_bb_c"] = coh.cpbl_bb_pct - coh.lg_bb_pct
    coh["cpbl_kbb_c"] = coh.cpbl_k_c - coh.cpbl_bb_c
    coh["d_whiff_c"] = (coh.cpbl_whiff_pct - coh.lg_whiff_pct) - (coh.aaa_whiff_pct - coh.aaa_season.map(aw))
    d = coh[["baseline_score", "cpbl_kbb_c"]].dropna()
    b = np.polyfit(d.baseline_score, d.cpbl_kbb_c, 1)
    coh["kbb_resid_baseline"] = coh.cpbl_kbb_c - np.polyval(b, coh.baseline_score)
    return sc, coh


P_TESTS = [
    ("P1 baseline vs centred CPBL K-BB% (primary)", "baseline_score", "cpbl_kbb_c", 1),
    ("P2 AAA K% vs centred CPBL K%", "k_pct_adj", "cpbl_k_c", 1),
    ("P3 AAA BB% vs centred CPBL BB%", "bb_pct_adj", "cpbl_bb_c", 1),
    ("P4 Fit vs CPBL K-BB% residual after baseline", "fit", "kbb_resid_baseline", 1),
    ("P5 F1 vs centred whiff% change", "fit_mix_whiff", "d_whiff_c", 1),
]
Q_TESTS = [
    ("Q1 baseline (estimated weights) vs CPBL wOBA (primary)", "baseline_score", "cpbl_woba", 1),
    ("Q2 baseline (judgment weights) vs CPBL wOBA", "baseline_score_judgment", "cpbl_woba", 1),
    ("Q3 AAA K% vs CPBL K%", "aaa_k_pct", "cpbl_k_pct", 1),
    ("Q4 AAA ISO vs CPBL ISO", "aaa_iso", "cpbl_iso", 1),
    ("Q5 AAA wOBA vs CPBL wOBA", "aaa_woba", "cpbl_woba", 1),
]


def _tests(df: pd.DataFrame, tests, rng) -> list[dict]:
    b = cfg()["bootstrap"]
    out = []
    for label, x, y, sign in tests:
        d = df[[x, y]].astype(float).dropna()
        if len(d) < 3:
            out.append({"test": label, "n": len(d), "rho": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                        "verdict": "too few"})
            continue
        est, lo, hi = backtest.boot_ci(d, x, y, b["draws"], b["ci"], rng)
        v = ("descriptive only (n below minimum)" if len(d) < cfg()["min_n_for_conclusion"]
             else "supports" if sign * lo > 0 and sign * hi > 0
             else "contradicts" if sign * lo < 0 and sign * hi < 0 else "inconclusive")
        out.append({"test": label, "n": len(d), "rho": est, "ci_lo": lo, "ci_hi": hi, "verdict": v})
    return out


# ------------------------------------------------------------------ hitters
def hitter_cohort(exp: pd.DataFrame, ids: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    h = cfg()["hitter"]
    e = exp.merge(ids[ids.role.eq("hitter")], on="site_name", how="inner")
    pr = hitter_predictors(h["aaa_season"])
    e = e.merge(pr, on="mlbam_id", how="left")
    e["excluded"] = np.select(
        [e.mlbam_id.isna(), e.aaa_pa.isna(), ~(e.pa.astype(float) >= h["min_cpbl_pa"])],
        ["no MLBAM id", f"AAA {h['aaa_season']} PA < {h['min_aaa_pa']}", f"CPBL PA < {h['min_cpbl_pa']}"], "")
    sc = e[["site_name", "english_name", "mlbam_id", "pa", "aaa_pa", "excluded"]].rename(columns={"pa": "cpbl_pa"})
    coh = e[e.excluded.eq("")].rename(columns={"woba": "cpbl_woba", "k_pct": "cpbl_k_pct", "iso": "cpbl_iso",
                                               "pa": "cpbl_pa"})
    return sc, coh


# ------------------------------------------------------------------ run
def run() -> dict:
    ids = identities()
    rng = np.random.default_rng(cfg()["bootstrap"]["seed"])
    ep = load_export("pitcher")
    try:
        eb = load_export("batter")
    except FileNotFoundError:      # not needed when no import hitter passes the AAA screen
        eb = pd.DataFrame({"site_name": ids.loc[ids.role.eq("hitter"), "site_name"], "pa": np.nan,
                           "woba": np.nan, "k_pct": np.nan, "iso": np.nan})
    psc, pc = pitcher_cohort(ep, ids)
    hsc, hc = hitter_cohort(eb, ids)
    pres = _tests(pc, P_TESTS, rng)
    desc = []
    for name, part in (("2026 cohort", pc[pc.season == 2026]), ("amendment-2 cohort", pc[pc.season < 2026])):
        desc += [dict(t, test=f"{t['test']} — {name}") for t in _tests(part, P_TESTS[:1], rng)]
    hres = _tests(hc, Q_TESTS, rng)
    aaa_all = pd.concat([backtest.aaa_extra(), milb_loader.load()], ignore_index=True)
    tr = translation(pc, _aaa_league_whiff(aaa_all))
    summ = {"pitchers": {"n": len(pc), "by_season": pc.season.value_counts().sort_index().to_dict(),
                         "n_with_baseline": int(pc.baseline_score.notna().sum()), "tests": pres, "descriptive": desc},
            "hitters": {"n": len(hc), "tests": hres}, "translation": tr}
    for name, df in (("backtest2026_pitcher_screening.csv", psc), ("backtest2026_pitcher_cohort.csv", pc),
                     ("backtest2026_hitter_screening.csv", hsc), ("backtest2026_hitter_cohort.csv", hc)):
        df.to_csv(processed(name), index=False)
    processed("backtest2026_summary.json").write_text(json.dumps(summ, indent=2, ensure_ascii=False, default=str))
    report(psc, pc, hsc, hc, summ)
    return summ


def report(psc, pc, hsc, hc, summ) -> None:
    pcols = ["season", "source", "english_name", "aaa_bf", "baseline_score", "k_pct_adj", "bb_pct_adj", "fit",
             "cpbl_k_c", "cpbl_bb_c", "cpbl_kbb_c"]
    hcols = ["english_name", "aaa_pa", "baseline_score", "baseline_score_judgment", "aaa_woba", "aaa_k_pct",
             "aaa_iso", "cpbl_pa", "cpbl_woba", "cpbl_k_pct", "cpbl_iso"]
    lines = ["# Backtest with 2026 CPBL outcomes (amendment 7; generated by `python -m src.backtest_2026`)", "",
             "2026 outcomes: stats.cpbl.com.tw, 2026 一軍例行賽, used without written authorization "
             "(see limitations L10). 2024-25 outcomes: REBAS (資料來源：野球革命 Rebas Open Data).", "",
             f"## Pitchers (pooled n = {summ['pitchers']['n']}, by outcome season {summ['pitchers']['by_season']})", "",
             md_table(pd.DataFrame(summ["pitchers"]["tests"]), floatfmt=".2f"), "",
             "Descriptive (P1 by cohort):", "", md_table(pd.DataFrame(summ["pitchers"]["descriptive"]), floatfmt=".2f"), "",
             md_table(pc[[c for c in pcols if c in pc]].sort_values("baseline_score", ascending=False), floatfmt=".3f"), "",
             "Screened out (2026):", "", md_table(psc[psc.excluded.ne("")]), "",
             f"## Hitters (2026, n = {summ['hitters']['n']})", "",
             md_table(pd.DataFrame(summ["hitters"]["tests"]), floatfmt=".2f"), "",
             md_table(hc[[c for c in hcols if c in hc]].sort_values("baseline_score", ascending=False), floatfmt=".3f"), "",
             "Screened out:", "", md_table(hsc[hsc.excluded.ne("")]), "",
             "## AAA -> CPBL translation (amendment 7b item 3, descriptive; survivors only)", "",
             md_table(pd.DataFrame(summ["translation"]["league_ratio"]), floatfmt=".3f"), "",
             md_table(pd.DataFrame(summ["translation"]["ols"]).astype(str)), ""]
    (path("reports") / "backtest_2026.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, ensure_ascii=False, default=str))
