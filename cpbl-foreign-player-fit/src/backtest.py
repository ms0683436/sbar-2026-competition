"""Backtest (amendment 2): did pitcher Fit anticipate how foreign pitchers actually pitched in the CPBL?

Cohort: every pitcher in data/reference/cpbl_foreign_pitchers.csv (REBAS name -> MLBAM id)
who threw >= min_aaa_tracked_pitches tracked AAA pitches in the aaa_window_days before his
first CPBL appearance of an outcome season, and faced >= min_cpbl_bf CPBL batters that season.
His AAA hand must match his REBAS hand (identity check).

Predictors come only from that AAA window; outcomes only from the CPBL season:
  F1, F2, Fit   fit_score.pitcher_fit on the window pitches, environment from the OTHER
                REBAS season (leave-season-out), Fit z-scored over the pooled cohort
  AAA K-BB%, whiff%, wOBA allowed   the same window
  CPBL K-BB%, whiff%, wOBA allowed  REBAS pitch data, same definitions

Hypotheses and the reporting rule are fixed in config `backtest` (committed before any
outcome was computed). Writes data/processed/backtest_cohort.csv, backtest_screening.csv,
backtest_summary.json and reports/backtest_tables.md. Never touches the 2026 rankings.

AAA 2023 (needed for pitchers who arrived at the start of 2024) is fetched from the
statsapi feed into the shared game cache, listed in its own schedule file
(schedule_backtest.csv) so the ranking pipeline never reads it.

Usage:  python -m src.backtest
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from . import fit_score, milb_loader, rebas_loader, statsapi_pbp
from .common import ROOT, config, md_table, path, processed, raw, zscore


def cfg() -> dict:
    return config()["backtest"]


# ------------------------------------------------------------------ data
def _schedule_path():
    return raw("statsapi_pbp", "schedule_backtest.csv")


def fetch_extra(workers: int = 4) -> pd.DataFrame:
    games = [statsapi_pbp.schedule(s, e).assign(season=int(y)) for y, (s, e) in cfg()["aaa_extra_windows"].items()]
    sched = pd.concat(games, ignore_index=True)
    raw("statsapi_pbp", "games").mkdir(parents=True, exist_ok=True)
    sched.to_csv(_schedule_path(), index=False)
    todo = [pk for pk in sched.game_pk if not statsapi_pbp._game_path(pk).exists()]
    print(f"backtest: {len(sched)} extra AAA games, {len(todo)} to download")
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(statsapi_pbp.fetch_game, todo))
    return sched


def aaa_extra() -> pd.DataFrame:
    out = processed("aaa_backtest_pitches.parquet")
    if out.exists():
        return pd.read_parquet(out)
    if not _schedule_path().exists():
        fetch_extra()
    sched = pd.read_csv(_schedule_path())
    cols = milb_loader.USECOLS + statsapi_pbp.PRE_COUNT
    frames = [pd.read_csv(statsapi_pbp._game_path(pk), usecols=cols, low_memory=False)
              for pk in sched.game_pk if statsapi_pbp._game_path(pk).exists()]
    df = milb_loader.to_schema(pd.concat(frames, ignore_index=True))
    df.to_parquet(out, index=False)
    return df


def reference() -> pd.DataFrame:
    ref = pd.read_csv(ROOT / cfg()["reference"], dtype={"rebas_id": str, "mlbam_id": "Int64"}, comment="#")
    return ref


# ------------------------------------------------------------------ cohort
def _rates(df: pd.DataFrame) -> dict:
    """K-BB%, whiff% and wOBA allowed with the same definitions on both sides."""
    d = df[~df.is_bunt.astype(bool)]
    pa = df[df.pa_end.astype(bool) & df.pa_event.ne("OTHER")]
    wd = pa[pa.woba_denom > 0]
    return {"pa": len(pa), "swings": int(d.swing.sum()),
            "k_minus_bb_pct": (pa.pa_event.eq("SO").sum() - pa.pa_event.eq("uBB").sum()) / len(pa) if len(pa) else np.nan,
            "whiff_pct": d.whiff.sum() / d.swing.sum() if d.swing.sum() else np.nan,
            "woba": wd.woba_value.sum() / wd.woba_denom.sum() if len(wd) else np.nan}


def _mode(s: pd.Series):
    s = s.dropna()
    return s.mode().iat[0] if len(s) else None


def screen(ref: pd.DataFrame, cpbl: pd.DataFrame, aaa: pd.DataFrame) -> pd.DataFrame:
    """One row per (reference pitcher, outcome season) with the eligibility verdict.
    Uses CPBL batters faced, first date and hand only (no CPBL rate stat)."""
    c = cfg()
    rows = []
    for r in ref.itertuples(index=False):
        for s in c["outcome_seasons"]:
            cp = cpbl[(cpbl.season == s) & (cpbl.pitcher_id == r.rebas_id)]
            if cp.empty:
                continue
            first = pd.Timestamp(cp.date.min())
            row = {"rebas_id": r.rebas_id, "rebas_name": r.rebas_name, "english_name": r.english_name,
                   "mlbam_id": r.mlbam_id, "season": s, "cpbl_first_date": first.date(),
                   "cpbl_bf": int((cp.pa_end.astype(bool) & cp.pa_event.ne("OTHER")).sum()),
                   "cpbl_hand": _mode(cp.p_throws)}
            if pd.isna(r.mlbam_id):
                w = aaa.iloc[0:0]
            else:
                lo = first - pd.Timedelta(days=c["aaa_window_days"])
                w = aaa[(aaa.pitcher_id == str(r.mlbam_id)) & (aaa.date >= lo) & (aaa.date < first)]
            row.update(aaa_pitches=len(w), aaa_hand=_mode(w.p_throws),
                       aaa_first=w.date.min().date() if len(w) else None,
                       aaa_last=w.date.max().date() if len(w) else None)
            if pd.isna(r.mlbam_id):
                reason = "no MLBAM id"
            elif row["aaa_pitches"] < c["min_aaa_tracked_pitches"]:
                reason = f"AAA pitches in window < {c['min_aaa_tracked_pitches']}"
            elif row["cpbl_bf"] < c["min_cpbl_bf"]:
                reason = f"CPBL BF < {c['min_cpbl_bf']}"
            elif row["cpbl_hand"] is None:
                reason = "hand missing in REBAS (identity check fails)"
            elif row["aaa_hand"] != row["cpbl_hand"]:
                reason = "hand mismatch"
            else:
                reason = ""
            row["excluded"] = reason
            rows.append(row)
    sc = pd.DataFrame(rows).sort_values(["season", "rebas_name"])
    ok = sc.excluded.eq("")
    first_ok = sc[ok].drop_duplicates("rebas_id", keep="first").index      # earliest season per pitcher
    sc.loc[ok & ~sc.index.isin(first_ok), "excluded"] = "counted in an earlier season"
    return sc.reset_index(drop=True)


# ------------------------------------------------------------------ Fit + outcomes
def compute(sc: pd.DataFrame, cpbl: pd.DataFrame, aaa_env: pd.DataFrame, aaa_all: pd.DataFrame) -> pd.DataFrame:
    c = cfg()
    coh = sc[sc.excluded.eq("")].copy()
    coh["key"] = coh.mlbam_id.astype(str)
    seasons = c["outcome_seasons"]
    base = fit_score.build_tables(cpbl, aaa_env, set(), set())
    fits = []
    for s in seasons:
        sub = coh[coh.season == s]
        if sub.empty:
            continue
        other = [x for x in seasons if x != s]
        win = []
        for r in sub.itertuples(index=False):
            first = pd.Timestamp(r.cpbl_first_date)
            lo = first - pd.Timedelta(days=c["aaa_window_days"])
            win.append(aaa_all[(aaa_all.pitcher_id == r.key) & (aaa_all.date >= lo) & (aaa_all.date < first)])
        w = pd.concat(win, ignore_index=True)
        w = w.assign(cell=fit_score.assign_cell(w))
        t = dict(base, pit=fit_score._agg(w, ["pitcher_id", "season", "game_id", "cell"]))
        env = fit_score.environment(t, cpbl_seasons=other, aaa_seasons=other)
        f = fit_score.pitcher_fit(t, env)[["fit_mix_whiff", "fit_mix_damage", "tracked_pitches_used"]]
        f["env_season"] = other[0]
        aaa_r = pd.DataFrame({k: _rates(g) for k, g in w.groupby("pitcher_id")}).T.add_prefix("aaa_")
        fits.append(f.join(aaa_r))
    F = pd.concat(fits)
    coh = coh.join(F, on="key")
    coh["fit"] = fit_score.combine_pitcher(coh)
    out = []
    for r in coh.itertuples(index=False):
        cp = cpbl[(cpbl.season == r.season) & (cpbl.pitcher_id == r.rebas_id)]
        out.append({f"cpbl_{k}": v for k, v in _rates(cp).items()})
    coh = pd.concat([coh.reset_index(drop=True), pd.DataFrame(out)], axis=1)
    coh["d_whiff_pct"] = coh.cpbl_whiff_pct - coh.aaa_whiff_pct
    coh["d_woba"] = coh.cpbl_woba - coh.aaa_woba
    b = np.polyfit(coh.aaa_k_minus_bb_pct.astype(float), coh.cpbl_k_minus_bb_pct.astype(float), 1)
    coh["kbb_resid"] = coh.cpbl_k_minus_bb_pct - np.polyval(b, coh.aaa_k_minus_bb_pct.astype(float))
    b = np.polyfit(coh.aaa_whiff_pct.astype(float), coh.cpbl_whiff_pct.astype(float), 1)
    coh["whiff_resid"] = coh.cpbl_whiff_pct - np.polyval(b, coh.aaa_whiff_pct.astype(float))
    ch = config()["fit"]["combined_heuristic"]
    coh["combined"] = ch["baseline"] * zscore(coh.aaa_k_minus_bb_pct.astype(float)) + ch["fit"] * coh.fit
    return coh


def spearman(a, b) -> float:
    s = pd.concat([pd.Series(a, dtype=float), pd.Series(b, dtype=float)], axis=1).dropna()
    return float(s.iloc[:, 0].rank().corr(s.iloc[:, 1].rank())) if len(s) > 2 else np.nan


def boot_ci(df: pd.DataFrame, x: str, y: str, draws: int, ci: float, rng) -> tuple[float, float, float]:
    d = df[[x, y]].astype(float).dropna().to_numpy()
    est = spearman(d[:, 0], d[:, 1])
    vals = []
    for _ in range(draws):
        i = rng.integers(0, len(d), len(d))
        v = spearman(d[i, 0], d[i, 1])
        if not np.isnan(v):
            vals.append(v)
    a = (1 - ci) / 2
    return est, float(np.quantile(vals, a)), float(np.quantile(vals, 1 - a))


TESTS = [  # (label, predictor, outcome, expected sign)
    ("H1 F1 vs Δwhiff% (primary)", "fit_mix_whiff", "d_whiff_pct", 1),
    ("H2 F2 vs ΔwOBA", "fit_mix_damage", "d_woba", -1),
    ("H3 Fit vs K-BB% residual", "fit", "kbb_resid", 1),
    ("ref AAA K-BB% vs CPBL K-BB%", "aaa_k_minus_bb_pct", "cpbl_k_minus_bb_pct", 1),
    ("ref combined vs CPBL K-BB%", "combined", "cpbl_k_minus_bb_pct", 1),
    ("ref Fit vs CPBL K-BB%", "fit", "cpbl_k_minus_bb_pct", 1),
]

# EXPLORATORY, added after the registered results were seen: is H1 regression to the mean?
# F1 is lower for pitchers with a high AAA whiff%, and a high AAA whiff% falls in the CPBL
# whatever the league. X1 asks whether F1 still ranks CPBL whiff% once AAA whiff% is
# regressed out (residual fitted once on the full cohort, not refitted per bootstrap draw).
EXPLORATORY = [
    ("X1 F1 vs CPBL whiff% residual after AAA whiff%", "fit_mix_whiff", "whiff_resid", 1),
    ("X2 AAA whiff% vs Δwhiff% (regression to the mean)", "aaa_whiff_pct", "d_whiff_pct", -1),
    ("X3 F1 vs AAA whiff%", "fit_mix_whiff", "aaa_whiff_pct", -1),
    ("X4 F1 vs CPBL whiff% (level)", "fit_mix_whiff", "cpbl_whiff_pct", 1),
]


def verdict(n: int, lo: float, hi: float, sign: int) -> str:
    if n < cfg()["min_n_for_conclusion"]:
        return "descriptive only (n below minimum)"
    if sign * lo > 0 and sign * hi > 0:
        return "supports"
    if sign * lo < 0 and sign * hi < 0:
        return "contradicts"
    return "inconclusive"


def analyse(coh: pd.DataFrame) -> dict:
    b = cfg()["bootstrap"]
    rng = np.random.default_rng(b["seed"])
    res = []
    for label, x, y, sign in TESTS:
        est, lo, hi = boot_ci(coh, x, y, b["draws"], b["ci"], rng)
        res.append({"test": label, "n": int(coh[[x, y]].dropna().shape[0]), "rho": est,
                    "ci_lo": lo, "ci_hi": hi, "expected_sign": "+" if sign > 0 else "-",
                    "verdict": verdict(len(coh), lo, hi, sign)})
    exp = []
    for label, x, y, sign in EXPLORATORY:
        est, lo, hi = boot_ci(coh, x, y, b["draws"], b["ci"], rng)
        exp.append({"test": label, "n": int(coh[[x, y]].dropna().shape[0]), "rho": est,
                    "ci_lo": lo, "ci_hi": hi, "expected_sign": "+" if sign > 0 else "-"})
    for s in sorted(coh.season.unique()):
        d = coh[coh.season == s]
        exp.append({"test": f"X5 H1 within outcome season {s}", "n": len(d), "rho": spearman(d.fit_mix_whiff, d.d_whiff_pct),
                    "ci_lo": np.nan, "ci_hi": np.nan, "expected_sign": "+"})
    loo = [spearman(coh.drop(i).fit_mix_whiff, coh.drop(i).d_whiff_pct) for i in coh.index]
    return {"n": len(coh), "by_season": coh.season.value_counts().sort_index().to_dict(), "tests": res,
            "exploratory": exp, "h1_leave_one_out_range": [min(loo), max(loo)]}


def report(sc: pd.DataFrame, coh: pd.DataFrame, summ: dict) -> None:
    show = coh[["season", "rebas_name", "english_name", "mlbam_id", "aaa_pitches", "cpbl_bf",
                "fit_mix_whiff", "fit_mix_damage", "fit", "aaa_k_minus_bb_pct", "cpbl_k_minus_bb_pct",
                "aaa_whiff_pct", "cpbl_whiff_pct", "aaa_woba", "cpbl_woba"]].sort_values("fit", ascending=False)
    ex = sc[sc.excluded.ne("")][["season", "rebas_name", "english_name", "cpbl_bf", "aaa_pitches", "excluded"]]
    lines = ["# Backtest tables (amendment 2, generated by `python -m src.backtest`)", "",
             f"Cohort n = {summ['n']} (by outcome season: {summ['by_season']}).", "",
             "## Tests (Spearman, 90% bootstrap CI over pitchers)", "",
             md_table(pd.DataFrame(summ["tests"]), floatfmt=".2f"), "",
             "## Exploratory (added after the registered results were seen)", "",
             md_table(pd.DataFrame(summ["exploratory"]), floatfmt=".2f"), "",
             f"H1 leave-one-out range: {summ['h1_leave_one_out_range'][0]:.2f} to {summ['h1_leave_one_out_range'][1]:.2f}.", "",
             "## Cohort", "", md_table(show, floatfmt=".3f"), "",
             "## Screened out", "", md_table(ex), ""]
    (path("reports") / "backtest_tables.md").write_text("\n".join(lines), encoding="utf-8")


def run() -> dict:
    cpbl = rebas_loader.load()
    aaa_env = milb_loader.load()
    aaa_all = pd.concat([aaa_extra(), aaa_env], ignore_index=True)
    sc = screen(reference(), cpbl, aaa_all)
    sc.to_csv(processed("backtest_screening.csv"), index=False)
    coh = compute(sc, cpbl, aaa_env, aaa_all)
    coh.to_csv(processed("backtest_cohort.csv"), index=False)
    summ = analyse(coh)
    processed("backtest_summary.json").write_text(json.dumps(summ, indent=2, ensure_ascii=False, default=str))
    report(sc, coh, summ)
    return summ


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, ensure_ascii=False, default=str))
