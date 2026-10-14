"""Amendment 11 — the pitcher backtest rebuilt as a historical forecast.

Same cohort and outcomes as amendment 7 (pooled 2024-26 arrivals), but every predictor of a
pitcher arriving in season Y uses only information available before Y (config `amendment11`,
committed before anything here was computed):

  baseline   amendment-5 pitcher baseline on AAA Y-1, park factors from AAA Y-1 only
  Fit        environment CPBL REBAS seasons < Y vs AAA of the same seasons; 2024 arrivals have
             no earlier REBAS season, so no Fit
  scale      F1, F2 and baseline z-scored against R(Y) = AAA Y-1 pitchers with >= 100 BF and
             >= 400 tracked pitches, not against the backtest cohort
  forecasts  fitted only on arrivals before Y (rolling origin)

Nothing here changes the ranking or the picks.

Usage:  python -m src.backtest_chrono
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import backtest, backtest_2026, baseline, baseline_weights, fit_score, milb_candidates, milb_loader, park
from . import rebas_loader, recruitment_backtest
from .common import config, md_table, path, processed, zscore


def cfg() -> dict:
    return config()["amendment11"]


# ------------------------------------------------------------------ predictors
def pitcher_baseline(year: int) -> pd.DataFrame:
    """backtest_2026.pitcher_baseline with park factors from AAA `year` alone."""
    pf = park.factors_for([year], recruitment_backtest._pitches([year]))
    s = baseline_weights.fetch("pitching", year, pf=pf)
    df = milb_candidates._combine(s, [c for c in milb_candidates.PITCH_SUM if c in s] + ["outs"], "pitching_BF")
    df["IP"] = df.outs / 3
    df = df[df.pitching_BF >= cfg()["pool"]["min_bf"]]
    empty = pd.DataFrame(columns=["player_id", "pitching_BF", "pitching_SO", "pitching_BB"])
    d = baseline.pitcher_metrics(df, year, s, empty)
    d = baseline.score(d, "pitcher", baseline.judgment_weights("pitcher"))
    out = d[["player_id", "pitching_BF", "baseline_score", "k_pct_adj", "bb_pct_adj"]].copy()
    out["mlbam_id"] = out.player_id.astype(int)
    return out.drop(columns="player_id").rename(columns={"pitching_BF": "aaa_bf"})


def _z(x: pd.Series, ref: pd.Series, clip: float | None = None) -> pd.Series:
    z = (x - ref.mean()) / ref.std(ddof=0)
    return z.clip(-clip, clip) if clip else z


def fit_scores(y: int, coh: pd.DataFrame, cpbl: pd.DataFrame, aaa_env: pd.DataFrame,
               aaa_all: pd.DataFrame, first: dict) -> tuple[pd.DataFrame, dict]:
    """F1 / F2 of arrival season `y`'s cohort (their pre-arrival window) and of the reference
    pool (AAA y-1), both under the environment of REBAS seasons < y."""
    c = cfg()
    env_seasons = [s for s in c["rebas_seasons"] if s < y]
    a_prev = aaa_all[aaa_all.season == y - 1]
    win = []
    for r in coh.itertuples(index=False):
        if r.source == "REBAS":                          # amendment-2 window: 365 days before the first game
            f = pd.Timestamp(first[(int(r.mlbam_id), int(r.season))])
            lo = f - pd.Timedelta(days=backtest.cfg()["aaa_window_days"])
            w = aaa_all[(aaa_all.pitcher_id == str(r.mlbam_id)) & (aaa_all.date >= lo) & (aaa_all.date < f)]
        else:                                            # amendment 7: AAA 2025 season
            w = a_prev[a_prev.pitcher_id == str(r.mlbam_id)]
        win.append(w.assign(pitcher_id="c" + str(r.mlbam_id)))
    pit = pd.concat(win + [a_prev], ignore_index=True)   # cohort ids prefixed "c" so pool rows stay apart
    pit = pit.assign(cell=fit_score.assign_cell(pit))
    base = fit_score.build_tables(cpbl, aaa_env, set(), set())
    t = dict(base, pit=fit_score._agg(pit, ["pitcher_id", "season", "game_id", "cell"]))
    env = fit_score.environment(t, cpbl_seasons=env_seasons, aaa_seasons=env_seasons)
    f = fit_score.pitcher_fit(t, env)[["fit_mix_whiff", "fit_mix_damage", "tracked_pitches_used"]]
    cf = f[f.index.str.startswith("c")].rename(index=lambda i: int(i[1:]))
    pf = f[~f.index.str.startswith("c")]
    pf = pf[pf.tracked_pitches_used >= c["pool"]["min_tracked_pitches"]]
    pf.index = pf.index.astype(int)
    return cf, {"pool": pf, "env_seasons": env_seasons}


def cohort() -> tuple[pd.DataFrame, dict]:
    c = cfg()
    coh = pd.read_csv(processed("backtest2026_pitcher_cohort.csv"))
    keep = ["season", "english_name", "rebas_name", "mlbam_id", "source", "cpbl_kbb_c", "cpbl_k_c", "cpbl_bb_c",
            "d_whiff_c", "baseline_score", "fit", "fit_mix_whiff"]
    coh = coh[keep].rename(columns={"baseline_score": "baseline_retro", "fit": "fit_retro", "fit_mix_whiff": "f1_retro"})
    coh = coh[coh.season.isin(c["arrival_seasons"])].copy()
    a2 = pd.read_csv(processed("backtest_cohort.csv"))
    first = {(int(r.mlbam_id), int(r.season)): r.cpbl_first_date for r in a2.itertuples(index=False)}
    cpbl = rebas_loader.load()
    aaa_env = milb_loader.load()
    aaa_all = pd.concat([backtest.aaa_extra(), aaa_env], ignore_index=True)
    parts, pools = [], {}
    for y, g in coh.groupby("season"):
        y = int(y)
        bl = pitcher_baseline(y - 1)
        g = g.merge(bl, on="mlbam_id", how="left")
        info = {"aaa_season": y - 1, "baseline_pool": int(len(bl))}
        if y in c["fit_arrival_seasons"]:
            cf, fs = fit_scores(y, g, cpbl, aaa_env, aaa_all, first)
            pool = fs["pool"].join(bl.set_index("mlbam_id")[["baseline_score"]], how="inner")
            g = g.join(cf[["fit_mix_whiff", "fit_mix_damage", "tracked_pitches_used"]], on="mlbam_id")
            w, clip = config()["fit"]["pitcher"]["components"], c["fit_clip"]
            pz = w["mix_whiff"] * _z(pool.fit_mix_whiff, pool.fit_mix_whiff, clip) + \
                w["mix_damage"] * _z(pool.fit_mix_damage, pool.fit_mix_damage, clip)
            g["fit"] = (w["mix_whiff"] * _z(g.fit_mix_whiff, pool.fit_mix_whiff, clip)
                        + w["mix_damage"] * _z(g.fit_mix_damage, pool.fit_mix_damage, clip))
            g["z_baseline"] = _z(g.baseline_score, pool.baseline_score)
            ch = config()["fit"]["combined_heuristic"]
            g["combined"] = ch["baseline"] * g.z_baseline + ch["fit"] * g.fit
            info |= {"env_seasons": fs["env_seasons"], "reference_pool": int(len(pool)),
                     "pool_fit_mean": float(pz.mean()), "pool_fit_sd": float(pz.std(ddof=0))}
        parts.append(g)
        pools[str(y)] = info
    return pd.concat(parts, ignore_index=True), pools


# ------------------------------------------------------------------ tests
def _verdict(n: int, lo: float, hi: float) -> str:
    if n < cfg()["min_n_for_conclusion"]:
        return "descriptive only (n below minimum)"
    return "supports" if lo > 0 else "contradicts" if hi < 0 else "inconclusive"


def _corr_test(label: str, d: pd.DataFrame, x: str, y: str, rng) -> dict:
    b = cfg()["bootstrap"]
    d = d[[x, y]].astype(float).dropna()
    est, lo, hi = backtest.boot_ci(d, x, y, b["draws"], b["ci"], rng)
    return {"test": label, "n": len(d), "rho": est, "ci_lo": lo, "ci_hi": hi, "verdict": _verdict(len(d), lo, hi)}


def paired_diff(d: pd.DataFrame, a: str, b_: str, y: str, rng) -> dict:
    """rho(a, y) - rho(b_, y) with a paired bootstrap over pitchers."""
    b = cfg()["bootstrap"]
    m = d[[a, b_, y]].astype(float).dropna().to_numpy()

    def stat(x):
        return backtest.spearman(x[:, 0], x[:, 2]) - backtest.spearman(x[:, 1], x[:, 2])

    vals = [v for v in (stat(m[rng.integers(0, len(m), len(m))]) for _ in range(b["draws"])) if not np.isnan(v)]
    q = (1 - b["ci"]) / 2
    lo, hi = float(np.quantile(vals, q)), float(np.quantile(vals, 1 - q))
    return {"test": "C3 C1 minus C2 (does Fit add to baseline)", "n": len(m), "rho": stat(m), "ci_lo": lo,
            "ci_hi": hi, "verdict": _verdict(len(m), lo, hi)}


def rolling_forecast(coh: pd.DataFrame, rng) -> dict:
    """R1: centred CPBL K-BB% of each origin season's arrivals, models fitted on earlier arrivals."""
    b = cfg()["bootstrap"]
    rows = []
    for y in cfg()["forecast_origins"]:
        tr = coh[(coh.season < y)].dropna(subset=["baseline_score", "cpbl_kbb_c"])
        te = coh[(coh.season == y)].dropna(subset=["baseline_score", "cpbl_kbb_c"])
        k, a = np.polyfit(tr.baseline_score, tr.cpbl_kbb_c, 1)
        for r in te.itertuples(index=False):
            rows.append({"origin": y, "english_name": r.english_name, "n_train": len(tr), "actual": r.cpbl_kbb_c,
                         "m0": tr.cpbl_kbb_c.mean(), "m1": a + k * r.baseline_score, "m1_slope": k})
    f = pd.DataFrame(rows)
    e0, e1 = (f.actual - f.m0).abs().to_numpy(), (f.actual - f.m1).abs().to_numpy()
    diffs = []
    for _ in range(b["draws"]):
        i = rng.integers(0, len(f), len(f))
        diffs.append(e1[i].mean() - e0[i].mean())
    q = (1 - b["ci"]) / 2
    return {"n_held_out": len(f), "mae_m0": float(e0.mean()), "mae_m1": float(e1.mean()),
            "m1_minus_m0": float(e1.mean() - e0.mean()),
            "ci": [float(np.quantile(diffs, q)), float(np.quantile(diffs, 1 - q))],
            "slopes": {str(y): float(g.m1_slope.iat[0]) for y, g in f.groupby("origin")},
            "rows": f.to_dict(orient="records")}


def analyse(coh: pd.DataFrame) -> dict:
    rng = np.random.default_rng(cfg()["bootstrap"]["seed"])
    fc = coh[coh.season.isin(cfg()["fit_arrival_seasons"]) & coh.fit.notna()]
    tests = [_corr_test("C1 combined vs centred CPBL K-BB% (primary)", fc, "combined", "cpbl_kbb_c", rng),
             _corr_test("C2 baseline vs centred CPBL K-BB% (C1 pitchers)", fc, "z_baseline", "cpbl_kbb_c", rng),
             paired_diff(fc, "combined", "z_baseline", "cpbl_kbb_c", rng),
             _corr_test("C4 Fit vs centred CPBL K-BB%", fc, "fit", "cpbl_kbb_c", rng),
             _corr_test("C5 F1 vs centred whiff% change", fc, "fit_mix_whiff", "d_whiff_c", rng),
             _corr_test("C6 baseline vs centred CPBL K-BB% (all arrivals)", coh, "baseline_score", "cpbl_kbb_c", rng),
             _corr_test("C7 AAA K% vs centred CPBL K% (all arrivals)", coh, "k_pct_adj", "cpbl_k_c", rng),
             _corr_test("C8 AAA BB% vs centred CPBL BB% (all arrivals)", coh, "bb_pct_adj", "cpbl_bb_c", rng)]
    return {"tests": tests, "forecast": rolling_forecast(coh, rng), "exploratory": exploratory(coh, fc),
            "input_change": {"baseline_rank_corr_retro_vs_chrono": backtest.spearman(coh.baseline_retro, coh.baseline_score),
                             "fit_rank_corr_retro_vs_chrono_2025_26": backtest.spearman(fc.fit_retro, fc.fit)}}


def exploratory(coh: pd.DataFrame, fc: pd.DataFrame) -> dict:
    """Not registered (added after C1 was seen): how fragile is C1?"""
    sp = backtest.spearman
    retro = 0.7 * zscore(coh.baseline_retro) + 0.3 * coh.fit_retro          # amendment-7 scale (cohort z)
    r9 = retro[fc.index]
    return {"baseline_fit_corr": sp(fc.z_baseline, fc.fit),
            "c1_by_baseline_weight": {f"{w:.1f}": sp(w * fc.z_baseline + (1 - w) * fc.fit, fc.cpbl_kbb_c)
                                      for w in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)},
            "c1_leave_one_out": [min(sp(fc.drop(i).combined, fc.drop(i).cpbl_kbb_c) for i in fc.index),
                                 max(sp(fc.drop(i).combined, fc.drop(i).cpbl_kbb_c) for i in fc.index)],
            "c1_2025_only": sp(fc[fc.season == 2025].combined, fc[fc.season == 2025].cpbl_kbb_c),
            "retro_combined_same_pitchers": sp(r9, fc.cpbl_kbb_c),
            "retro_combined_all_17": sp(retro, coh.cpbl_kbb_c)}


# ------------------------------------------------------------------ run
def run() -> dict:
    coh, pools = cohort()
    out = analyse(coh) | {"n": len(coh), "by_season": {str(k): int(v) for k, v in coh.season.value_counts().sort_index().items()},
                          "pools": pools}
    retro = json.loads(processed("backtest2026_summary.json").read_text())["pitchers"]["tests"]
    out["retrospective"] = retro
    coh.to_csv(processed("backtest_chrono_cohort.csv"), index=False)
    processed("backtest_chrono_summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    report(coh, out)
    return out


def report(coh: pd.DataFrame, out: dict) -> None:
    f = out["forecast"]
    cols = ["season", "english_name", "baseline_score", "baseline_retro", "fit", "fit_retro", "combined",
            "cpbl_kbb_c", "cpbl_k_c", "cpbl_bb_c"]
    L = ["# Chronological pitcher backtest (amendment 11; generated by `python -m src.backtest_chrono`)", "",
         "Every predictor of a pitcher arriving in season Y uses only data from before Y: AAA Y-1 (or the "
         "365-day window before his first CPBL game), CPBL REBAS seasons before Y, and earlier arrivals' "
         "outcomes. 2024 arrivals have no chronological Fit (no REBAS season before 2024). The cohort and its "
         "outcomes were already known (amendments 2 and 7), so this is a stricter re-analysis, not a fresh "
         "confirmation. 2024-25 outcomes: 資料來源：野球革命 Rebas Open Data; 2026: stats.cpbl.com.tw "
         "(limitations L10).", "",
         f"Cohort n = {out['n']} (by arrival season {out['by_season']}).", "",
         "Reference pools: " + "; ".join(f"{y}: AAA {p['aaa_season']}, baseline pool {p['baseline_pool']}"
                                         + (f", Fit env REBAS {p['env_seasons']}, reference pool {p['reference_pool']}"
                                            if "reference_pool" in p else ", no Fit")
                                         for y, p in out["pools"].items()), "",
         "## Registered tests (Spearman, 90% bootstrap CI)", "", md_table(pd.DataFrame(out["tests"]), ".2f"), "",
         "## R1 rolling-origin forecast of centred CPBL K-BB%", "",
         f"Held-out arrivals {f['n_held_out']} (2025 trained on 2024; 2026 trained on 2024-25). "
         f"MAE M0 (training mean) {f['mae_m0']:.4f}, M1 (OLS on baseline) {f['mae_m1']:.4f}; "
         f"M1 - M0 = {f['m1_minus_m0']:+.4f} (90% CI {f['ci'][0]:+.4f} to {f['ci'][1]:+.4f}; resampling held-out "
         f"pitchers only, models fixed). M1 slopes: {', '.join(f'{k}: {v:.3f}' for k, v in f['slopes'].items())}.", "",
         md_table(pd.DataFrame(f["rows"]).drop(columns="m1_slope"), ".3f"), "",
         "## Exploratory (not registered; added after C1 was seen)", "",
         f"Baseline vs Fit on the C1 pitchers: rho {out['exploratory']['baseline_fit_corr']:.2f}. "
         "C1 by baseline weight (Fit = 1 - weight): "
         + ", ".join(f"{k}: {v:.2f}" for k, v in out["exploratory"]["c1_by_baseline_weight"].items())
         + f". Leave-one-out range {out['exploratory']['c1_leave_one_out'][0]:.2f} to "
         f"{out['exploratory']['c1_leave_one_out'][1]:.2f}; 2025 arrivals only (n 7) "
         f"{out['exploratory']['c1_2025_only']:.2f}. Amendment-7 combined (cohort z-scores) on the same pitchers "
         f"{out['exploratory']['retro_combined_same_pitchers']:.2f}, on all 17 {out['exploratory']['retro_combined_all_17']:.2f}.", "",
         "## Retrospective version (amendment 7, for comparison)", "",
         md_table(pd.DataFrame(out["retrospective"]), ".2f"), "",
         f"Rank correlation of inputs, retrospective vs chronological: baseline "
         f"{out['input_change']['baseline_rank_corr_retro_vs_chrono']:.2f}, Fit (2025-26) "
         f"{out['input_change']['fit_rank_corr_retro_vs_chrono_2025_26']:.2f}.", "",
         "## Cohort", "", md_table(coh[[c for c in cols if c in coh]].sort_values(["season", "english_name"]), ".3f"), ""]
    (path("reports") / "backtest_chrono.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    r = run()
    print(json.dumps({k: v for k, v in r.items() if k not in ("retrospective",)}, indent=1, ensure_ascii=False, default=str))
