"""Amendment 18 (B and R) — the current ranking backtested and the recruitment replay ranked by
Marcel (rules in config/scoring.yaml, amendment18; rule S, Marcel-style Stuff+, is in stuff_rank).

  M1  0.5 z(Marcel K-BB%) + 0.2 z(Stuff+) + 0.3 Fit, amendment-11 cohort, pre-arrival inputs, each z
      against amendment 11's reference pool R(Y); Spearman with centred CPBL K-BB% (2025-26 arrivals)
  M2  the same without Fit (5/7, 2/7), all 2024-26 arrivals
  R   amendment-6 folds, lists ranked by Marcel K-BB% (starter pool; starter + conversion pool with
      the amendment-17 starter-equivalent adjustment)

Nothing here changes the ranking or the picks.

Usage:  python -m src.a18_checks
"""
from __future__ import annotations

import functools
import json

import numpy as np
import pandas as pd

from . import backtest_chrono, conversion_pool, marcel, milb_loader, recruitment_backtest, signability, stuff_rank
from .common import config, md_table, path, processed


def cfg() -> dict:
    return config()["amendment18"]


def _rng(k: int = 0):
    return np.random.default_rng(cfg()["backtest"]["bootstrap"]["seed"] + k)


# ------------------------------------------------------------------ B: backtest of the current ranking
@functools.lru_cache(maxsize=None)
def _inputs():
    from . import backtest, fit_score, rebas_loader
    aaa_env = milb_loader.load()
    return rebas_loader.load(), aaa_env, pd.concat([backtest.aaa_extra(), aaa_env], ignore_index=True)


def _pool(y: int) -> pd.DataFrame:
    """Amendment 11's reference pool R(Y): AAA Y-1 pitchers with >= 100 BF and >= 400 tracked pitches,
    tracked as amendment 11 counts them (non-bunt pitches in a Fit cell). For Fit arrival seasons
    the membership is amendment 11's own (backtest_chrono.fit_scores with an empty cohort); for
    earlier seasons the same count without the Fit environment."""
    from . import fit_score
    c = backtest_chrono.cfg()
    bl = backtest_chrono.pitcher_baseline(y - 1)
    cpbl, aaa_env, aaa_all = _inputs()
    if y in c["fit_arrival_seasons"]:
        empty = pd.DataFrame(columns=["mlbam_id", "season", "source"])
        _, fs = backtest_chrono.fit_scores(y, empty, cpbl, aaa_env, aaa_all, {})
        ids = set(fs["pool"].index.astype(int))
    else:
        a = aaa_all[aaa_all.season == y - 1]
        a = a[~a.is_bunt.astype(bool) & fit_score.assign_cell(a).notna()]
        n = a.groupby("pitcher_id").size()
        ids = set(n[n >= c["pool"]["min_tracked_pitches"]].index.astype(int))
    return bl[bl.mlbam_id.isin(ids)]


def backtest() -> tuple[pd.DataFrame, dict]:
    coh = pd.read_csv(processed("marcel_v2_a11.csv"))
    w = config()["amendment13"]["weights"]
    clip = config()["amendment13"]["stuff_clip"]
    g = stuff_rank.game_sums()
    parts, info = [], {}
    for y, d in coh.groupby("season"):
        y = int(y)
        pool = _pool(y)
        m = marcel.project("pitcher", y, seasons=[y - 1, y - 2, y - 3]).set_index("player_id").score_c
        st = stuff_rank.candidate_stuff(g, seasons=[y - 1, y - 2, y - 3]).set_index("pitcher_id").stuff_plus
        pm, ps = pool.mlbam_id.map(m), pool.mlbam_id.map(st)
        d = d.copy()
        d["stuff_a18"] = d.mlbam_id.map(st)
        d["z_marcel"] = backtest_chrono._z(d.marcel, pm.dropna())
        d["z_stuff"] = backtest_chrono._z(d.stuff_a18, ps.dropna(), clip).fillna(0.0)
        d["m1"] = w["baseline"] * d.z_marcel + w["stuff"] * d.z_stuff + w["fit"] * d.fit
        d["m2"] = (w["baseline"] * d.z_marcel + w["stuff"] * d.z_stuff) / (w["baseline"] + w["stuff"])
        info[str(y)] = {"pool": int(len(pool)), "pool_with_stuff": int(ps.notna().sum()),
                        "stuff_seasons_with_data": sorted(int(s) for s in g.season.unique() if s in (y - 1, y - 2, y - 3))}
        parts.append(d)
    coh = pd.concat(parts, ignore_index=True)
    rng = _rng()
    fc = coh[coh.season.isin(backtest_chrono.cfg()["fit_arrival_seasons"])].dropna(subset=["m1", "combined", "cpbl_kbb_c"])
    t = backtest_chrono._corr_test
    pd_ = backtest_chrono.paired_diff
    out = {"pools": info,
           "tests": [t("M1 Marcel + Stuff+ + Fit vs centred CPBL K-BB% (2025-26 arrivals; primary)", fc, "m1", "cpbl_kbb_c", rng),
                     t("C1 amendment-11 combined (old baseline + Fit), same pitchers", fc, "combined", "cpbl_kbb_c", rng),
                     t("Marcel alone, same pitchers", fc, "z_marcel", "cpbl_kbb_c", rng),
                     t("Stuff+ (rule S) alone, same pitchers", fc, "z_stuff", "cpbl_kbb_c", rng),
                     t("M2 Marcel + Stuff+ (no Fit) vs centred CPBL K-BB% (all 2024-26 arrivals)", coh, "m2", "cpbl_kbb_c", rng)],
           "paired": [pd_(fc, "m1", "combined", "cpbl_kbb_c", rng) | {"test": "M1 minus C1"},
                      pd_(fc, "m1", "z_marcel", "cpbl_kbb_c", rng) | {"test": "M1 minus Marcel alone"}]}
    return coh, out


# ------------------------------------------------------------------ R: recruitment replay ranked by Marcel
def replay() -> dict:
    rng = _rng(1)
    res = {}
    for name, f in config()["amendment6"]["backtest"]["folds"].items():
        s = recruitment_backtest.stints(f["seasons"])
        app = conversion_pool.appearances(recruitment_backtest._pitches(f["seasons"]))
        eff = conversion_pool.role_effect(app, rng)
        usage = conversion_pool.pitcher_usage(app)
        df = conversion_pool.pools(s, usage)
        pos = recruitment_backtest.positives(f["positives"])
        reach = set(pos.mlbam_id) & set(s.loc[s.pitching_BF > 0, "player_id"].astype(int))
        t = signability.tiers(df.mlbam_id, f["as_of"])
        F = int(max(f["seasons"])) + 1
        r = {"as_of": f["as_of"], "forecast": F, "primary": f["primary"], "positives_with_aaa": len(reach)}
        for arm in cfg()["replay_arms"]:
            scale = None
            if arm == "expanded_adj":
                conv = df[df.conversion_pool]
                scale = conversion_pool.role_scale(conv.mlbam_id, eff, conv.set_index("mlbam_id").relief_share, forecast=F)
            m = marcel.project("pitcher", F, seasons=[F - 1, F - 2, F - 3], scale=scale).set_index("player_id").score
            d = df.assign(baseline_score=df.mlbam_id.map(m)).merge(t[["mlbam_id", "sign_tier", "signable"]],
                                                                  on="mlbam_id", how="left")
            if arm == "starter":
                d = d[d.starter_pool]
            miss = d.baseline_score.isna()           # no Marcel line: not listed, counted
            r[f"{arm}_no_marcel"] = int(miss.sum())
            r[f"{arm}_positives_no_marcel"] = int(d[miss].mlbam_id.isin(pos.mlbam_id).sum())
            d = d[~miss]
            keep = d.signable.fillna(False).astype(bool)
            out, hit = recruitment_backtest.evaluate_arm(d.assign(name=d.mlbam_id.map(pos.set_index("mlbam_id").name)),
                                                         keep, set(pos.mlbam_id), len(reach),
                                                         config()["amendment6"]["backtest"]["ks"])
            out["hits"] = hit.to_dict(orient="records")
            r[arm] = out
        res[name] = r
    return res


def published_replay() -> dict:
    """The baseline-ranked arms already published (amendment 9), for the side-by-side table."""
    s = json.loads(processed(conversion_pool.FROZEN).read_text())
    return {k: {arm: {x: v[arm][x] for x in ("N", "positives_in_list", "hits@10", "random@10", "hits@20", "random@20",
                                                "median_percentile")} for arm in cfg()["replay_arms"]}
            for k, v in s["folds"].items()}


def run() -> dict:
    coh, b = backtest()
    out = {"backtest": b, "replay": replay(), "replay_baseline_published": published_replay(),
           "stuff_reliability": stuff_rank.reliability()}
    coh.to_csv(processed("a18_backtest_cohort.csv"), index=False)
    processed("a18_summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float))
    report(out)
    return out


def report(out: dict) -> None:
    b, r = out["backtest"], out["stuff_reliability"]
    L = ["# Amendment 18: the current ranking backtested, Marcel recruitment replay (generated by "
         "`python -m src.a18_checks`)", "",
         f"Stuff+ (rule S): seasons weighted 3 / 2 / 1, regressed toward 100 by R = {r['R']:.0f} pitches (split-half r "
         f"{r['r']:.3f}, {r['nbar']:.0f} pitches per half, {r['n_pitchers']} pitchers, AAA 2025).", "",
         "## B. Backtest (amendment-11 cohort, pre-arrival inputs only)", "",
         "z against amendment 11's reference pool R(Y). The Stuff+ model was trained on AAA 2025 (other pitchers' AAA "
         "run values, not CPBL outcomes), so 2025 arrivals use a model that saw later AAA data.", "",
         md_table(pd.DataFrame(b["tests"]), ".3f"), "", "Paired differences (bootstrap over pitchers):", "",
         md_table(pd.DataFrame(b["paired"]), ".3f"), "",
         "Reference pools: " + "; ".join(f"{y}: {v['pool']} pitchers, Stuff+ seasons {v['stuff_seasons_with_data']}"
                                          for y, v in b["pools"].items()) + ".", "",
         "## R. Recruitment replay, lists ranked by Marcel K-BB% (exploratory: the positives were seen before)", ""]
    for k, f in out["replay"].items():
        old = out["replay_baseline_published"].get(k, {})
        rows = []
        for arm in cfg()["replay_arms"]:
            for lab, x in (("Marcel", f[arm]), ("baseline (published)", old.get(arm))):
                if x:
                    rows.append({"arm": arm, "ranked by": lab, "list N": x["N"], "positives in list": x["positives_in_list"],
                                 "hits@10": x["hits@10"], "random@10": x["random@10"], "hits@20": x["hits@20"],
                                 "random@20": x["random@20"], "median pct rank": x["median_percentile"]})
        L += [f"### Fold {k} ({'primary' if f['primary'] else 'exploratory'}; as of {f['as_of']}, Marcel forecast "
              f"{f['forecast']}; positives with AAA {f['positives_with_aaa']})", "", md_table(pd.DataFrame(rows), ".2f"), "",
              "Coverage: pool pitchers without a Marcel line (dropped) " + "; ".join(
                  f"{arm} {f[arm + '_no_marcel']} (positives {f[arm + '_positives_no_marcel']})"
                  for arm in cfg()["replay_arms"]) + ".", ""]
        hits = pd.DataFrame([h | {"arm": arm} for arm in cfg()["replay_arms"] for h in f[arm]["hits"]])
        if len(hits):
            L += [md_table(hits.pivot_table(index="name", columns="arm", values="list_rank").reset_index(), ".0f"), ""]
    (path("reports") / "a18_checks.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    print(json.dumps(run(), indent=1, ensure_ascii=False, default=float)[:3000])
