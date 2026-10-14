"""Amendment 9 — AAA reliever conversion pool (config `amendment9`, committed before any result).

CPBL clubs use import pitchers as starters, but many imports were AAA relievers. Pitchers who
miss the amendment-6 starter pool but threw >= 400 tracked AAA pitches over the two pool
seasons form a separate conversion pool:

  role effect   within-pitcher (relief rate - start rate) of SO / BF, BB / BF and HR / BF over
                pitcher-seasons with >= 30 BF in both roles, harmonic-mean BF weights
  adjustment    counts minus relief BF x role effect (one adjustment; no extra TTO term)
  score         amendment-5 baseline, each metric's z against the STARTER pool's mean and SD
  flags         AAA GS and the largest single-appearance pitch count (stretched / partial / none)

The recruitment replay compares three arms on the amendment-6 folds; the CPBL cohort check
applies the adjustment to the amendment-7 pooled cohort. The starter-pool picks do not change.

Usage:  python -m src.conversion_pool
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import (backtest, backtest_2026, baseline, baseline_weights, eligibility, milb_candidates, park,
               recruitment_backtest, signability)
from .common import config, md_table, path, processed

RATES = {"SO": ["SO"], "BB": ["uBB", "IBB"], "HR": ["HR"]}
COUNT = {"SO": "pitching_SO", "BB": "pitching_BB", "HR": "pitching_HR"}


def cfg() -> dict:
    return config()["amendment9"]


# ------------------------------------------------------------------ pitch-level appearances
def appearances(a: pd.DataFrame) -> pd.DataFrame:
    """One row per pitcher x game: role, pitches, BF and SO / BB / HR."""
    pa = a[a.pa_end.astype(bool) & a.pa_event.ne("OTHER")]
    g = ["season", "pitcher_id", "game_id"]
    out = a.groupby(g).agg(role=("pitcher_role", "first"), pitches=("pitch_seq", "size"))
    out["bf"] = pa.groupby(g).size()
    for k, ev in RATES.items():
        out[k] = pa[pa.pa_event.isin(ev)].groupby(g).size()
    return out.fillna(0).reset_index()


def role_effect(app: pd.DataFrame, rng) -> dict:
    """Relief minus start rate, within pitcher-season, harmonic-mean BF weights; bootstrap CI."""
    c = cfg()
    s = app.groupby(["season", "pitcher_id", app.role.eq("SP").map({True: "SP", False: "RP"})])[
        ["bf", *RATES]].sum().unstack("role")
    s = s[(s[("bf", "SP")] >= c["role_effect"]["min_bf_each_role"]) & (s[("bf", "RP")] >= c["role_effect"]["min_bf_each_role"])]
    w = 2 / (1 / s[("bf", "SP")] + 1 / s[("bf", "RP")])
    d = pd.DataFrame({k: s[(k, "RP")] / s[("bf", "RP")] - s[(k, "SP")] / s[("bf", "SP")] for k in RATES})
    b = c["bootstrap"]
    a_ = (1 - b["ci"]) / 2
    out = {"n_pitcher_seasons": int(len(d))}
    for k in RATES:
        est = float(np.average(d[k], weights=w))
        boots = []
        for _ in range(b["draws"]):
            i = rng.integers(0, len(d), len(d))
            boots.append(np.average(d[k].iloc[i], weights=w.iloc[i]))
        out[k] = {"effect": est, "ci": [float(np.quantile(boots, a_)), float(np.quantile(boots, 1 - a_))]}
    return out


def pitcher_usage(app: pd.DataFrame) -> pd.DataFrame:
    """Per pitcher over the given appearances: tracked pitches, relief BF share, max pitches."""
    g = app.groupby("pitcher_id")
    out = pd.DataFrame({"tracked_pitches": g.pitches.sum(), "max_game_pitches": g.pitches.max(),
                        "tracked_starts": g.role.apply(lambda r: int(r.eq("SP").sum()))})
    bf = app.groupby(["pitcher_id", app.role.eq("SP")]).bf.sum().unstack(fill_value=0)
    out["relief_share"] = (bf.get(False, 0) / bf.sum(axis=1)).reindex(out.index).fillna(0)
    out.index = out.index.astype(int)
    return out


def starting_evidence(max_pitches: pd.Series) -> pd.Series:
    f = cfg()["flags"]
    return pd.Series(np.select([max_pitches >= f["stretched_min_pitches"], max_pitches >= f["partial_min_pitches"]],
                               ["stretched", "partial"], "none"), index=max_pitches.index)


def adjust_counts(df: pd.DataFrame, eff: dict) -> pd.DataFrame:
    """Starter-equivalent counts: minus (relief share x season BF) x role effect."""
    df = df.copy()
    rbf = df.relief_share.fillna(0) * df.pitching_BF
    for k, col in COUNT.items():
        df[col] = (df[col] - rbf * eff[k]["effect"]).clip(lower=0)
    return df


# ------------------------------------------------------------------ pools and scores
def pools(s: pd.DataFrame, usage: pd.DataFrame) -> pd.DataFrame:
    """Two-season totals with starter / conversion membership (amendment 6 rule 1, amendment 9 rule 1)."""
    g, c = config()["amendment6"]["pool"]["pitcher"], cfg()["pool"]
    df = milb_candidates._combine(s, [x for x in milb_candidates.PITCH_SUM if x in s] + ["outs"], "pitching_BF")
    df["IP"] = df.outs / 3
    df = milb_candidates._latest(s, df, ["pitching_GS", "pitching_IP"])
    df["mlbam_id"] = df.player_id.astype(int)
    df = df.join(usage, on="mlbam_id")
    df["tracked_pitches"] = df.tracked_pitches.fillna(0)
    df["starter_pool"] = (df.pitching_GS >= g["min_gs"]) & (df.IP >= g["min_ip"]) & (df.latest_pitching_IP >= g["latest_min_ip"])
    df["conversion_pool"] = (~df.starter_pool & (df.tracked_pitches >= c["min_tracked_pitches"])
                             & (df.latest_pitching_IP >= c["latest_min_ip"]))
    df["starting_evidence"] = starting_evidence(df.max_game_pitches.fillna(0))
    return df[df.starter_pool | df.conversion_pool].reset_index(drop=True)


def score_vs(d: pd.DataFrame, ref: pd.DataFrame) -> pd.Series:
    """Amendment-5 baseline with each metric's z against the mean and SD of `ref` (the
    unadjusted starter pool)."""
    spec, clip = config()["baseline"]["pitcher"], config()["baseline"]["winsor_z"]
    w = baseline.judgment_weights("pitcher")
    total = 0
    for m, sp in spec.items():
        r = pd.to_numeric(ref[m])
        z = ((pd.to_numeric(d[m]) - r.mean()) / r.std(ddof=0)).clip(-clip, clip)
        total = total + w[m] * sp["sign"] * z
    return total


def _metrics(df: pd.DataFrame, s: pd.DataFrame, year: int) -> pd.DataFrame:
    empty = pd.DataFrame(columns=["player_id", "pitching_BF", "pitching_SO", "pitching_BB"])
    return baseline.pitcher_metrics(df, year, s, empty)


def scored(df: pd.DataFrame, s: pd.DataFrame, year: int, eff: dict | None,
           ref: pd.Series | None = None) -> pd.DataFrame:
    """Baseline for starter + conversion rows; conversion rows adjusted when eff is given.
    z reference: the unadjusted rows in `ref` (default: the starter pool)."""
    ref = df.starter_pool if ref is None else ref
    base = _metrics(df, s, year)
    df = df.copy()
    conv = df.conversion_pool.to_numpy()
    if eff is not None:
        df.loc[conv] = adjust_counts(df.loc[conv], eff)
    d = _metrics(df, s, year)
    d["baseline_score"] = score_vs(d, base[ref.to_numpy()])
    return d


# ------------------------------------------------------------------ recruitment replay
def run_fold(name: str, f: dict, rng) -> dict:
    s = recruitment_backtest.stints(f["seasons"])
    a = recruitment_backtest._pitches(f["seasons"])
    app = appearances(a)
    eff = role_effect(app, rng)
    df = pools(s, pitcher_usage(app))
    pos = recruitment_backtest.positives(f["positives"])
    reach = set(pos.mlbam_id) & set(s.loc[s.pitching_BF > 0, "player_id"].astype(int))
    t = signability.tiers(df.mlbam_id, f["as_of"])
    res = {"as_of": f["as_of"], "seasons": f["seasons"], "primary": f["primary"], "role_effect": eff,
           "positives": len(pos), "positives_with_aaa": len(reach),
           "pool_sizes": {"starter": int(df.starter_pool.sum()), "conversion": int(df.conversion_pool.sum())}}
    detail = []
    for arm in cfg()["arms"]:
        d = scored(df, s, int(max(f["seasons"])), eff if arm == "expanded_adj" else None)
        d = d.merge(t[["mlbam_id", "sign_tier", "signable"]], on="mlbam_id", how="left")
        if arm == "starter":
            d = d[d.starter_pool]
        keep = d.signable.fillna(False).astype(bool)
        r, hit = recruitment_backtest.evaluate_arm(d.assign(name=d.mlbam_id.map(pos.set_index("mlbam_id").name)),
                                                   keep, set(pos.mlbam_id), len(reach), config()["amendment6"]["backtest"]["ks"])
        hit = hit.merge(d[["mlbam_id", "conversion_pool", "starting_evidence"]], on="mlbam_id", how="left")
        conv_hit = hit[hit.conversion_pool.fillna(False).astype(bool)]
        r["conversion_positives_in_list"] = int(len(conv_hit))
        r["conversion_median_percentile"] = float(np.median(conv_hit.list_rank / r["N"])) if len(conv_hit) else None
        miss = pos[pos.mlbam_id.isin(reach) & ~pos.mlbam_id.isin(hit.mlbam_id)]
        why = d.set_index("mlbam_id")
        r["missed"] = {n: ("not in pool" if m not in why.index else str(why.sign_tier.get(m)))
                       for m, n in zip(miss.mlbam_id, miss.name)}
        res[arm] = r
        detail.append(hit.assign(fold=name, arm=arm))
    res["detail"] = pd.concat(detail).to_dict(orient="records")
    return res


# ------------------------------------------------------------------ CPBL cohort (amendment 7)
def cohort_check(rng) -> list[dict]:
    """Amendment-7 pooled cohort: P1 with the starter-equivalent adjustment vs unadjusted."""
    f = processed("backtest2026_pitcher_cohort.csv")
    if not f.exists():
        return []
    coh = pd.read_csv(f)
    adj = []
    for y in sorted(coh.aaa_season.unique()):
        seasons = [int(y), int(y) + 1]
        a = recruitment_backtest._pitches(seasons)
        app = appearances(a)
        eff = role_effect(app, rng)
        usage = pitcher_usage(app[app.season == y])
        pf = park.factors_for(seasons, a)
        s = baseline_weights.fetch("pitching", int(y), pf=pf)
        df = milb_candidates._combine(s, [c for c in milb_candidates.PITCH_SUM if c in s] + ["outs"], "pitching_BF")
        df["IP"] = df.outs / 3
        df = df[df.pitching_BF >= backtest_2026.cfg()["pitcher"]["baseline_min_bf"]].copy()
        df["mlbam_id"] = df.player_id.astype(int)
        df = df.join(usage, on="mlbam_id")
        df["conversion_pool"] = df.mlbam_id.isin(coh.loc[coh.aaa_season == y, "mlbam_id"])   # adjust cohort only
        d = scored(df, s, int(y), eff, ref=pd.Series(True, index=df.index))              # z vs all >= 100 BF
        d = d[d.conversion_pool][["mlbam_id", "baseline_score", "relief_share"]].assign(aaa_season=y)
        adj.append(d)
    adj = pd.concat(adj).rename(columns={"baseline_score": "baseline_score_adj"})
    coh = coh.merge(adj, on=["mlbam_id", "aaa_season"], how="left")
    coh.to_csv(processed("conversion_cohort.csv"), index=False)
    b = cfg()["bootstrap"]
    out = []
    for label, x in (("P1 unadjusted (amendment 7)", "baseline_score"), ("P1 starter-equivalent", "baseline_score_adj")):
        est, lo, hi = backtest.boot_ci(coh, x, "cpbl_kbb_c", b["draws"], b["ci"], rng)
        out.append({"test": label, "n": int(coh[[x, "cpbl_kbb_c"]].dropna().shape[0]), "rho": est, "ci_lo": lo, "ci_hi": hi})
    return out


# ------------------------------------------------------------------ 2026 list
def list_2026(rng) -> tuple[pd.DataFrame, dict]:
    seasons = config()["amendment6"]["pool"]["seasons"]
    s = recruitment_backtest.stints(seasons)
    app = appearances(recruitment_backtest._pitches(seasons))
    eff = role_effect(app, rng)
    df = pools(s, pitcher_usage(app))
    el = eligibility.check(df[["mlbam_id", "name"]], "pitcher")
    df = df.merge(el[["mlbam_id", "eligibility_status"]], on="mlbam_id", how="left")
    df = df[df.eligibility_status.eq("eligible")].reset_index(drop=True)
    d = scored(df, s, int(max(seasons)), eff)
    raw = scored(df, s, int(max(seasons)), None)
    d["baseline_score_unadjusted"] = raw.baseline_score.to_numpy()
    t = signability.tiers(d.mlbam_id)
    d = d.merge(t[["mlbam_id", "sign_tier", "signable"]], on="mlbam_id", how="left")
    d["rank_all"] = d.baseline_score.rank(ascending=False, method="first").astype(int)
    cols = ["mlbam_id", "name", "teams", "starter_pool", "conversion_pool", "sign_tier", "signable", "pitching_G",
            "pitching_GS", "IP", "pitching_BF", "relief_share", "max_game_pitches", "starting_evidence", "k_pct_adj",
            "bb_pct_adj", "fip", "baseline_score", "baseline_score_unadjusted", "rank_all"]
    return d[cols].sort_values("rank_all"), eff


def run() -> dict:
    rng = np.random.default_rng(cfg()["bootstrap"]["seed"])
    folds = {k: run_fold(k, f, rng) for k, f in config()["amendment6"]["backtest"]["folds"].items()}
    lst, eff = list_2026(rng)
    lst.to_csv(processed("conversion_pool_pitchers.csv"), index=False)
    coh = cohort_check(rng)
    out = {"role_effect_2025_2026": eff, "folds": folds, "cohort": coh,
           "pool_2026": {"starter": int(lst.starter_pool.sum()), "conversion": int(lst.conversion_pool.sum()),
                         "conversion_signable": int((lst.conversion_pool & lst.signable.fillna(False).astype(bool)).sum())}}
    processed("conversion_pool_summary.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    report(out, lst)
    return out


def _eff_rows(eff: dict) -> pd.DataFrame:
    return pd.DataFrame([{"rate": k, "relief - start": v["effect"], "ci_lo": v["ci"][0], "ci_hi": v["ci"][1]}
                         for k, v in eff.items() if k in RATES])


def report(out: dict, lst: pd.DataFrame) -> None:
    L = ["# Conversion pool (amendment 9; generated by `python -m src.conversion_pool`)", "",
         "AAA pitchers outside the starter pool with >= 400 tracked pitches over the two pool seasons, scored with "
         "the amendment-5 baseline against the starter pool's mean and SD after a starter-equivalent adjustment. "
         "The primary picks stay with the starter pool. Backtest folds were already inspected (exploratory).", "",
         f"## Role effect, AAA 2025-26 (n = {out['role_effect_2025_2026']['n_pitcher_seasons']} pitcher-seasons with >= 30 BF in both roles)", "",
         md_table(_eff_rows(out["role_effect_2025_2026"]), ".4f"), ""]
    for k, r in out["folds"].items():
        L += [f"## Recruitment replay, fold {k} ({'primary' if r['primary'] else 'exploratory'}): as of {r['as_of']}, "
              f"AAA {r['seasons'][0]}-{r['seasons'][-1]}; positives {r['positives']}, with AAA pitching {r['positives_with_aaa']}; "
              f"starter pool {r['pool_sizes']['starter']}, conversion pool {r['pool_sizes']['conversion']}", ""]
        rows = [{"arm": arm, "list N": r[arm]["N"], "positives in list": r[arm]["positives_in_list"],
                 "coverage": r[arm]["coverage"], "hits@10": r[arm]["hits@10"], "random@10": r[arm]["random@10"],
                 "hits@20": r[arm]["hits@20"], "random@20": r[arm]["random@20"],
                 "median pct rank": r[arm]["median_percentile"],
                 "conversion positives": r[arm]["conversion_positives_in_list"],
                 "their median pct rank": r[arm]["conversion_median_percentile"]} for arm in cfg()["arms"]]
        L += [md_table(pd.DataFrame(rows), ".2f"), ""]
        det = pd.DataFrame(r["detail"])
        if len(det):
            L += [md_table(det.pivot_table(index="name", columns="arm", values="list_rank").reset_index(), ".0f"), ""]
        for arm in cfg()["arms"]:
            if r[arm]["missed"]:
                L.append(f"Missed by `{arm}`: " + ", ".join(f"{n} ({w})" for n, w in r[arm]["missed"].items()))
        L += ["", "Role effect in this fold:", "", md_table(_eff_rows(r["role_effect"]), ".4f"), ""]
    if out["cohort"]:
        L += ["## CPBL cohort (amendment 7 pooled, descriptive)", "",
              md_table(pd.DataFrame(out["cohort"]), ".2f"), ""]
    p = out["pool_2026"]
    top = lst[lst.conversion_pool & lst.signable.fillna(False).astype(bool)].head(15)
    L += [f"## 2026 conversion list (signable T1 + T2: {p['conversion_signable']} of {p['conversion']}; starter pool {p['starter']})", "",
          "rank_all = rank among starter + conversion pool on one scale (all tiers).", "",
          md_table(top.drop(columns=["starter_pool", "conversion_pool", "signable"]), ".3f"), ""]
    (path("reports") / "conversion_pool.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    r = run()
    print(json.dumps({k: v for k, v in r.items() if k != "folds"}, indent=1, default=str))
