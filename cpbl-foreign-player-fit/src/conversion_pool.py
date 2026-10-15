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
           "pool_sizes": {"starter": int(df.starter_pool.sum()), "conversion": int(df.conversion_pool.sum())},
           "conversion_reachable": int(df.loc[df.conversion_pool, "mlbam_id"].isin(reach).sum())}
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
        n_conv_reach = int(df.loc[df.conversion_pool, "mlbam_id"].isin(reach).sum())   # any tier
        r["conversion_positives_in_list"] = int(len(conv_hit))
        r["conversion_coverage"] = len(conv_hit) / n_conv_reach if n_conv_reach else None
        r["conversion_median_percentile"] = float(np.median(conv_hit.list_rank / r["N"])) if len(conv_hit) else None
        for k in config()["amendment6"]["backtest"]["ks"]:
            r[f"conversion_hits@{k}"] = int((conv_hit.list_rank <= k).sum())
            r[f"conversion_random@{k}"] = round(k * len(conv_hit) / r["N"], 2) if r["N"] else None
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
        # role effect and relief share from the pre-arrival AAA season only (no outcome-year data);
        # park factors stay as amendment 7 registered them ([Y, Y + 1])
        app = appearances(recruitment_backtest._pitches([int(y)]))
        eff = role_effect(app, rng)
        usage = pitcher_usage(app)
        seasons = [int(y), int(y) + 1]
        pf = park.factors_for(seasons, recruitment_backtest._pitches(seasons))
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
    for label, x in (("P1 unadjusted (amendment 7)", "baseline_score"),
                     ("P1 starter-equivalent (role effect from the pre-arrival season)", "baseline_score_adj")):
        est, lo, hi = backtest.boot_ci(coh, x, "cpbl_kbb_c", b["draws"], b["ci"], rng)
        out.append({"test": label, "n": int(coh[[x, "cpbl_kbb_c"]].dropna().shape[0]), "rho": est, "ci_lo": lo, "ci_hi": hi})
    return out


# ------------------------------------------------------------------ amendment 17: main-ranking scale
def _zref(x: pd.Series, ref: pd.Series, clip: float | None = None) -> pd.Series:
    """z of x against the mean and SD (ddof 0) of ref, as common.zscore does within a pool."""
    r = pd.to_numeric(ref, errors="coerce")
    x = pd.to_numeric(x, errors="coerce")
    sd = r.std(ddof=0)
    z = (x - r.mean()) / sd if sd > 0 else x * 0.0       # no spread: everyone average, as common.zscore
    return z.clip(-clip, clip) if clip else z


def role_scale(ids, eff: dict, pooled_share: pd.Series, mult: float = 1.0, forecast: int | None = None) -> pd.DataFrame:
    """Marcel per-stint changes (amendment 17): SO, BB, HR minus relief BF x role effect for the
    conversion pitchers' AAA stints in the Marcel window; relief share by season from the pitch
    data, else the pooled 2025-26 share."""
    from . import marcel
    F = forecast or marcel.cfg()["forecast_season"]
    ids = {int(i) for i in ids}
    rows = []
    for y in (F - 1, F - 2, F - 3):
        st = marcel.stints("pitching", y)
        if st.empty:
            continue
        st = st[st.player_id.astype(int).isin(ids)]
        pts = recruitment_backtest._pitches([y])
        share = (pitcher_usage(appearances(pts)).relief_share if len(pts)       # no pitch data: pooled share
                 else pd.Series(dtype=float))
        sh = st.player_id.astype(int).map(share).fillna(st.player_id.astype(int).map(pooled_share)).fillna(0)
        rbf = sh * st.pitching_BF
        r = pd.DataFrame({"player_id": st.player_id.astype(int), "season": y, "team_id": st.team_id.astype(int)})
        for k, col in COUNT.items():                        # additive, so a zero count can rise
            r["+" + col] = (st[col] - mult * rbf * eff[k]["effect"]).clip(lower=0) - st[col]
        rows.append(r)
    return pd.concat(rows).set_index(["player_id", "season", "team_id"])


def _conversion_fit(ids) -> pd.DataFrame:
    """Fit inputs for the conversion pitchers with the starters' environment (ids do not enter it)."""
    from . import fit_score
    cpbl, aaa = fit_score.load_inputs()
    t = fit_score.build_tables(cpbl, aaa, {str(int(i)) for i in ids}, set())
    pf = fit_score.pitcher_fit(t, fit_score.environment(t))
    pf.index = pf.index.astype(int)
    return pf


def a17(lst: pd.DataFrame, eff: dict) -> tuple[pd.DataFrame, dict]:
    """Conversion pool scored on the amendment-13/15 combined scale, z against the starter pool."""
    from . import marcel, stuff_rank
    w = config()["amendment13"]["weights"]
    clip = config()["amendment13"]["stuff_clip"]
    P = pd.read_csv(processed("pitcher_fit_results.csv"))
    S = P[P.fit_available].copy()                           # the ranked starter pool
    C = lst[lst.conversion_pool].copy()
    pooled = C.set_index("mlbam_id").relief_share
    F = marcel.cfg()["forecast_season"]
    def marcel_k_bb(scale):
        m = marcel.project("pitcher", F, scale=scale).set_index("player_id").score
        return S.mlbam_id.astype(int).map(m), C.mlbam_id.astype(int).map(m)
    variants = {"main": role_scale(C.mlbam_id, eff, pooled), "x2": role_scale(C.mlbam_id, eff, pooled, 2.0), "none": None}
    mk = {k: marcel_k_bb(v) for k, v in variants.items()}
    st = stuff_rank.candidate_stuff(stuff_rank.game_sums()).set_index("pitcher_id").stuff_plus
    s_stuff, c_stuff = S.mlbam_id.astype(int).map(st), C.mlbam_id.astype(int).map(st)
    pf = _conversion_fit(C.mlbam_id)
    cf = C.mlbam_id.astype(int).map(lambda i: i).to_frame("id").join(pf, on="id")
    fw = config()["fit"]["pitcher"]["components"]
    c_fit = (fw["mix_whiff"] * _zref(cf.fit_mix_whiff, S.fit_mix_whiff, 2.5)
             + fw["mix_damage"] * _zref(cf.fit_mix_damage, S.fit_mix_damage, 2.5))
    c_fit_av = cf.tracked_pitches_used.fillna(0) >= config()["samples"]["min_pitcher_tracked_pitches"]
    c_fit = c_fit.where(c_fit_av, 0.0).fillna(0.0).to_numpy()
    zs_s = _zref(s_stuff, s_stuff, clip).fillna(0.0)
    zs_c = _zref(c_stuff, s_stuff, clip).fillna(0.0)
    def score(s_m, c_m, stuff_c=zs_c):
        s_ = w["baseline"] * _zref(s_m, s_m) + w["stuff"] * zs_s + w["fit"] * np.nan_to_num(S.fit_score)
        c_ = w["baseline"] * _zref(c_m, s_m) + w["stuff"] * stuff_c + w["fit"] * c_fit
        return s_, c_
    s_main, c_main = score(*mk["main"])
    check = float(np.nanmax(np.abs(s_main.to_numpy() - S.combined_heuristic.to_numpy())))
    C["a17_marcel_k_bb"] = mk["main"][1].to_numpy()
    C["a17_stuff_plus"] = c_stuff.to_numpy()
    C["a17_fit"] = c_fit
    C["a17_fit_available"] = c_fit_av.to_numpy()
    C["a17_score"] = c_main.to_numpy()
    C["a17_score_x2"] = score(*mk["x2"])[1].to_numpy()
    C["a17_score_no_stuff"] = score(*mk["main"], stuff_c=0.0)[1].to_numpy()
    C["a17_score_no_adjustment"] = score(*mk["none"])[1].to_numpy()
    if not np.isfinite(s_main).all():
        raise ValueError("amendment 17: a starter score is not finite")
    ok = np.isfinite(C.a17_score)                           # no Marcel line: listed, not ranked
    allv = pd.concat([s_main, C.a17_score[ok]], ignore_index=True)
    C["a17_rank_all"] = C.a17_score.where(ok).map(lambda v: int((allv > v).sum()) + 1 if np.isfinite(v) else np.nan)
    sig = s_main[S.signable.fillna(False).astype(bool).to_numpy()].to_numpy()
    C["a17_pct_signable_starters"] = C.a17_score.map(lambda v: float((sig <= v).mean() * 100) if np.isfinite(v) else np.nan)
    C["a9_rank_all"] = C.rank_all
    out = {"n_starters": int(len(S)), "n_conversion": int(len(C)), "n_signable_starters": int(len(sig)),
           "n_conversion_fit_available": int(c_fit_av.sum()), "n_conversion_with_stuff": int(C.a17_stuff_plus.notna().sum()),
           "n_conversion_not_ranked_no_marcel": int((~ok).sum()),
           "starter_score_reproduced_max_abs_diff": check}
    if check > 1e-9:
        raise ValueError(f"amendment 17: starter scores do not reproduce the main ranking (max diff {check})")
    return C.sort_values("a17_score", ascending=False), out


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
    # sensitivity (not registered): pure relievers may convert worse than the swingmen the effect is
    # learned from, so also score with twice the effect
    twice = {k: {"effect": 2 * v["effect"]} for k, v in eff.items() if k in RATES}
    d["baseline_score_2x_effect"] = scored(df, s, int(max(seasons)), twice).baseline_score.to_numpy()
    t = signability.tiers(d.mlbam_id)
    d = d.merge(t[["mlbam_id", "sign_tier", "signable"]], on="mlbam_id", how="left")
    d["rank_all"] = d.baseline_score.rank(ascending=False, method="first").astype(int)
    cols = ["mlbam_id", "name", "teams", "starter_pool", "conversion_pool", "sign_tier", "signable", "pitching_G",
            "pitching_GS", "IP", "pitching_BF", "relief_share", "max_game_pitches", "starting_evidence", "k_pct_adj",
            "bb_pct_adj", "fip", "baseline_score", "baseline_score_unadjusted", "baseline_score_2x_effect", "rank_all"]
    return d[cols].sort_values("rank_all"), eff


FROZEN = "conversion_pool_summary_a9.json"     # amendment 9 as published; amendment 17 does not re-run it


def run() -> dict:
    rng = np.random.default_rng(cfg()["bootstrap"]["seed"])
    frozen = (json.loads(processed(FROZEN).read_text())
              if "amendment17" in config() and processed(FROZEN).exists() else None)
    if frozen is None:
        folds = {k: run_fold(k, f, rng) for k, f in config()["amendment6"]["backtest"]["folds"].items()}
    lst, eff = list_2026(rng)
    lst.to_csv(processed("conversion_pool_pitchers.csv"), index=False)
    if frozen is None:
        coh = cohort_check(rng)
    else:                                           # replay and cohort check: amendment 9's published results
        folds, coh = frozen["folds"], frozen["cohort"]
        old = frozen["role_effect_2025_2026"]
        if any(abs(eff[k]["effect"] - old[k]["effect"]) > 1e-12 for k in RATES):
            raise ValueError("amendment 17: the 2025-26 role effect no longer matches amendment 9's")
        eff = old                                   # same point estimates; keep the published intervals
    a17_list, a17_info = (a17(lst, eff) if "amendment17" in config() and processed("pitcher_fit_results.csv").exists()
                          else (None, None))
    if a17_list is not None:
        a17_list.to_csv(processed("conversion_pool_a17.csv"), index=False)
    out = {"role_effect_2025_2026": eff, "folds": folds, "cohort": coh, "a17": a17_info,
           "a9_replay_and_cohort_frozen": frozen is not None,
           "pool_2026": {"starter": int(lst.starter_pool.sum()), "conversion": int(lst.conversion_pool.sum()),
                         "conversion_signable": int((lst.conversion_pool & lst.signable.fillna(False).astype(bool)).sum())}}
    processed("conversion_pool_summary.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    report(out, lst, a17_list)
    return out


def _eff_rows(eff: dict) -> pd.DataFrame:
    return pd.DataFrame([{"rate": k, "relief - start": v["effect"], "ci_lo": v["ci"][0], "ci_hi": v["ci"][1]}
                         for k, v in eff.items() if k in RATES])


def report(out: dict, lst: pd.DataFrame, a17_list: pd.DataFrame | None = None) -> None:
    L = ["# Conversion pool (amendment 9; generated by `python -m src.conversion_pool`)", "",
         "AAA pitchers outside the starter pool with >= 400 tracked pitches over the two pool seasons, scored with "
         "the amendment-5 baseline against the starter pool's mean and SD after a starter-equivalent adjustment. "
         "The primary picks stay with the starter pool. Backtest folds were already inspected (exploratory).", "",
         f"## Role effect, AAA 2025-26 (n = {out['role_effect_2025_2026']['n_pitcher_seasons']} pitcher-seasons with >= 30 BF in both roles)", "",
         md_table(_eff_rows(out["role_effect_2025_2026"]), ".4f"), "",
         "Caveats: the effect is learned from pitchers who both started and relieved; applying it to pure "
         "relievers is an extrapolation, and clubs chose who could start, so the true conversion penalty for "
         "a pure reliever is plausibly larger (scores too optimistic, mainly K% and FIP; BB% unclear). The "
         "bootstrap treats pitcher-seasons as independent and the cohort check holds the adjustment fixed, so "
         "both intervals are too narrow. The 2026 list also shows a score with twice the effect (sensitivity, "
         "not registered).", ""]
    if out.get("a9_replay_and_cohort_frozen"):
        L += ["The recruitment replay and the cohort check below are amendment 9's published results "
              f"(`{FROZEN}`); amendment 17 changes only the 2026 list's scoring (last section).", ""]
    for k, r in out["folds"].items():
        L += [f"## Recruitment replay, fold {k} ({'primary' if r['primary'] else 'exploratory'}): as of {r['as_of']}, "
              f"AAA {r['seasons'][0]}-{r['seasons'][-1]}; positives {r['positives']}, with AAA pitching {r['positives_with_aaa']}; "
              f"starter pool {r['pool_sizes']['starter']}, conversion pool {r['pool_sizes']['conversion']}", ""]
        rows = [{"arm": arm, "list N": r[arm]["N"], "positives in list": r[arm]["positives_in_list"],
                 "coverage": r[arm]["coverage"], "hits@10": r[arm]["hits@10"], "random@10": r[arm]["random@10"],
                 "hits@20": r[arm]["hits@20"], "random@20": r[arm]["random@20"],
                 "median pct rank": r[arm]["median_percentile"],
                 } for arm in cfg()["arms"]]
        L += [md_table(pd.DataFrame(rows), ".2f"), ""]
        sub = [{"arm": arm, "conversion positives in list": r[arm]["conversion_positives_in_list"],
                "coverage": r[arm]["conversion_coverage"], "hits@10": r[arm]["conversion_hits@10"],
                "random@10": r[arm]["conversion_random@10"], "hits@20": r[arm]["conversion_hits@20"],
                "random@20": r[arm]["conversion_random@20"], "median pct rank": r[arm]["conversion_median_percentile"]}
               for arm in cfg()["arms"] if arm != "starter"]
        L += [f"Conversion-pool positives only (denominator: {r['conversion_reachable']} positives in the conversion pool, any tier):", "",
              md_table(pd.DataFrame(sub), ".2f"), ""]
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
    sig = lst[lst.conversion_pool & lst.signable.fillna(False).astype(bool)]
    t15 = set(sig.nlargest(15, "baseline_score").mlbam_id)
    t15x = set(sig.nlargest(15, "baseline_score_2x_effect").mlbam_id)
    st = sig[sig.starting_evidence.eq("stretched")].head(10)
    L += [f"With twice the role effect, {len(t15 & t15x)} of these 15 stay in the top 15.", "",
          "Signable conversion candidates with starting evidence 'stretched' (>= 75 pitches in one AAA game):", "",
          md_table(st[["mlbam_id", "name", "sign_tier", "pitching_G", "pitching_GS", "max_game_pitches",
                       "baseline_score", "baseline_score_2x_effect", "rank_all"]], ".3f"), "",
          "Starting history covers AAA in the two pool seasons only; earlier MLB or AAA starts are not counted.", ""]
    if a17_list is not None:
        i = out["a17"]
        C = a17_list
        sig = C[C.signable.fillna(False).astype(bool)]
        cols = ["name", "sign_tier", "pitching_G", "pitching_GS", "max_game_pitches", "starting_evidence",
                "a17_marcel_k_bb", "a17_stuff_plus", "a17_fit", "a17_score", "a17_rank_all", "a17_pct_signable_starters",
                "a9_rank_all", "a17_score_x2", "a17_score_no_stuff", "a17_score_no_adjustment"]
        top = lambda col: set(sig.nlargest(15, col).mlbam_id)
        t15 = top("a17_score")
        L += ["## Amendment 17: the conversion pool on the main ranking's scale", "",
              "score = 0.5 z(Marcel K-BB%, starter-equivalent stints) + 0.2 z(Stuff+) + 0.3 Fit, every z against the "
              f"{i['n_starters']} Fit-available starters (their scores reproduce the main ranking: max difference "
              f"{i['starter_score_reproduced_max_abs_diff']:.1e}). rank_all = rank among those starters plus the "
              f"{i['n_conversion']} conversion pitchers; pct = percentile among the {i['n_signable_starters']} signable "
              f"starters; a9_rank_all = the old amendment-9 rank (old baseline, different pool size). Conversion pitchers "
              f"with Fit: {i['n_conversion_fit_available']}, with Stuff+: {i['n_conversion_with_stuff']}; not ranked (no Marcel "
              f"line): {i['n_conversion_not_ranked_no_marcel']}.", "",
              "Signable conversion candidates, top 15:", "", md_table(sig.head(15)[cols], ".3f"), "",
              "Sensitivities (reported, never choose): of these 15, still in the signable top 15 with twice the role "
              f"effect {len(t15 & top('a17_score_x2'))}, with no Stuff+ term {len(t15 & top('a17_score_no_stuff'))}, with "
              f"no role adjustment {len(t15 & top('a17_score_no_adjustment'))}.", "",
              "Signable with starting evidence 'stretched':", "",
              md_table(sig[sig.starting_evidence.eq("stretched")].head(10)[cols], ".3f"), ""]
    (path("reports") / "conversion_pool.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    r = run()
    print(json.dumps({k: v for k, v in r.items() if k != "folds"}, indent=1, default=str))
