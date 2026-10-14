"""Phase 5 — adversarial validation of the Fit indices.

Test A  baseline vs Fit: rank overlap, correlation, and how much of Fit is explained
        by baseline stats (is Fit just "high-whiff AAA pitcher" again?)
Test B  stability: Poisson game bootstrap (AAA + CPBL games, and 2026 game logs for the
        baseline), weight perturbation, environment 2024 vs 2025, player 2025 vs 2026
Test C  negative cases: strong baseline / weak Fit, with the driving cells
Test D  falsification: elite-pitcher confound (between vs within-pitcher velocity slope),
        hand/velocity/count adjustment, pitch-classification coarsening, CPBL radar
        offset, foreign-pitcher proxy removal, selection bias of the tracked subset
Writes reports/evaluation_tables.md, data/processed/evaluation_summary.json, figures.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from . import baseline, baseline_weights, fit_score, milb_candidates, milb_loader
from .common import config, md_table, path, processed, zscore
from .cpbl_profile import AAA_COLOR, CPBL_COLOR, GRID, INK, INK2, SURFACE, _style

RNG = np.random.default_rng(config()["evaluation"]["random_seed"])


def spearman(a, b) -> float:
    s = pd.concat([pd.Series(a), pd.Series(b)], axis=1).dropna()
    return float(s.iloc[:, 0].rank().corr(s.iloc[:, 1].rank())) if len(s) > 2 else np.nan


# ------------------------------------------------------------------ Test A
def test_a(P: pd.DataFrame, H: pd.DataFrame) -> dict:
    out = {}
    for role, D, fitcol, ctrl in (
        ("pitcher", P, "fit_score", ["k_minus_bb_pct_raw", "k_pct", "bb_pct", "fip_raw", "aaa_whiff_pct_tracked",
                                     "swstr_pct_aaa", "ip_per_gs"]),
        ("hitter", H, "fit_score", ["woba_raw", "iso_raw", "k_pct_raw", "bb_pct_raw", "whiff_pct_aaa"]),
    ):
        d = D[D.fit_available].copy()
        top_b = d.nsmallest(5, "baseline_rank_tracked").name.tolist()
        top_f = d.nsmallest(5, "fit_rank").name.tolist()
        top_c = d.nsmallest(5, "combined_rank").name.tolist()
        ctrl = [c for c in ctrl if d[c].notna().any()]      # amendment 6: statsapi tables carry no swing counts
        X = d[ctrl].apply(lambda s: s.fillna(s.mean()))
        X = sm.add_constant((X - X.mean()) / X.std())
        r2 = sm.OLS(d[fitcol], X).fit().rsquared
        cors = {c: spearman(d[fitcol], d[c]) for c in ctrl}
        if role == "pitcher":
            d["ff_velo_proxy"] = d[["usage_FB94+"]].sum(axis=1) + 0.5 * d["usage_FB91-94"]
            cors["share_FB94+ (+half 91-94)"] = spearman(d[fitcol], d.ff_velo_proxy)
        out[role] = {"n": len(d), "spearman_baseline_vs_fit": spearman(d.baseline_score, d[fitcol]),
                     "r2_fit_on_baseline_stats": r2, "fit_vs_stats_spearman": cors,
                     "top5_baseline": top_b, "top5_fit": top_f, "top5_combined": top_c,
                     "top5_overlap_baseline_fit": len(set(top_b) & set(top_f)),
                     "top5_overlap_baseline_combined": len(set(top_b) & set(top_c))}
    return out


# ------------------------------------------------------------------ Test B
def _baseline_logs(role: str, cand: pd.DataFrame):
    """Per-player per-game counting stats for the baseline bootstrap: 2026 mirror logs, plus
    (amendment 6, two-season pool) the earlier pool seasons from the statsapi gameLog."""
    g = milb_loader.game_logs(2026)
    g = g[g.player_id.isin(cand.mlbam_id)]
    a6 = milb_loader.amendment6()
    if a6:
        group = "pitching" if role == "pitcher" else "hitting"
        early = [milb_loader.statsapi_game_logs(y, cand.mlbam_id, group) for y in a6["pool"]["seasons"] if y != 2026]
        g = pd.concat([g, *early], ignore_index=True)
        g = g[g.player_id.isin(cand.mlbam_id)]
        # season x league of every game line, so the bootstrap can re-mix league averages
        st = milb_loader.candidate_stats("pitching" if role == "pitcher" else "batting")
        key = st.drop_duplicates(["season", "team_id"]).set_index(["season", "team_id"]).team_league
        idx = pd.MultiIndex.from_arrays([pd.to_datetime(g.game_date).dt.year, g.team_id])
        g["sl"] = key.reindex(idx).values
    return g


_LG: dict = {}


def _league_cached(kind: str) -> pd.DataFrame:
    if kind not in _LG:
        _LG[kind] = baseline._league_table(kind, None)
    return _LG[kind]


def old_vs_new(P: pd.DataFrame, H: pd.DataFrame) -> pd.DataFrame:
    """Amendment 6: the amendment-5 ranking (frozen copy of the merged results) next to the new one."""
    rows = []
    for role, D in (("pitcher", P), ("hitter", H)):
        old = pd.read_csv(processed("amendment5_frozen", f"{role}_fit_results.csv"))
        basis = "combined_rank" if role == "pitcher" else "baseline_rank_tracked"
        old = old[old.fit_available].sort_values(basis)
        if role == "hitter" and "amendment15" in config():
            basis = "marcel_rank_tracked"
        old["old_rank"] = np.arange(1, len(old) + 1)
        old["old_rank_non40"] = np.nan
        o40 = ~old.on_40man.fillna(False).astype(bool)
        old.loc[o40, "old_rank_non40"] = np.arange(1, int(o40.sum()) + 1)
        new = D[D.fit_available].sort_values(basis).copy()
        new["new_rank"] = np.arange(1, len(new) + 1)
        sig = new.signable.fillna(False).astype(bool)
        new["new_rank_signable"] = np.nan
        new.loc[sig, "new_rank_signable"] = np.arange(1, int(sig.sum()) + 1)
        m = old[["mlbam_id", "name", "old_rank", "old_rank_non40"]].merge(
            new[["mlbam_id", "name", "new_rank", "new_rank_signable", "sign_tier", "in_one_season_pool"]],
            on="mlbam_id", how="outer", suffixes=("_old", ""))
        m["name"] = m.name.fillna(m.name_old)
        keep = (m.old_rank_non40 <= 10) | (m.new_rank_signable <= 10) | (m.old_rank <= 5) | (m.new_rank <= 5)
        rows.append(m[keep].drop(columns="name_old").assign(role=role))
    out = pd.concat(rows).sort_values(["role", "new_rank_signable", "old_rank_non40"])
    out.to_csv(processed("amendment6_old_vs_new.csv"), index=False)
    return out


def _boot_league(g: pd.DataFrame, role: str, cols: list[str]) -> pd.DataFrame:
    """Amendment 6: league averages re-mixed with the resampled BF / PA shares per season x league."""
    w = "pitching_BF" if role == "pitcher" else "batting_PA"
    sh = g.dropna(subset=["sl"]).pivot_table(index="player_id", columns="sl", values=w, aggfunc="sum").fillna(0)
    sh = sh.div(sh.sum(axis=1).replace(0, np.nan), axis=0)
    lg = _league_cached("pitching" if role == "pitcher" else "batting")
    return pd.DataFrame({c: sh.mul(lg[c].reindex(sh.columns), axis=1).sum(axis=1, min_count=1) for c in cols})


def baseline_boot(role: str, cand: pd.DataFrame, logs: pd.DataFrame, w: pd.Series | None) -> pd.Series:
    """Recompute the pre-registered baseline score from (re-weighted) 2026 game logs."""
    g = logs.copy()
    if w is not None:
        f = g.game_id.map(w).fillna(0).values
        num = g.select_dtypes("number").columns.difference(["player_id", "game_id", "season", "league_id",
                                                             "team_id", "opp_id", "team_org_id", "opp_org_id"])
        g[num] = g[num].mul(f, axis=0)
    lo, hi = config()["baseline"]["recent_window"]
    rec = g[(g.game_date >= lo) & (g.game_date <= hi)]
    d = cand.set_index("mlbam_id").copy()
    if role == "pitcher":
        s = g.groupby("player_id")[["pitching_BF", "pitching_SO", "pitching_BB", "pitching_HBP", "pitching_HR",
                                    "pitching_IP", "pitching_GS"]].sum()
        r = rec.groupby("player_id")[["pitching_BF", "pitching_SO", "pitching_BB"]].sum()
        s = s.reindex(d.index).fillna(0)
        r = r.reindex(d.index).fillna(0)
        kbb = (s.pitching_SO - s.pitching_BB) / s.pitching_BF
        lg_adj = d.lg_k_minus_bb
        cf = d.fip_raw - (13 * d.pitching_HR + 3 * (d.pitching_BB + d.pitching_HBP) - 2 * d.pitching_SO) / d.IP
        if "sl" in g:       # amendment 6: league mix follows the resampled games
            lgb = _boot_league(g, role, ["k_minus_bb_pct", "k_pct", "bb_pct", "fip", "cfip"]).reindex(d.index)
            d = d.assign(lg_k_minus_bb=lgb.k_minus_bb_pct.fillna(d.lg_k_minus_bb), lg_k_pct=lgb.k_pct.fillna(d.lg_k_pct),
                         lg_bb_pct=lgb.bb_pct.fillna(d.lg_bb_pct), lg_fip=lgb.fip.fillna(d.lg_fip))
            lg_adj, cf = d.lg_k_minus_bb, lgb.cfip.fillna(cf)
        x = pd.DataFrame(index=d.index)
        x["k_minus_bb_pct"] = kbb - lg_adj
        x["fip"] = (13 * s.pitching_HR + 3 * (s.pitching_BB + s.pitching_HBP) - 2 * s.pitching_SO) / s.pitching_IP + cf - d.lg_fip
        k = config()["samples"]["recent_shrink_bf"]
        x["recent_k_minus_bb_pct"] = ((r.pitching_SO - r.pitching_BB) + kbb * k) / (r.pitching_BF + k) - lg_adj
        x["ip_per_gs"] = s.pitching_IP / s.pitching_GS.replace(0, np.nan)
        x["gs"] = s.pitching_GS
        x["k_pct_adj"] = s.pitching_SO / s.pitching_BF - d.lg_k_pct         # amendment 5
        x["bb_pct_adj"] = s.pitching_BB / s.pitching_BF - d.lg_bb_pct
    else:
        cols = ["batting_PA", "batting_AB", "batting_H", "batting_2B", "batting_3B", "batting_HR", "batting_BB",
                "batting_IBB", "batting_SO", "batting_HBP", "batting_SF", "batting_TB"]
        if "sl" in g:       # amendment 6: league mix follows the resampled games
            lgb = _boot_league(g, role, ["woba", "iso", "bb_pct", "k_pct"]).reindex(d.index)
            d = d.assign(**{f"lg_{c}": lgb[c].fillna(d[f"lg_{c}"]) for c in ("woba", "iso", "bb_pct", "k_pct")})
        s = baseline._hitter_rates(g.groupby("player_id")[cols].sum().reindex(d.index).fillna(0))
        r = baseline._hitter_rates(rec.groupby("player_id")[cols].sum().reindex(d.index).fillna(0))
        x = pd.DataFrame(index=d.index)
        for c in ("woba", "iso", "bb_pct", "k_pct"):
            x[c] = s[c] - d[f"lg_{c}"]
        k = config()["samples"]["recent_shrink_bf"]
        x["recent_woba"] = (r.woba_num + s.woba * k) / (r.woba_den + k) - d.lg_woba
    x = x.replace([np.inf, -np.inf], np.nan).apply(lambda c: c.fillna(c.median()))
    return baseline.score(x.reset_index(), role).set_index("mlbam_id").baseline_score


def poisson_weights(ids) -> pd.Series:
    ids = pd.Index(pd.unique(np.asarray(ids)))
    return pd.Series(RNG.poisson(1.0, len(ids)).astype(float), index=ids)


def test_b(r: dict, P: pd.DataFrame, H: pd.DataFrame, reps: int | None = None) -> dict:
    reps = reps or config()["evaluation"]["bootstrap_reps"]
    t = r["tables"]
    Pt = P[P.fit_available].copy()
    Ht = H[H.fit_available].copy()
    Pt["pid"], Ht["bid"] = Pt.mlbam_id.astype(str), Ht.mlbam_id.astype(str)
    plogs, hlogs = _baseline_logs("pitcher", Pt), _baseline_logs("hitter", Ht)
    cpbl_games = t["env_cpbl"].game_id.unique()
    # one weight per AAA game, shared by the environment (2024-25), the player tables
    # (2025-26) and the 2026 game logs behind the baseline, so a resampled game moves
    # baseline and Fit together. Game ids are compared as strings (logs store integers).
    plogs["game_id"], hlogs["game_id"] = plogs.game_id.astype(str), hlogs.game_id.astype(str)
    a13 = "amendment13" in config()
    if a13:                              # amendment 13: Stuff+ re-weighted with the same AAA games
        from . import stuff_rank
        sums = stuff_rank.game_sums()
    aaa_games = pd.unique(np.concatenate([t[k].game_id.astype(str).unique() for k in ("env_aaa", "pit", "hit", "hand_hit")]
                                         + [plogs.game_id.unique(), hlogs.game_id.unique()]
                                         + ([sums.game_pk.astype(str).unique()] if a13 else [])))
    a15 = "amendment15" in config()
    if a15:
        from . import marcel
    ch = config()["fit"]["combined_heuristic"]
    store = {k: [] for k in ("p_fit", "p_comb", "p_base", "h_fit", "h_comb", "h_base")}
    for _ in range(reps):
        wc, wa = poisson_weights(cpbl_games), poisson_weights(aaa_games)
        env = fit_score.environment(t, wc, wa)
        pf = fit_score.pitcher_fit(t, env, wa).reindex(Pt.pid)
        hf = fit_score.hitter_fit(t, env, wa).reindex(Ht.bid)
        pfit = fit_score.combine_pitcher(pf).values
        hfit = zscore(hf.fit_total, 2.5).values
        if a15:     # amendment 15: Marcel from the re-weighted 2025-26 stints (2024 held fixed)
            pb = marcel.candidate_scores("pitcher", Pt.mlbam_id, marcel.log_scale(plogs, wa)).score.values
            hb = marcel.candidate_scores("hitter", Ht.mlbam_id, marcel.log_scale(hlogs, wa)).score.values
        else:
            pb = baseline_boot("pitcher", Pt, plogs, wa).reindex(Pt.mlbam_id).values
            hb = baseline_boot("hitter", Ht, hlogs, wa).reindex(Ht.mlbam_id).values
        if a13:
            st = stuff_rank.candidate_stuff(sums, wa).set_index("pitcher_id").stuff_plus
            pstuff = Pt.mlbam_id.map(st).reset_index(drop=True)
        for key, f, b in (("p", pfit, pb), ("h", hfit, hb)):
            if key == "p" and a13:
                comb = stuff_rank.combined(pd.Series(b), pstuff, pd.Series(f)).values
            else:
                comb = ch["baseline"] * zscore(pd.Series(b)).values + ch["fit"] * np.nan_to_num(f)
            store[f"{key}_fit"].append(np.asarray(f, dtype=float))
            store[f"{key}_base"].append(np.asarray(b, dtype=float))
            store[f"{key}_comb"].append(np.asarray(comb, dtype=float))

    def ranks(S: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        S = S if mask is None else np.where(mask, S, -np.inf)
        R = pd.DataFrame(S).rank(axis=1, ascending=False, method="first").values
        return R if mask is None else np.where(mask, R, np.nan)

    out = {}
    for key, D in (("p", Pt), ("h", Ht)):
        res = pd.DataFrame({"name": D.name.values, "mlbam_id": D.mlbam_id.values,
                            "on_40man": D.on_40man.fillna(False).astype(bool).values})
        free = ~res.on_40man.values
        sig = D.signable.fillna(False).astype(bool).values if "signable" in D else None
        for kind in ("fit", "base", "comb"):
            S = np.vstack(store[f"{key}_{kind}"])
            R = ranks(S)
            res[f"{kind}_rank_med"] = np.median(R, axis=0)
            res[f"{kind}_rank_p05"] = np.percentile(R, 5, axis=0)
            res[f"{kind}_rank_p95"] = np.percentile(R, 95, axis=0)
            res[f"{kind}_p_top1"] = (R == 1).mean(axis=0)
            res[f"{kind}_p_top5"] = (R <= 5).mean(axis=0)
            # among candidates NOT on a 40-man roster (amendment 1 primary-pick pool)
            Rf = ranks(S, np.broadcast_to(free, S.shape))
            res[f"{kind}_p_top1_non40"] = np.where(free, (Rf == 1).mean(axis=0), np.nan)
            res[f"{kind}_p_top3_non40"] = np.where(free, (Rf <= 3).mean(axis=0), np.nan)
            if sig is not None:      # amendment 6: among signable candidates (tiers T1 + T2)
                Rs = ranks(S, np.broadcast_to(sig, S.shape))
                res[f"{kind}_p_top1_signable"] = np.where(sig, (Rs == 1).mean(axis=0), np.nan)
                res[f"{kind}_p_top3_signable"] = np.where(sig, (Rs <= 3).mean(axis=0), np.nan)
        out[key] = res.sort_values("comb_rank_med")
    return out


def weight_perturbation(P: pd.DataFrame, H: pd.DataFrame, n: int = 400) -> dict:
    """Dirichlet perturbation of the baseline weights in use (amendment 4: estimated); Top-1 frequency.
    A weight estimated at exactly 0 gets a small floor so the Dirichlet is defined."""
    res = {}
    for role, D in (("pitcher", P[P.fit_available]), ("hitter", H[H.fit_available])):
        active = baseline.active_weights(role)
        names = list(config()["baseline"][role])
        base_w = np.maximum(np.array([active[m] for m in names]), 0.02)
        base_w = base_w / base_w.sum()
        tops_base, tops_comb = [], []
        for _ in range(n):
            w = RNG.dirichlet(base_w * 20)
            s = baseline.score(D, role, dict(zip(names, w)))
            tops_base.append(s.iloc[0]["name"])
            fw = RNG.uniform(0.0, 0.5)
            if role == "pitcher":
                a = RNG.uniform(0.2, 0.8)
                fit = fit_score.combine_pitcher(s, {"mix_whiff": a, "mix_damage": 1 - a})
            else:
                fit = s.fit_score
            comb = (1 - fw) * zscore(s.baseline_score) + fw * fit
            tops_comb.append(s.loc[comb.idxmax(), "name"])
        res[role] = {"baseline_top1_freq": pd.Series(tops_base).value_counts(normalize=True).head(5).round(3).to_dict(),
                     "combined_top1_freq": pd.Series(tops_comb).value_counts(normalize=True).head(5).round(3).to_dict()}
        grid = []
        for wb, wf in config()["fit"]["weight_grid"]:
            comb = wb * zscore(D.baseline_score) + wf * D.fit_score
            grid.append({"w_baseline": wb, "w_fit": wf, "top1": D.loc[comb.idxmax(), "name"],
                         "top3": ", ".join(D.loc[comb.nlargest(3).index, "name"])})
        res[role]["grid"] = grid
    return res


def season_split(r: dict, P: pd.DataFrame, H: pd.DataFrame) -> dict:
    t = r["tables"]
    e24 = fit_score.environment(t, cpbl_seasons=[2024], aaa_seasons=[2024])
    e25 = fit_score.environment(t, cpbl_seasons=[2025], aaa_seasons=[2025])
    env_corr_w = spearman(e24.whiff_contrast, e25.whiff_contrast)
    env_corr_d = spearman(e24.damage_contrast, e25.damage_contrast)
    out = {"env_whiff_contrast_2024": e24.whiff_contrast.round(3).to_dict(),
           "env_whiff_contrast_2025": e25.whiff_contrast.round(3).to_dict(),
           "env_damage_contrast_2024": e24.damage_contrast.round(3).to_dict(),
           "env_damage_contrast_2025": e25.damage_contrast.round(3).to_dict(),
           "env_whiff_contrast_spearman_24v25": env_corr_w,
           "env_damage_contrast_spearman_24v25": env_corr_d}
    # player-level reliability: same pooled environment, player data split by season
    # (amendment 1: the two player_seasons, 2025 vs 2026)
    env = r["env"]
    s1, s2 = config()["samples"]["player_seasons"]
    out["player_seasons_compared"] = [s1, s2]
    a, b = fit_score.pitcher_fit(t, env, seasons=[s1]), fit_score.pitcher_fit(t, env, seasons=[s2])
    j = a.join(b, lsuffix="_1", rsuffix="_2", how="inner")
    j = j[(j.tracked_pitches_used_1 >= 300) & (j.tracked_pitches_used_2 >= 300)]
    out["pitcher_n_both_seasons"] = len(j)
    out["pitcher_mix_whiff_r_season"] = spearman(j.fit_mix_whiff_1, j.fit_mix_whiff_2)
    out["pitcher_mix_damage_r_season"] = spearman(j.fit_mix_damage_1, j.fit_mix_damage_2)
    cols = ["fit_mix_whiff", "fit_mix_damage"]
    out["pitcher_fit_score_r_season"] = spearman(
        fit_score.combine_pitcher(j[[c + "_1" for c in cols]].set_axis(cols, axis=1)),
        fit_score.combine_pitcher(j[[c + "_2" for c in cols]].set_axis(cols, axis=1)))
    a, b = fit_score.hitter_fit(t, env, seasons=[s1]), fit_score.hitter_fit(t, env, seasons=[s2])
    j = a.join(b, lsuffix="_1", rsuffix="_2", how="inner")
    j = j[(j.tracked_pa_used_1 >= 100) & (j.tracked_pa_used_2 >= 100)]
    out["hitter_n_both_seasons"] = len(j)
    for c in ("fit_total", "fit_velo", "fit_type", "fit_hand"):
        out[f"hitter_{c}_r_season"] = spearman(j[f"{c}_1"], j[f"{c}_2"])
    # environment fixed to one season, all player data: does the ranking depend on env season?
    for lab, e in (("2024env", e24), ("2025env", e25)):
        pf = fit_score.pitcher_fit(t, e)
        out[f"pitcher_fit_rank_corr_pooled_vs_{lab}"] = spearman(fit_score.combine_pitcher(pf), fit_score.combine_pitcher(fit_score.pitcher_fit(t, env)))
        hf = fit_score.hitter_fit(t, e)
        out[f"hitter_fit_rank_corr_pooled_vs_{lab}"] = spearman(hf.fit_total, fit_score.hitter_fit(t, env).fit_total)
    return out


# ------------------------------------------------------------------ Test C
def test_c(P: pd.DataFrame, H: pd.DataFrame, env: pd.DataFrame) -> dict:
    out = {}
    d = P[P.fit_available]
    top = d[d.baseline_rank_tracked <= 10]
    neg = top.sort_values("fit_score").iloc[0]
    contrib = {c: round(float(neg[f"f1_contrib_{c}"]), 3) for c in fit_score.CELLS}
    usage = {c: round(float(neg[f"usage_{c}"]), 3) for c in fit_score.CELLS}
    out["pitcher"] = {"name": neg["name"], "baseline_rank_tracked": int(neg.baseline_rank_tracked),
                      "fit_rank": int(neg.fit_rank), "k_minus_bb_pct": round(float(neg.k_minus_bb_pct_raw), 3),
                      "fip": round(float(neg.fip_raw), 2), "fit_mix_whiff": round(float(neg.fit_mix_whiff), 3),
                      "fit_mix_damage": round(float(neg.fit_mix_damage), 2), "usage": usage, "f1_contrib": contrib,
                      "aaa_usage_mean": {c: round(float(d[f"usage_{c}"].mean()), 3) for c in fit_score.CELLS}}
    d = H[H.fit_available]
    top = d[d.baseline_rank_tracked <= 10]
    neg = top.sort_values("fit_score").iloc[0]
    out["hitter"] = {"name": neg["name"], "baseline_rank_tracked": int(neg.baseline_rank_tracked),
                     "fit_rank": int(neg.fit_rank), "woba": round(float(neg.woba_raw), 3),
                     "fit_type": round(float(neg.fit_type), 2), "fit_velo": round(float(neg.fit_velo), 2),
                     "fit_hand": round(float(neg.fit_hand), 2),
                     "shape_wOBA_pts": {c: (None if pd.isna(neg[f"shape_{c}"]) else round(float(neg[f"shape_{c}"]), 1))
                                        for c in fit_score.CELLS},
                     "pa_share_shift": {c: round(float(env.loc[c, "pa_share_cpbl"] - env.loc[c, "pa_share_aaa"]), 3)
                                        for c in fit_score.CELLS}}
    return out


# ------------------------------------------------------------------ Test D
def velocity_slopes(cpbl: pd.DataFrame, aaa: pd.DataFrame) -> pd.DataFrame:
    """Four-seam whiff vs velocity: pooled (between+within) vs within-pitcher slope."""
    rows = []
    for lg, d in (("CPBL", cpbl), ("AAA", aaa)):
        f = d[(d.pitch_type == "FF") & d.swing & ~d.is_bunt.astype(bool) & d.velo_mph.notna()].copy()
        f["v"] = f.velo_mph.round()
        a = f.groupby(["pitcher_id", "v"]).agg(whiffs=("whiff", "sum"), swings=("swing", "size")).reset_index()
        a["fails"] = a.swings - a.whiffs
        pooled = smf.glm("whiffs + fails ~ v", a, family=sm.families.Binomial()).fit()
        # within-pitcher: demean velocity by pitcher (Mundlak), pitchers with >=50 FF swings
        big = a.groupby("pitcher_id").swings.transform("sum") >= 50
        b = a[big].copy()
        b["v_mean"] = b.groupby("pitcher_id").v.transform(lambda s: np.average(s, weights=b.loc[s.index, "swings"]))
        b["v_within"] = b.v - b.v_mean
        within = smf.glm("whiffs + fails ~ v_within + v_mean", b, family=sm.families.Binomial()).fit()
        rows.append({"league": lg, "ff_swings": int(a.swings.sum()), "pitchers": a.pitcher_id.nunique(),
                     "pooled_slope_per_mph": pooled.params["v"], "pooled_se": pooled.bse["v"],
                     "within_slope_per_mph": within.params["v_within"], "within_se": within.bse["v_within"],
                     "between_slope_per_mph": within.params["v_mean"], "between_se": within.bse["v_mean"]})
    return pd.DataFrame(rows)


def test_d(r: dict, P: pd.DataFrame, H: pd.DataFrame) -> dict:
    cpbl, aaa, t = r["cpbl"], r["aaa"], r["tables"]
    env = r["env"]
    Pt, Ht = P[P.fit_available], H[H.fit_available]
    pids, bids = set(Pt.mlbam_id.astype(str)), set(Ht.mlbam_id.astype(str))
    base_p = fit_score.combine_pitcher(fit_score.pitcher_fit(t, env))
    base_h = fit_score.hitter_fit(t, env).fit_total
    out = {}

    # D1 elite / hard-throwing CPBL pitchers drive the velocity "weakness"?
    ff = cpbl[cpbl.pitch_type == "FF"].groupby("pitcher_id").velo_mph.agg(["mean", "size"])
    ff = ff[ff["size"] >= 200].sort_values("mean", ascending=False)
    hard10 = ff.head(10).index
    c94 = cpbl[(cpbl.pitch_type.isin(["FF", "SI"])) & (cpbl.velo_mph >= 94)]
    out["D1_share_of_FB94+_by_top10_velo_pitchers"] = float(c94.pitcher_id.isin(hard10).mean())
    out["D1_top10_velo_pitchers_mean_ff_mph"] = float(ff.head(10)["mean"].mean())
    t2 = fit_score.build_tables(cpbl[~cpbl.pitcher_id.isin(hard10)], aaa, pids, bids)
    e2 = fit_score.environment(t2)
    out["D1_env_without_top10_velo"] = e2[["swings_cpbl", "whiff_contrast", "damage_contrast"]].round(3).to_dict()
    out["D1_pitcher_fit_corr_without_top10"] = spearman(base_p, fit_score.combine_pitcher(fit_score.pitcher_fit(t2, e2)))
    out["D1_hitter_fit_corr_without_top10"] = spearman(base_h, fit_score.hitter_fit(t2, e2).fit_total)
    vs = velocity_slopes(cpbl, aaa[aaa.season.isin(config()["samples"]["env_seasons"])])
    out["D1_velocity_slopes"] = vs.round(4).to_dict(orient="records")

    # D2 hand / velocity / count adjustment (GLM in cpbl_profile): raw vs adjusted contrast
    wc = pd.read_parquet(processed("env_whiff_contrast.parquet"))
    pooled = wc[wc.pair == "pooled"]
    out["D2_raw_vs_adjusted_contrast_spearman"] = spearman(pooled.raw_logit_diff, pooled.adj_logit_diff)
    out["D2_interaction_p"] = wc.groupby("pair").interaction_p.first().round(4).to_dict()

    # D3 classification coarsening: FB velo bands + BRK + OFF only
    def coarse(df):
        df = df.copy()
        df["pitch_type"] = df.pitch_type.replace({"FC": "SL", "CU": "SL", "SPL": "CH"})
        return df
    t3 = fit_score.build_tables(coarse(cpbl), coarse(aaa), pids, bids)
    e3 = fit_score.environment(t3)
    out["D3_pitcher_fit_corr_coarse"] = spearman(base_p, fit_score.combine_pitcher(fit_score.pitcher_fit(t3, e3)))
    out["D3_hitter_fit_corr_coarse"] = spearman(base_h, fit_score.hitter_fit(t3, e3).fit_total)
    us = r["env"][["usage_cpbl", "usage_aaa"]].round(3)
    out["D3_usage_table"] = us.to_dict()

    # D4 measurement: CPBL broadcast radar offset of +-1 mph
    for shift in (-1.0, 1.0):
        t4 = fit_score.build_tables(cpbl, aaa, pids, bids, cpbl_velo_shift=shift)
        e4 = fit_score.environment(t4)
        out[f"D4_pitcher_fit_corr_velo_shift_{shift:+.0f}"] = spearman(base_p, fit_score.combine_pitcher(fit_score.pitcher_fit(t4, e4)))
        out[f"D4_hitter_fit_corr_velo_shift_{shift:+.0f}"] = spearman(base_h, fit_score.hitter_fit(t4, e4).fit_total)
        out[f"D4_FB94+_whiff_contrast_shift_{shift:+.0f}"] = float(e4.loc["FB94+", "whiff_contrast"])

    # D5 foreign-pitcher proxy: drop CPBL starters in the top quartile of FF velocity
    sp = cpbl[cpbl.pitcher_role == "SP"].pitcher_id.value_counts()
    sp = sp[sp >= 500].index
    q = ff.loc[ff.index.intersection(sp), "mean"]
    proxy = q[q >= q.quantile(0.75)].index
    t5 = fit_score.build_tables(cpbl[~cpbl.pitcher_id.isin(proxy)], aaa, pids, bids)
    e5 = fit_score.environment(t5)
    out["D5_proxy_pitchers_removed"] = len(proxy)
    out["D5_proxy_share_of_cpbl_pitches"] = float(cpbl.pitcher_id.isin(proxy).mean())
    out["D5_pa_share_cpbl_without_proxy"] = e5.pa_share_cpbl.round(3).to_dict()
    out["D5_pitcher_fit_corr"] = spearman(base_p, fit_score.combine_pitcher(fit_score.pitcher_fit(t5, e5)))
    out["D5_hitter_fit_corr"] = spearman(base_h, fit_score.hitter_fit(t5, e5).fit_total)

    # D6 selection bias of the tracked subset
    sel = {}
    for role, D in (("pitcher", P), ("hitter", H)):
        a, b = D[D.fit_available], D[~D.fit_available]
        sel[role] = {"tracked_n": len(a), "untracked_n": len(b),
                     "tracked_age_mean": float(a.age_2026.mean()), "untracked_age_mean": float(b.age_2026.mean()),
                     "tracked_baseline_mean": float(a.baseline_score.mean()),
                     "untracked_baseline_mean": float(b.baseline_score.mean()),
                     "untracked_in_overall_top10": int((b.baseline_rank <= 10).sum())}
    out["D6_selection"] = sel

    # D7 park confound: baseline recomputed from ROAD games only (2026 game logs)
    park = {}
    for role, D in (("pitcher", P), ("hitter", H)):
        logs = _baseline_logs(role, D)
        road = baseline_boot(role, D, logs[logs["loc"] == "A"], None)
        full = baseline_boot(role, D, logs, None)
        rk_road = road.rank(ascending=False)
        top_full = D.nsmallest(10, "baseline_rank")
        park[role] = {"spearman_full_vs_road": spearman(full, road),
                      "top10_full_with_road_rank": {n: int(rk_road.get(m, np.nan)) for n, m in
                                                     zip(top_full.name, top_full.mlbam_id)},
                      "road_top5": [D.set_index("mlbam_id").loc[m, "name"] for m in road.nlargest(5).index]}
        if role == "hitter":
            g = logs[logs["loc"] == "A"].groupby("player_id")[["batting_PA", "batting_AB", "batting_H", "batting_2B",
                "batting_3B", "batting_HR", "batting_BB", "batting_IBB", "batting_SO", "batting_HBP", "batting_SF",
                "batting_TB"]].sum()
            rr = baseline._hitter_rates(g)
            park[role]["road_woba_top10"] = {n: (round(float(rr.loc[m, "woba"]), 3), int(rr.loc[m, "batting_PA"]))
                                             for n, m in zip(top_full.name, top_full.mlbam_id) if m in rr.index}
    out["D7_park_road_only"] = park
    return out


# ------------------------------------------------------------------ recommendation rule
def recommend(P: pd.DataFrame, H: pd.DataFrame, season: dict, boot: dict) -> dict:
    """Pre-registered rule (config.recruitment, amendment 1). Returns shortlists + picks."""
    rc = config()["recruitment"]
    hitter_rel = season.get("hitter_fit_total_r_season", np.nan)
    hitter_uses_fit = bool(hitter_rel >= rc["hitter_fit_min_reliability"])
    out = {"hitter_fit_reliability": hitter_rel, "hitter_uses_fit": hitter_uses_fit}
    keep = ["name", "mlbam_id", "age_2026", "teams", "baseline_rank", "baseline_rank_tracked", "marcel_rank_tracked",
            "marcel_score", "stuff_plus", "fit_rank", "combined_rank", "combined_rank_a13", *milb_candidates.REC_COLS, *milb_candidates.SIGN_COLS]
    rows = []
    hitter_base = "marcel_rank_tracked" if "amendment15" in config() else "baseline_rank_tracked"
    for role, D, key, rank_col in (("pitcher", P, "p", "combined_rank"),
                                   ("hitter", H, "h", "combined_rank" if hitter_uses_fit else hitter_base)):
        d = D[D.fit_available].sort_values(rank_col).copy()
        d["rule_rank"] = np.arange(1, len(d) + 1)
        d["on_40man"] = d.on_40man.fillna(False).astype(bool)
        d["rule_rank_non40"] = np.nan
        d.loc[~d.on_40man, "rule_rank_non40"] = np.arange(1, int((~d.on_40man).sum()) + 1)
        b = boot[key].set_index("mlbam_id")
        kind = "comb" if rank_col == "combined_rank" else "base"
        d["boot_p_top1"] = d.mlbam_id.map(b[f"{kind}_p_top1"])
        d["boot_p_top1_non40"] = d.mlbam_id.map(b[f"{kind}_p_top1_non40"])
        d["boot_p_top3_non40"] = d.mlbam_id.map(b[f"{kind}_p_top3_non40"])
        a6 = "signable" in d
        if a6:      # amendment 6: the primary pick comes from the signable tiers (T1 + T2)
            d["signable"] = d.signable.fillna(False).astype(bool)
            d["rule_rank_signable"] = np.nan
            d.loc[d.signable, "rule_rank_signable"] = np.arange(1, int(d.signable.sum()) + 1)
            d["boot_p_top1_signable"] = d.mlbam_id.map(b[f"{kind}_p_top1_signable"])
            d["boot_p_top3_signable"] = d.mlbam_id.map(b[f"{kind}_p_top3_signable"])
        d["role"], d["rank_basis"] = role, rank_col
        top = d.head(10)
        free = d[~d.on_40man].head(10)
        pool = d[d.signable].head(10) if a6 else free
        rows.append(pd.concat([top, free, pool]).drop_duplicates("mlbam_id"))
        out[role] = {"rank_basis": rank_col, "overall_top": d.iloc[0]["name"],
                     "overall_top_on_40man": bool(d.iloc[0].on_40man),
                     "primary_pick": pool.iloc[0]["name"] if len(pool) else None,
                     "alternates": pool.iloc[1:3].name.tolist(),
                     "primary_pick_rule": "signable (T1+T2), amendment 6" if a6 else "not on a 40-man, amendment 1",
                     "non40_pick": free.iloc[0]["name"] if len(free) else None,
                     "n_ranked": len(d), "n_non40": int((~d.on_40man).sum())}
        if a6:
            out[role]["n_signable"] = int(d.signable.sum())
            out[role]["tier_counts"] = d.sign_tier.value_counts().to_dict()
    cols = ["role", "rank_basis", "rule_rank", "rule_rank_non40", "rule_rank_signable", "boot_p_top1",
            "boot_p_top1_non40", "boot_p_top3_non40", "boot_p_top1_signable", "boot_p_top3_signable", *keep]
    sl = pd.concat(rows)
    sl[[c for c in cols if c in sl.columns]].to_csv(processed("recommendation_shortlist.csv"), index=False)
    return out


# ------------------------------------------------------------------ figures & report
def figures(P, H, boot):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir = path("figures")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), facecolor=SURFACE)
    for ax, D, title, col in ((axes[0], P, "Pitchers", CPBL_COLOR), (axes[1], H, "Hitters", AAA_COLOR)):
        d = D[D.fit_available]
        ax.scatter(zscore(d.baseline_score), d.fit_score, s=22, color=col, edgecolor=SURFACE, lw=0.8)
        offsets = [(-10, 8), (-20, -14), (-70, 8)]       # staggered so top-3 labels never collide
        for (_, row), off in zip(d.nsmallest(3, "combined_rank").iterrows(), offsets):
            ax.annotate(row["name"], (zscore(d.baseline_score)[row.name], row.fit_score), fontsize=7, color=INK,
                        xytext=off, textcoords="offset points",
                        arrowprops=dict(arrowstyle="-", color=INK2, lw=0.5, shrinkA=0, shrinkB=3))
        ax.axhline(0, color=GRID, lw=0.8); ax.axvline(0, color=GRID, lw=0.8)
        _style(ax, f"{title}: baseline z vs CPBL Fit z (n={len(d)})", xlabel="baseline (z)", ylabel="Fit (z)")
    fig.tight_layout(); fig.savefig(fig_dir / "fig5_baseline_vs_fit.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), facecolor=SURFACE)
    for ax, key, title, col in ((axes[0], "p", "Pitchers", CPBL_COLOR), (axes[1], "h", "Hitters", AAA_COLOR)):
        b = boot[key].head(12).iloc[::-1]
        y = np.arange(len(b))
        ax.hlines(y, b.comb_rank_p05, b.comb_rank_p95, color=col, lw=2)
        ax.plot(b.comb_rank_med, y, "o", color=col, ms=6, mec=SURFACE)
        ax.set_yticks(y, b.name, fontsize=8)
        _style(ax, f"{title}: combined rank, 90% bootstrap interval", xlabel="rank (1 = best)")
        ax.grid(axis="x", color=GRID, lw=0.6)
    fig.tight_layout(); fig.savefig(fig_dir / "fig6_bootstrap_rank_intervals.png", dpi=150); plt.close(fig)


def run(reps: int | None = None) -> dict:
    cpbl, aaa = fit_score.load_inputs()
    c = milb_candidates.run(aaa)
    b = baseline.run({k: v[1] for k, v in c.items()})
    r = fit_score.run(b, cpbl, aaa)
    P, H = r["pitcher"], r["hitter"]
    bw = baseline_weights._table()
    summary = {"baseline_weights": bw.round(4).to_dict(orient="records"), "A": test_a(P, H)}
    boot = test_b(r, P, H, reps)
    summary["B_weights"] = weight_perturbation(P, H)
    summary["B_season"] = season_split(r, P, H)
    summary["C"] = test_c(P, H, r["env"])
    summary["D"] = test_d(r, P, H)
    for key in ("p", "h"):
        boot[key].to_csv(processed(f"bootstrap_ranks_{'pitchers' if key == 'p' else 'hitters'}.csv"), index=False)
    summary["B_boot_top"] = {key: boot[key].head(8).round(3).to_dict(orient="records") for key in ("p", "h")}
    summary["recommendation"] = recommend(P, H, summary["B_season"], boot)
    if milb_loader.amendment6():
        from . import signability
        summary["signability_t2_check"] = signability.t2_check(pd.concat([P.mlbam_id, H.mlbam_id]))
        ovn = old_vs_new(P, H)
    figures(P, H, boot)
    with open(processed("evaluation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o), ensure_ascii=False)
    # markdown tables for the reports
    lines = ["# Evaluation tables (auto-generated by `python -m src.evaluate`)\n"]
    lines.append("## Bootstrap rank stability — pitchers (Fit-available subset)\n")
    lines.append(md_table(boot["p"].head(15), ".2f"))
    lines.append("\n## Bootstrap rank stability — hitters (Fit-available subset)\n")
    lines.append(md_table(boot["h"].head(15), ".2f"))
    lines.append("\n## Recommendation shortlist (pre-registered rule, amendment 1)\n")
    sl = pd.read_csv(processed("recommendation_shortlist.csv"))
    lines.append(md_table(sl[[c for c in ["role", "rule_rank", "rule_rank_non40", "rule_rank_signable", "name", "age_2026",
                              "teams", "on_40man", "sign_tier", "roster_club", "roster_status", "mlb_2026_g",
                              "boot_p_top1", "boot_p_top1_non40", "boot_p_top1_signable"] if c in sl]], ".2f"))
    if milb_loader.amendment6():
        lines.append("\n## Amendment 5 (one-season pool, non-40 rule) vs amendment 6 (two seasons, signable tiers)\n")
        lines.append(md_table(ovn[["role", "name", "old_rank", "old_rank_non40", "new_rank", "new_rank_signable",
                                   "sign_tier", "in_one_season_pool"]], ".0f"))
    lines.append("\n## Baseline weights (amendment 4: estimated 2025 -> 2026 AAA; judgment = original)\n")
    lines.append(md_table(bw[["role", "metric", "judgment_weight", "estimated_weight", "ci_lo", "ci_hi", "share_zero",
                              "n", "used"]], ".3f"))
    lines.append(md_table(bw.groupby("role")[["n", "n_predictor_pool", "n_without_target", "cv_spearman_estimated",
                                             "spearman_estimated_in_sample", "spearman_judgment"]].first(), ".3f", index=True))
    lines.append(f"\n## Fit environment cells (2024-25, method: {config()['fit'].get('environment_method', 'pooled')})\n")
    lines.append(md_table(r["env"][["swings_cpbl", "W_cpbl", "swings_aaa", "W_aaa", "whiff_contrast", "pa_cpbl",
                                    "woba_cpbl", "pa_aaa", "woba_aaa", "damage_contrast", "pa_share_cpbl",
                                    "pa_share_aaa"]], ".3f", index=True))
    lines.append("\n## Velocity slopes (four-seam whiff per mph, logit)\n")
    lines.append(md_table(pd.DataFrame(summary["D"]["D1_velocity_slopes"]), ".4f"))
    (path("reports") / "evaluation_tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    import sys
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else None
    s = run(reps)
    print(json.dumps(s, indent=1, default=str, ensure_ascii=False)[:20000])
