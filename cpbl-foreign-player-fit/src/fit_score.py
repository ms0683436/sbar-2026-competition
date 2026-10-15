"""Phase 4 — CPBL Fit indices.

Common partition of pitches into 9 CELLS (no cell is counted twice):
  FB<88, FB88-91, FB91-94, FB94+   four-seam + sinker by velocity (mph; CPBL km/h converted)
  FC, SL, CU, CH, SPL              other harmonised pitch types

Seasons (amendment 1): environment = config samples.env_seasons on BOTH sides (2024-25,
the CPBL REBAS span); player profiles = samples.player_seasons of AAA (2025-26).

Environment (per cell, pooled 2024-25 unless stated):
  whiff contrast  w_c = [logit W_CPBL(c) - logit W_AAA(c)] - Δ̄
  damage contrast d_c = [wOBA_CPBL(c) - wOBA_AAA(c)]      - Δ̄_dmg
  where Δ̄ is the AAA-weighted mean gap, so a pitcher/hitter with the AAA-average
  profile scores exactly 0: Fit isolates the *shape* match, not the overall
  level difference between leagues (that level is the baseline's job).

  AMENDMENT 3 (config fit.environment_method = within_pitcher): the per-cell gaps above
  mix "how hitters react to a pitch" with "who throws it" (CPBL 94+ mph fastballs come
  from its best pitchers). Each league's cell effects are therefore estimated WITHIN
  pitcher, with a pitcher fixed effect absorbing his overall level and a location effect
  (in zone / out of zone / unknown) so a pitch type is compared at the same location:
      whiffs ~ Binomial(swings, expit(α_pitcher + β_cell + γ_zone))   (per league)
      mean wOBA = α_pitcher + β_cell + γ_zone, weighted by PA          (per league)
  and the contrasts become w_c = (β^CPBL_c - β^AAA_c) - Σ_c s_c (β^CPBL_c - β^AAA_c),
  with s_c the AAA swing (PA) shares, i.e. the same centering as above. The pooled
  contrasts are kept as whiff_contrast_pooled / damage_contrast_pooled. The overall
  level Δ̄ and the AAA cell rates are unchanged, and F1 uses
  L_cpbl_fit = logit W_AAA(c) + Δ̄ + w_c (identical to logit W_CPBL(c) in pooled mode).

PITCHER FIT (AAA tracked pitches, player_seasons)
  a_pc  = logit(shrunk whiff%_p,c) - logit W_AAA(c)            pitcher's skill on that pitch
  F1 mix_whiff  = 100 * {Σ_c s_pc [σ(logit W_CPBL(c) + a_pc) - σ(logit W_AAA(c) + Δ̄ + a_pc)] - REF}
                  REF = same sum for the AAA-average swing mix with a = 0 (so that profile scores 0)
                  extra whiffs per 100 swings attributable to CPBL-specific weakness
                  (s_pc = pitcher's share of swings in cell c)
  F2 mix_damage = -1000 * Σ_c u_pc d_c   (u_pc = pitch usage share)  wOBA points prevented
  Fit = 0.6 z(F1) + 0.4 z(F2)   [heuristic weights, pre-registered, perturbed in evaluate]

HITTER FIT (AAA tracked PAs, player_seasons)
  sk_hc = n/(n+k) * (wOBA_h,c - wOBA_AAA(c) - δ_h)     split SHAPE (own level δ_h removed)
  G_type + G_velo = 1000 * Σ_c (p_CPBL(c) - p_AAA(c)) sk_hc   (PA-ending pitch distribution shift)
      G_velo = FB-cell part, G_type = non-FB part (additive decomposition)
  G_hand = 1000 * (pLHP_CPBL,side - pLHP_AAA,side) * platoon_dev_h   (k = shrink_k_platoon_pa)
  Fit = G_type + G_velo + G_hand    (all in expected wOBA points; no subjective weights)

Neither Fit is a projection of CPBL ERA/OPS; both are relative-shape indices.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import cpbl_profile, milb_loader, rebas_loader
from .common import config, expit, logit, processed, zscore

FB_CELLS = ["FB<88", "FB88-91", "FB91-94", "FB94+"]
OTHER_CELLS = ["FC", "SL", "CU", "CH", "SPL"]
CELLS = FB_CELLS + OTHER_CELLS


def assign_cell(df: pd.DataFrame, velo_shift: float = 0.0) -> pd.Series:
    vb = config()["velocity_bands_mph"]
    fb = df.pitch_type.isin(["FF", "SI"])
    band = pd.cut(df.velo_mph + velo_shift, vb["edges"], labels=["FB<88", "FB88-91", "FB91-94", "FB94+"],
                  right=False).astype(object)
    cell = pd.Series(np.where(fb, band, df.pitch_type), index=df.index, dtype=object)
    cell[fb & df.velo_mph.isna()] = None
    return cell.where(cell.isin(CELLS))


def _agg(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    d = df[~df.is_bunt.astype(bool) & df.cell.notna()]
    d = d.assign(pa_w=(d.pa_end & (d.woba_denom > 0)).astype(float), woba_num=d.woba_value.fillna(0) * (d.woba_denom > 0))
    return (d.groupby(keys, observed=True)
            .agg(pitches=("swing", "size"), swings=("swing", "sum"), whiffs=("whiff", "sum"),
                 pa=("pa_w", "sum"), woba_num=("woba_num", "sum"))
            .reset_index())


def _zone(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(zone=df.in_zone.map({True: "in", False: "out"}).fillna("unknown"))


def build_tables(cpbl: pd.DataFrame, aaa: pd.DataFrame, pitcher_ids, hitter_ids,
                 cpbl_velo_shift: float = 0.0) -> dict:
    smp = config()["samples"]
    cpbl = cpbl.assign(cell=assign_cell(cpbl, cpbl_velo_shift))
    aaa = aaa.assign(cell=assign_cell(aaa))
    cpbl = cpbl[cpbl.season.isin(smp["env_seasons"])]
    env_aaa = aaa[aaa.season.isin(smp["env_seasons"])]       # environment: same seasons as CPBL
    aaa = aaa[aaa.season.isin(smp["player_seasons"])]         # players: most recent AAA seasons
    t = {
        "env_cpbl": _agg(cpbl, ["season", "game_id", "cell"]),
        "env_aaa": _agg(env_aaa, ["season", "game_id", "cell"]),
        # per-pitcher environment tables for the within-pitcher contrasts (amendment 3)
        "envp_cpbl": _agg(_zone(cpbl), ["pitcher_id", "season", "game_id", "cell", "zone"]),
        "envp_aaa": _agg(_zone(env_aaa), ["pitcher_id", "season", "game_id", "cell", "zone"]),
        "pit": _agg(aaa[aaa.pitcher_id.isin(pitcher_ids)], ["pitcher_id", "season", "game_id", "cell"]),
        "hit": _agg(aaa[aaa.batter_id.isin(hitter_ids)], ["batter_id", "season", "game_id", "cell"]),
    }
    # handedness tables for hitters (PA level)
    def hand(df, keys):
        p = df[df.pa_end & (df.woba_denom > 0) & df.p_throws.isin(["L", "R"]) & df.stand.isin(["L", "R"])]
        return (p.groupby(keys, observed=True)
                .agg(pa=("pa_end", "size"), woba_num=("woba_value", "sum")).reset_index())
    t["hand_cpbl"] = hand(cpbl, ["season", "game_id", "stand", "p_throws"])
    t["hand_aaa"] = hand(env_aaa, ["season", "game_id", "stand", "p_throws"])
    t["hand_hit"] = hand(aaa[aaa.batter_id.isin(hitter_ids)], ["batter_id", "season", "game_id", "stand", "p_throws"])
    return t


def _weighted(df: pd.DataFrame, weights: pd.Series | None, seasons=None) -> pd.DataFrame:
    if seasons is not None:
        df = df[df.season.isin(seasons)]
    if weights is None:
        return df
    w = df.game_id.map(weights).fillna(0.0).values
    df = df.copy()
    for c in ("pitches", "swings", "whiffs", "pa", "woba_num"):
        if c in df.columns:
            df[c] = df[c] * w
    return df


def _fe_effects(factors: list[np.ndarray], y: np.ndarray, n: np.ndarray, logistic: bool, target: int,
               iters: int = 500, tol: float = 1e-9) -> pd.Series:
    """Effects of factors[target] in y ~ Σ_k effect_k(factor_k), by cyclic block updates.

    logistic: y = successes out of n trials (logit link, one Newton step per block per sweep);
    otherwise y = weighted mean outcome with weight n (least squares, exact block solves).
    The target effects are returned relative to their n-weighted mean (only differences
    are identified).
    """
    levels, idx = zip(*(np.unique(f, return_inverse=True) for f in factors))
    eff = [np.zeros(len(lv)) for lv in levels]
    ridge = 1e-6
    for _ in range(iters):
        old = eff[target].copy()
        for k in range(len(factors)):
            eta = sum(e[i] for e, i in zip(eff, idx))
            if logistic:
                mu = expit(eta)
                g = np.bincount(idx[k], y - n * mu, len(levels[k]))
                h = np.bincount(idx[k], n * mu * (1 - mu), len(levels[k])) + ridge
            else:
                g = np.bincount(idx[k], n * (y - eta), len(levels[k]))
                h = np.bincount(idx[k], n, len(levels[k])) + ridge
            eff[k] = eff[k] + g / h
        if np.max(np.abs(eff[target] - old)) < tol:
            break
    w = np.bincount(idx[target], n, len(levels[target]))
    b = eff[target] - np.average(eff[target], weights=w)
    return pd.Series(b, index=levels[target])


def within_pitcher_effects(tp: pd.DataFrame, min_swings: float, min_pa: float) -> pd.DataFrame:
    """Per-cell whiff (logit) and damage (wOBA) effects for one league, within pitcher and location."""
    g = tp.groupby(["pitcher_id", "cell", "zone"])[["swings", "whiffs", "pa", "woba_num"]].sum().reset_index()
    sw = g.groupby("pitcher_id").swings.transform("sum")
    w = g[(sw >= min_swings) & (g.swings > 0)]
    bw = _fe_effects([w.pitcher_id.values, w.cell.values, w.zone.values], w.whiffs.values.astype(float),
                     w.swings.values.astype(float), logistic=True, target=1)
    pa = g.groupby("pitcher_id").pa.transform("sum")
    d = g[(pa >= min_pa) & (g.pa > 0)]
    bd = _fe_effects([d.pitcher_id.values, d.cell.values, d.zone.values], (d.woba_num / d.pa).values,
                     d.pa.values.astype(float), logistic=False, target=1)
    return pd.DataFrame({"beta_whiff": bw, "beta_woba": bd}).reindex(CELLS)


def environment(t: dict, w_cpbl=None, w_aaa=None, cpbl_seasons=None, aaa_seasons=None,
                method: str | None = None) -> pd.DataFrame:
    method = method or config()["fit"].get("environment_method", "pooled")
    k = config()["samples"]["shrink_k_env_swings"]
    c = _weighted(t["env_cpbl"], w_cpbl, cpbl_seasons).groupby("cell")[["pitches", "swings", "whiffs", "pa", "woba_num"]].sum()
    a = _weighted(t["env_aaa"], w_aaa, aaa_seasons).groupby("cell")[["pitches", "swings", "whiffs", "pa", "woba_num"]].sum()
    e = c.join(a, lsuffix="_cpbl", rsuffix="_aaa").reindex(CELLS)
    lg_c = e.whiffs_cpbl.sum() / e.swings_cpbl.sum()
    e["W_cpbl"] = (e.whiffs_cpbl + k * lg_c) / (e.swings_cpbl + k)   # small cells shrink to league rate
    e["W_aaa"] = e.whiffs_aaa / e.swings_aaa
    e["L_cpbl"], e["L_aaa"] = logit(e.W_cpbl), logit(e.W_aaa)
    sw = e.swings_aaa / e.swings_aaa.sum()
    e["gap"] = e.L_cpbl - e.L_aaa
    e["level_gap"] = (sw * e.gap).sum()
    e["whiff_contrast"] = e.gap - e.level_gap
    e["woba_cpbl"] = e.woba_num_cpbl / e.pa_cpbl
    e["woba_aaa"] = e.woba_num_aaa / e.pa_aaa
    pw = e.pa_aaa / e.pa_aaa.sum()
    e["dmg_gap"] = e.woba_cpbl - e.woba_aaa
    e["dmg_level_gap"] = (pw * e.dmg_gap).sum()
    e["damage_contrast"] = e.dmg_gap - e.dmg_level_gap
    e["whiff_contrast_pooled"], e["damage_contrast_pooled"] = e.whiff_contrast, e.damage_contrast
    if method == "within_pitcher":
        wp = config()["fit"]["within_pitcher"]
        bc = within_pitcher_effects(_weighted(t["envp_cpbl"], w_cpbl, cpbl_seasons), wp["min_swings"], wp["min_pa"])
        ba = within_pitcher_effects(_weighted(t["envp_aaa"], w_aaa, aaa_seasons), wp["min_swings"], wp["min_pa"])
        dw = bc.beta_whiff - ba.beta_whiff
        dd = bc.beta_woba - ba.beta_woba
        e["whiff_contrast"] = dw - (sw * dw).sum()
        e["damage_contrast"] = dd - (pw * dd).sum()
    elif method != "pooled":
        raise ValueError(f"unknown environment_method {method!r}")
    e["L_cpbl_fit"] = e.L_aaa + e.level_gap + e.whiff_contrast
    e["usage_cpbl"] = e.pitches_cpbl / e.pitches_cpbl.sum()
    e["usage_aaa"] = e.pitches_aaa / e.pitches_aaa.sum()
    e["pa_share_cpbl"] = e.pa_cpbl / e.pa_cpbl.sum()
    e["pa_share_aaa"] = e.pa_aaa / e.pa_aaa.sum()
    # hand shares per batter side (PA vs LHP)
    hc = _weighted(t["hand_cpbl"], w_cpbl, cpbl_seasons).groupby(["stand", "p_throws"]).pa.sum().unstack()
    ha = _weighted(t["hand_aaa"], w_aaa, aaa_seasons).groupby(["stand", "p_throws"])[["pa", "woba_num"]].sum()
    e.attrs["pL_cpbl"] = (hc["L"] / hc.sum(axis=1)).to_dict()
    hpa = ha.pa.unstack()
    e.attrs["pL_aaa"] = (hpa["L"] / hpa.sum(axis=1)).to_dict()
    hw = (ha.woba_num / ha.pa).unstack()
    e.attrs["aaa_platoon_gap"] = (hw["L"] - hw["R"]).to_dict()   # vsLHP - vsRHP by batter side
    return e


def pitcher_fit(t: dict, env: pd.DataFrame, w_aaa=None, seasons=None) -> pd.DataFrame:
    k = config()["samples"]["shrink_k_whiff_swings"]
    p = _weighted(t["pit"], w_aaa, seasons).groupby(["pitcher_id", "cell"])[["pitches", "swings", "whiffs"]].sum()
    p = p.reset_index().merge(env[["W_aaa", "L_aaa", "L_cpbl_fit", "level_gap", "damage_contrast"]],
                              left_on="cell", right_index=True)
    p["whiff_shrunk"] = (p.whiffs + k * p.W_aaa) / (p.swings + k)
    p["a"] = logit(p.whiff_shrunk) - p.L_aaa
    tot = p.groupby("pitcher_id")[["pitches", "swings"]].transform("sum")
    p["s"] = p.swings / tot.swings
    p["u"] = p.pitches / tot.pitches
    p["f1_term"] = p.s * (expit(p.L_cpbl_fit + p.a) - expit(p.L_aaa + p.level_gap + p.a))
    p["f2_term"] = -p.u * p.damage_contrast
    g = p.groupby("pitcher_id")
    # reference: AAA-average swing mix with AAA-average skill (a = 0); subtracting it makes
    # the AAA-average profile score exactly 0 despite the logistic non-linearity
    sw = env.swings_aaa / env.swings_aaa.sum()
    ref = float((sw * (expit(env.L_cpbl_fit) - expit(env.L_aaa + env.level_gap))).sum())
    out = pd.DataFrame({
        "tracked_pitches_used": g.pitches.sum(), "tracked_swings_used": g.swings.sum(),
        "fit_mix_whiff": 100 * (g.f1_term.sum() - ref), "fit_mix_damage": 1000 * g.f2_term.sum(),
        "aaa_whiff_pct_tracked": g.whiffs.sum() / g.swings.sum(),
    })
    use = p.pivot_table(index="pitcher_id", columns="cell", values="u", fill_value=0).reindex(columns=CELLS, fill_value=0)
    out = out.join(use.add_prefix("usage_"))
    contrib = p.pivot_table(index="pitcher_id", columns="cell", values="f1_term", fill_value=0) * 100
    out = out.join(contrib.reindex(columns=CELLS, fill_value=0).add_prefix("f1_contrib_"))
    return out


def hitter_fit(t: dict, env: pd.DataFrame, w_aaa=None, seasons=None) -> pd.DataFrame:
    k = config()["samples"]["shrink_k_woba_pa"]
    kp = config()["samples"]["shrink_k_platoon_pa"]
    h = _weighted(t["hit"], w_aaa, seasons).groupby(["batter_id", "cell"])[["pa", "woba_num"]].sum().reset_index()
    h = h.merge(env[["woba_aaa", "pa_share_cpbl", "pa_share_aaa"]], left_on="cell", right_index=True)
    h["dev"] = h.woba_num / h.pa.where(h.pa > 0) - h.woba_aaa
    h["dev"] = h.dev.fillna(0)
    lvl = (h.dev * h.pa).groupby(h.batter_id).sum() / h.groupby("batter_id").pa.sum()
    h["delta_h"] = h.batter_id.map(lvl)
    h["sk"] = h.pa / (h.pa + k) * (h.dev - h.delta_h)
    h["term"] = (h.pa_share_cpbl - h.pa_share_aaa) * h.sk
    h["is_fb"] = h.cell.isin(FB_CELLS)
    g = h.groupby("batter_id")
    out = pd.DataFrame({
        "tracked_pa_used": g.pa.sum(),
        "fit_velo": 1000 * h[h.is_fb].groupby("batter_id").term.sum(),
        "fit_type": 1000 * h[~h.is_fb].groupby("batter_id").term.sum(),
        "aaa_woba_tracked": (h.woba_num.groupby(h.batter_id).sum() / g.pa.sum()),
    }).fillna({"fit_velo": 0, "fit_type": 0})
    sk = h.pivot_table(index="batter_id", columns="cell", values="sk").reindex(columns=CELLS)
    out = out.join((1000 * sk).add_prefix("shape_"))
    # handedness
    hh = _weighted(t["hand_hit"], w_aaa, seasons).groupby(["batter_id", "stand", "p_throws"])[["pa", "woba_num"]].sum()
    rows = {}
    for bid, sub in hh.groupby(level=0):
        sub = sub.droplevel(0)
        by_side = sub.groupby(level=0).pa.sum()
        if by_side.sum() == 0:
            continue
        side = by_side.idxmax()
        if by_side.min() / by_side.sum() > 0.2 and len(by_side) == 2:    # switch hitter
            rows[bid] = (side + "/S", 0.0, np.nan)
            continue
        try:
            vl, vr = sub.loc[(side, "L")], sub.loc[(side, "R")]
        except KeyError:
            rows[bid] = (side, 0.0, np.nan)
            continue
        if vl.pa == 0 or vr.pa == 0:
            rows[bid] = (side, 0.0, np.nan)
            continue
        split = vl.woba_num / vl.pa - vr.woba_num / vr.pa
        n_eff = 2 / (1 / vl.pa + 1 / vr.pa)
        dev = n_eff / (n_eff + kp) * (split - env.attrs["aaa_platoon_gap"][side])
        dp = env.attrs["pL_cpbl"][side] - env.attrs["pL_aaa"][side]
        rows[bid] = (side, 1000 * dp * dev, 1000 * dev)
    hand = pd.DataFrame.from_dict(rows, orient="index", columns=["bat_side", "fit_hand", "platoon_dev"])
    out = out.join(hand)
    out["fit_hand"] = out.fit_hand.fillna(0)
    w = config()["fit"]["hitter"]["components"]
    out["fit_total"] = w["pitch_mix"] * out.fit_type + w["velocity"] * out.fit_velo + w["hand"] * out.fit_hand
    return out


def combine_pitcher(f: pd.DataFrame, weights: dict | None = None) -> pd.Series:
    w = weights or config()["fit"]["pitcher"]["components"]
    return w["mix_whiff"] * zscore(f.fit_mix_whiff, 2.5) + w["mix_damage"] * zscore(f.fit_mix_damage, 2.5)


def load_inputs():
    cpbl = cpbl_profile.add_bands(rebas_loader.load())
    aaa = cpbl_profile.add_bands(milb_loader.load())
    return cpbl, aaa


def run(baseline: dict, cpbl=None, aaa=None) -> dict:
    if cpbl is None:
        cpbl, aaa = load_inputs()
    smp = config()["samples"]
    bp, bh = baseline["pitcher"].copy(), baseline["hitter"].copy()
    bp["pid"], bh["bid"] = bp.mlbam_id.astype(str), bh.mlbam_id.astype(str)
    t = build_tables(cpbl, aaa, set(bp.pid), set(bh.bid))
    env = environment(t)
    env.to_csv(processed("fit_environment_cells.csv"))
    pf = pitcher_fit(t, env)
    hf = hitter_fit(t, env)
    P = bp.merge(pf, left_on="pid", right_index=True, how="left")
    H = bh.merge(hf, left_on="bid", right_index=True, how="left")
    P["fit_available"] = P.tracked_pitches_used.fillna(0) >= smp["min_pitcher_tracked_pitches"]
    H["fit_available"] = H.tracked_pa_used.fillna(0) >= smp["min_hitter_tracked_pa"]
    ch = config()["fit"]["combined_heuristic"]
    for D, fitcol in ((P, None), (H, "fit_total")):
        m = D.fit_available
        if fitcol is None:
            D.loc[m, "fit_score"] = combine_pitcher(D[m])
        else:
            D.loc[m, "fit_score"] = zscore(D.loc[m, fitcol], 2.5)
        D.loc[m, "fit_rank"] = D.loc[m, "fit_score"].rank(ascending=False, method="first")
        D.loc[m, "baseline_rank_tracked"] = D.loc[m, "baseline_score"].rank(ascending=False, method="first")
        D.loc[m, "combined_heuristic"] = ch["baseline"] * zscore(D.loc[m, "baseline_score"]) + ch["fit"] * D.loc[m, "fit_score"]
        D.loc[m, "combined_rank"] = D.loc[m, "combined_heuristic"].rank(ascending=False, method="first")
    if "amendment13" in config():        # Stuff+ enters the pitcher ranking (amendment 13)
        from . import stuff_rank
        st = stuff_rank.candidate_stuff(stuff_rank.game_sums()).set_index("pitcher_id")
        P["stuff_plus"] = P.mlbam_id.map(st.stuff_plus)
        P["stuff_pitches"] = P.mlbam_id.map(st.stuff_pitches)
        # amendment 18 replaces Stuff+ by its Marcel-style version; the old one is kept for the
        # earlier rankings (amendments 13 and 15) and the ablation
        a13 = stuff_rank.candidate_stuff_a13(stuff_rank.game_sums()).set_index("pitcher_id")
        P["stuff_plus_a13"] = P.mlbam_id.map(a13.stuff_plus)
        m = P.fit_available
        P.loc[m, "combined_a6"] = P.loc[m, "combined_heuristic"]
        P.loc[m, "combined_rank_a6"] = P.loc[m, "combined_rank"]
        P.loc[m, "combined_a13"] = stuff_rank.combined(P.loc[m, "baseline_score"], P.loc[m, "stuff_plus_a13"],
                                                       P.loc[m, "fit_score"])           # amendment 13 as published
        P.loc[m, "combined_rank_a13"] = P.loc[m, "combined_a13"].rank(ascending=False, method="first")
        # rule S (amendment 18) applies whichever baseline the ranking uses
        stuff_now = "stuff_plus" if "amendment18" in config() else "stuff_plus_a13"
        P.loc[m, "combined_heuristic"] = stuff_rank.combined(P.loc[m, "baseline_score"], P.loc[m, stuff_now],
                                                             P.loc[m, "fit_score"])
        P.loc[m, "combined_rank"] = P.loc[m, "combined_heuristic"].rank(ascending=False, method="first")
    if "amendment15" in config():        # AAA Marcel replaces the baseline in the ranking (amendment 15)
        from . import marcel
        for D in (P, H):
            role = "pitcher" if D is P else "hitter"
            mp = marcel.candidate_scores(role, D.mlbam_id)
            D["marcel_score"] = mp.score.to_numpy()
            D["marcel_share_data"] = mp[f"share_data_{'k' if role == 'pitcher' else 'woba'}"].to_numpy()
            if role == "pitcher":
                D["marcel_k_pct"], D["marcel_bb_pct"], D["marcel_fip"] = mp.k.to_numpy(), mp.bb.to_numpy(), mp.fip.to_numpy()
            else:
                D["marcel_iso"], D["marcel_k_pct"], D["marcel_bb_pct"] = mp.iso.to_numpy(), mp.k.to_numpy(), mp.bb.to_numpy()
            m = D.fit_available
            D.loc[m, "marcel_rank_tracked"] = D.loc[m, "marcel_score"].rank(ascending=False, method="first")
        m = P.fit_available
        if marcel.active("pitcher"):             # unless V1 found Marcel "clearly worse"
            P.loc[m, "combined_heuristic"] = stuff_rank.combined(P.loc[m, "marcel_score"], P.loc[m, "stuff_plus"],
                                                                 P.loc[m, "fit_score"])
            P.loc[m, "combined_rank"] = P.loc[m, "combined_heuristic"].rank(ascending=False, method="first")
            P.loc[m, "combined_a15"] = stuff_rank.combined(P.loc[m, "marcel_score"], P.loc[m, "stuff_plus_a13"],
                                                           P.loc[m, "fit_score"])     # amendment 15 as published
            P.loc[m, "combined_rank_a15"] = P.loc[m, "combined_a15"].rank(ascending=False, method="first")
    P = P.sort_values(["fit_available", "combined_rank"], ascending=[False, True])
    H = H.sort_values(["fit_available", "combined_rank"], ascending=[False, True])
    P.drop(columns="pid").to_csv(processed("pitcher_fit_results.csv"), index=False)
    H.drop(columns="bid").to_csv(processed("hitter_fit_results.csv"), index=False)
    return {"pitcher": P, "hitter": H, "env": env, "tables": t, "cpbl": cpbl, "aaa": aaa}


if __name__ == "__main__":
    from . import baseline, milb_candidates
    cpbl, aaa = load_inputs()
    c = milb_candidates.run(aaa)
    b = baseline.run({k: v[1] for k, v in c.items()})
    r = run(b, cpbl, aaa)
    pd.set_option("display.width", 250)
    print(r["env"][["swings_cpbl", "W_cpbl", "swings_aaa", "W_aaa", "whiff_contrast", "pa_cpbl", "woba_cpbl", "pa_aaa",
                    "woba_aaa", "damage_contrast", "pa_share_cpbl", "pa_share_aaa"]].round(3).to_string())
    print(r["env"].attrs)
    P = r["pitcher"][r["pitcher"].fit_available]
    print(P[["name", "baseline_rank_tracked", "fit_rank", "combined_rank", "fit_mix_whiff", "fit_mix_damage",
             "aaa_whiff_pct_tracked", "k_minus_bb_pct_raw", "usage_FB94+", "tracked_pitches_used"]].head(20).round(3).to_string())
    H = r["hitter"][r["hitter"].fit_available]
    print(H[["name", "baseline_rank_tracked", "fit_rank", "combined_rank", "fit_type", "fit_velo", "fit_hand", "fit_total",
             "woba_raw", "tracked_pa_used"]].head(20).round(3).to_string())
