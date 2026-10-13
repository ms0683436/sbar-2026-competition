"""Phase 1 — CPBL environment profile (+ the identically-defined AAA reference).

Outputs
  data/processed/cpbl_batter_profile.parquet       CPBL hitters as a population, per cell
  data/processed/cpbl_pitching_environment.parquet CPBL pitch usage / velocity / handedness
  data/processed/aaa_batter_profile.parquet        same metrics for AAA (reference league)
  data/processed/env_contrast.parquet              CPBL-vs-AAA pitch-type contrasts (H1)
  figures/*.png

Metric definitions (identical for both leagues, bunts removed):
  Whiff%  = whiffs / swings                     (swing = swinging strike, foul, foul tip, in play)
  Chase%  = swings on out-of-zone pitches / out-of-zone pitches
  K%      = strikeouts / PA ending on that cell (PA-ending pitch attributes, Savant convention)
  wOBA    = sum(woba_value) / sum(woba_denom) over PAs ending on that cell (generic weights)
  Hard%   = CPBL: REBAS 'hardness' == H among tagged balls in play (manual, broadcast-based)
            AAA : EV >= 95 mph among tracked balls in play          -> NOT comparable across leagues
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from . import milb_loader, rebas_loader
from .common import config, logit, path, processed, shrink_rate, wilson

CPBL_COLOR, AAA_COLOR = "#2a78d6", "#eb6834"
INK, INK2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
PITCH_ORDER = ["FF", "SI", "FC", "SL", "CU", "CH", "SPL"]


def add_bands(df: pd.DataFrame) -> pd.DataFrame:
    vb = config()["velocity_bands_mph"]
    fb = df.pitch_type.isin(["FF", "SI"])
    df["velo_band"] = pd.cut(df.velo_mph, vb["edges"], labels=vb["labels"], right=False).astype(object)
    df.loc[~fb, "velo_band"] = None
    df["platoon"] = np.where(df.p_throws.isna() | df.stand.isna(), None,
                             np.where(df.p_throws == df.stand, "same", "opposite"))
    df["hand_matchup"] = df.p_throws.fillna("?") + "HP v " + df.stand.fillna("?") + "HB"
    df["zone_cat"] = np.where(df.in_zone.isna(), None, np.where(df.in_zone == True, "in_zone", "out_of_zone"))  # noqa: E712
    return df


def cell_metrics(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    d = df[~df.is_bunt.astype(bool)].copy()
    d["ooz"] = d.in_zone.eq(False)
    d["ooz_swing"] = d.ooz & d.swing
    d["bip"] = d.pa_end & d.pa_event.isin(["1B", "2B", "3B", "HR", "OUT", "SF"])
    if d.league.iat[0] == "CPBL":
        d["hard_tag"] = d.bip & d.hardness.notna()
        d["hard"] = d.hard_tag & d.hardness.eq("H")
    else:
        d["hard_tag"] = d.bip & d.launch_speed.notna()
        d["hard"] = d.hard_tag & (d.launch_speed >= 95)
    d["is_so"] = d.pa_event.eq("SO")
    g = d.groupby(by, dropna=True, observed=True)
    m = g.agg(n_pitches=("swing", "size"), swings=("swing", "sum"), whiffs=("whiff", "sum"),
              ooz_pitches=("ooz", "sum"), ooz_swings=("ooz_swing", "sum"),
              pa=("pa_end", "sum"), so=("is_so", "sum"),
              woba_num=("woba_value", "sum"), woba_den=("woba_denom", "sum"),
              bip_tagged=("hard_tag", "sum"), hard=("hard", "sum"),
              n_pitchers=("pitcher_id", "nunique"), n_batters=("batter_id", "nunique")).reset_index()
    m["whiff_pct"] = m.whiffs / m.swings
    m["whiff_lo"], m["whiff_hi"] = wilson(m.whiffs, m.swings)
    m["chase_pct"] = m.ooz_swings / m.ooz_pitches
    m["chase_lo"], m["chase_hi"] = wilson(m.ooz_swings, m.ooz_pitches)
    m["k_pct"] = m.so / m.pa
    m["k_lo"], m["k_hi"] = wilson(m.so, m.pa)
    m["woba"] = m.woba_num / m.woba_den
    # normal-approx SE for wOBA (per-PA values are bounded; adequate for n>=100)
    m["woba_se"] = np.nan
    m["hard_pct"] = m.hard / m.bip_tagged
    return m


def _woba_se(df: pd.DataFrame, by: list[str]) -> pd.Series:
    pa = df[df.pa_end & (df.woba_denom > 0)]
    g = pa.groupby(by, observed=True).woba_value
    return (g.std(ddof=1) / np.sqrt(g.size())).rename("woba_se")


DIMENSIONS = {
    "overall": [],
    "pitch_type": ["pitch_type"],
    "pitch_group": ["pitch_group"],
    "pitcher_hand": ["p_throws"],
    "hand_matchup": ["hand_matchup"],
    "platoon": ["platoon"],
    "fb_velo_band": ["velo_band"],
    "zone": ["zone_cat"],
    "pitch_type_x_hand": ["pitch_type", "p_throws"],
}


def batter_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Long table: league x season(all/each) x dimension x level, with league deltas."""
    k = config()["samples"]["shrink_k_env_swings"]
    min_cell = config()["samples"]["min_cell_pitches_env"]
    out = []
    for season in ["ALL"] + sorted(df.season.unique().tolist()):
        d = df if season == "ALL" else df[df.season == season]
        d = d.assign(_all="all")
        lg = cell_metrics(d, ["_all"]).iloc[0]
        for dim, cols in DIMENSIONS.items():
            by = cols or ["_all"]
            m = cell_metrics(d, by)
            se = _woba_se(d, by)
            m = m.drop(columns="woba_se").merge(se.reset_index(), on=by, how="left")
            m["level"] = m[by].astype(str).agg(" | ".join, axis=1)
            m = m.drop(columns=by)
            m.insert(0, "dimension", dim)
            m.insert(0, "season", str(season))
            m.insert(0, "league", d.league.iat[0])
            m["league_whiff_pct"] = lg.whiff_pct
            m["whiff_delta_pp"] = 100 * (m.whiff_pct - lg.whiff_pct)
            m["whiff_shrunk"] = shrink_rate(m.whiffs, m.swings, lg.whiff_pct, k)
            m["league_chase_pct"] = lg.chase_pct
            m["chase_delta_pp"] = 100 * (m.chase_pct - lg.chase_pct)
            m["league_k_pct"] = lg.k_pct
            m["k_delta_pp"] = 100 * (m.k_pct - lg.k_pct)
            m["league_woba"] = lg.woba
            m["woba_delta"] = m.woba - lg.woba
            m["small_sample"] = m.n_pitches < min_cell
            out.append(m)
    return pd.concat(out, ignore_index=True)


def pitching_environment(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season in ["ALL"] + sorted(df.season.unique().tolist()):
        d = df if season == "ALL" else df[df.season == season]
        d = d[d.pitch_type.notna()]
        segs = {"all": d, "LHP": d[d.p_throws == "L"], "RHP": d[d.p_throws == "R"],
                "SP": d[d.pitcher_role == "SP"], "RP": d[d.pitcher_role == "RP"]}
        for seg, s in segs.items():
            n = len(s)
            for pt, x in s.groupby("pitch_type"):
                v = x.velo_mph.dropna()
                rows.append({"league": d.league.iat[0], "season": str(season), "metric_group": "pitch_usage",
                             "segment": seg, "pitch_type": pt, "n": len(x), "share": len(x) / n,
                             "velo_n": len(v), "velo_mean_mph": v.mean(), "velo_sd_mph": v.std(),
                             "velo_p10_mph": v.quantile(.1), "velo_p50_mph": v.median(),
                             "velo_p90_mph": v.quantile(.9), "velo_mean_kmh": v.mean() * 1.609344})
        fb = d[d.pitch_type.isin(["FF", "SI"]) & d.velo_band.notna()]
        for b, x in fb.groupby("velo_band"):
            rows.append({"league": d.league.iat[0], "season": str(season), "metric_group": "fb_velo_band",
                         "segment": "all", "pitch_type": f"FF/SI {b} mph", "n": len(x), "share": len(x) / len(fb)})
        pa = d[d.pa_end]
        for h, x in pa.groupby("p_throws"):
            rows.append({"league": d.league.iat[0], "season": str(season), "metric_group": "pitcher_hand_PA",
                         "segment": "all", "pitch_type": f"{h}HP", "n": len(x), "share": len(x) / len(pa)})
        for h, x in pa.groupby("stand"):
            rows.append({"league": d.league.iat[0], "season": str(season), "metric_group": "batter_hand_PA",
                         "segment": "all", "pitch_type": f"{h}HB", "n": len(x), "share": len(x) / len(pa)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ contrasts
def _agg_for_glm(df: pd.DataFrame) -> pd.DataFrame:
    d = df[~df.is_bunt.astype(bool) & df.swing & df.pitch_type.notna() & df.velo_mph.notna()
           & df.platoon.notna()].copy()
    # velocity relative to the pitch type's pooled mean (2 mph bins) -> matched comparison
    d["velo_c"] = d.velo_mph - d.groupby("pitch_type").velo_mph.transform("mean")
    d["velo_bin"] = (d.velo_c / 2).round() * 2
    d["two_strike"] = (d.strikes == 2).astype(int)
    a = (d.groupby(["league", "season", "pitch_type", "platoon", "two_strike", "velo_bin"], observed=True)
         .agg(whiffs=("whiff", "sum"), swings=("swing", "size")).reset_index())
    a["velo_c"] = a.velo_bin
    return a


def whiff_contrast(cpbl: pd.DataFrame, aaa: pd.DataFrame, season_pairs=None) -> pd.DataFrame:
    """Adjusted CPBL-vs-AAA whiff contrast per pitch type.

    Binomial GLM on aggregated swings:
      logit P(whiff) ~ league * pitch_type + pitch_type:velo_c + platoon + two_strike
    The league:pitch_type terms give how much MORE (or less) CPBL hitters whiff on a
    pitch type than AAA hitters, at equal velocity, platoon and count. We report:
      raw_logit_diff   : unadjusted log-odds difference
      adj_logit_diff   : model-based difference (velocity/platoon/count adjusted)
      centered_contrast: adj_logit_diff minus its AAA-swing-weighted mean
                         => pitch-type-SPECIFIC weakness net of the overall level gap
    """
    if season_pairs is None:
        season_pairs = {"pooled": (None, None), "2024": (2024, 2024), "2025": (2025, 2025)}
    out = []
    for label, (cs, as_) in season_pairs.items():
        c = cpbl if cs is None else cpbl[cpbl.season == cs]
        a = aaa if as_ is None else aaa[aaa.season == as_]
        agg = _agg_for_glm(pd.concat([c, a]))
        agg["league"] = pd.Categorical(agg.league, ["AAA", "CPBL"])
        agg["pitch_type"] = pd.Categorical(agg.pitch_type, PITCH_ORDER)
        agg["fails"] = agg.swings - agg.whiffs
        model = smf.glm("whiffs + fails ~ league * pitch_type + pitch_type:velo_c + platoon + two_strike",
                        data=agg, family=sm.families.Binomial()).fit()
        null = smf.glm("whiffs + fails ~ league + pitch_type + pitch_type:velo_c + platoon + two_strike",
                       data=agg, family=sm.families.Binomial()).fit()
        lr = 2 * (model.llf - null.llf)
        dof = model.df_model - null.df_model
        from scipy.stats import chi2
        p_inter = chi2.sf(lr, dof)
        # per-type adjusted difference = league coef + interaction coef
        cov = model.cov_params()
        base = "league[T.CPBL]"
        tot = agg.groupby(["league", "pitch_type"], observed=True)[["whiffs", "swings"]].sum()
        aaa_w = tot.loc["AAA"].swings / tot.loc["AAA"].swings.sum()
        rows = []
        for pt in PITCH_ORDER:
            term = f"league[T.CPBL]:pitch_type[T.{pt}]"
            if pt == PITCH_ORDER[0]:
                est, var = model.params[base], cov.loc[base, base]
            else:
                est = model.params[base] + model.params[term]
                var = cov.loc[base, base] + cov.loc[term, term] + 2 * cov.loc[base, term]
            cw, cn = tot.loc[("CPBL", pt)]
            aw, an = tot.loc[("AAA", pt)]
            rows.append({"pair": label, "pitch_type": pt, "cpbl_swings": cn, "cpbl_whiff_pct": cw / cn,
                         "aaa_swings": an, "aaa_whiff_pct": aw / an,
                         "raw_logit_diff": logit(cw / cn) - logit(aw / an),
                         "adj_logit_diff": est, "adj_se": np.sqrt(var)})
        r = pd.DataFrame(rows)
        w = aaa_w.reindex(r.pitch_type).values
        mean = np.sum(w * r.adj_logit_diff)
        r["level_gap"] = mean
        r["centered_contrast"] = r.adj_logit_diff - mean
        # SE of centered contrast ignoring the (small) covariance with the weighted mean
        r["centered_lo"] = r.centered_contrast - 1.96 * r.adj_se
        r["centered_hi"] = r.centered_contrast + 1.96 * r.adj_se
        r["interaction_LR"] = lr
        r["interaction_df"] = dof
        r["interaction_p"] = p_inter
        out.append(r)
    return pd.concat(out, ignore_index=True)


def damage_contrast(cpbl: pd.DataFrame, aaa: pd.DataFrame) -> pd.DataFrame:
    """wOBA on PAs ending with each pitch type, CPBL vs AAA, centred on the overall gap."""
    rows = []
    for label, (cs, as_) in {"pooled": (None, None), "2024": (2024, 2024), "2025": (2025, 2025)}.items():
        c = cpbl if cs is None else cpbl[cpbl.season == cs]
        a = aaa if as_ is None else aaa[aaa.season == as_]
        res = {}
        for lg, d in (("CPBL", c), ("AAA", a)):
            p = d[d.pa_end & (d.woba_denom > 0) & d.pitch_type.notna()]
            g = p.groupby("pitch_type").woba_value
            res[lg] = pd.DataFrame({"pa": g.size(), "woba": g.mean(), "se": g.std() / np.sqrt(g.size())})
        r = res["CPBL"].join(res["AAA"], lsuffix="_cpbl", rsuffix="_aaa").reindex(PITCH_ORDER)
        r["diff"] = r.woba_cpbl - r.woba_aaa
        w = r.pa_aaa / r.pa_aaa.sum()
        r["level_gap"] = (w * r["diff"]).sum()
        r["centered_contrast"] = r["diff"] - r.level_gap
        r["se"] = np.sqrt(r.se_cpbl**2 + r.se_aaa**2)
        r["centered_lo"] = r.centered_contrast - 1.96 * r.se
        r["centered_hi"] = r.centered_contrast + 1.96 * r.se
        r.insert(0, "pair", label)
        rows.append(r.reset_index())
    return pd.concat(rows, ignore_index=True)


# ------------------------------------------------------------------ figures
def _style(ax, title, xlabel=None, ylabel=None):
    import matplotlib.pyplot as plt  # noqa: F401
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)


def figures(prof: pd.DataFrame, env: pd.DataFrame, wc: pd.DataFrame, cpbl: pd.DataFrame, aaa: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir = path("figures")

    # 1. whiff% by pitch type, both leagues, Wilson CI
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=SURFACE)
    x = np.arange(len(PITCH_ORDER))
    for i, (lg, col) in enumerate((("CPBL", CPBL_COLOR), ("AAA", AAA_COLOR))):
        p = prof[(prof.league == lg) & (prof.season == "ALL") & (prof.dimension == "pitch_type")].set_index("level")
        p = p.reindex(PITCH_ORDER)
        ax.bar(x + (i - .5) * .38, 100 * p.whiff_pct, .36, color=col, label=lg)
        ax.errorbar(x + (i - .5) * .38, 100 * p.whiff_pct,
                    yerr=[100 * (p.whiff_pct - p.whiff_lo), 100 * (p.whiff_hi - p.whiff_pct)],
                    fmt="none", ecolor=INK2, lw=1, capsize=2)
    ax.set_xticks(x, PITCH_ORDER)
    _style(ax, "Whiff% (whiffs / swings) by pitch type — CPBL 2024-25 vs AAA 2024-25", ylabel="Whiff %")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(fig_dir / "fig1_whiff_by_pitch_type.png", dpi=150); plt.close(fig)

    # 2. pitch usage
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=SURFACE)
    for i, (lg, col) in enumerate((("CPBL", CPBL_COLOR), ("AAA", AAA_COLOR))):
        e = env[(env.league == lg) & (env.season == "ALL") & (env.metric_group == "pitch_usage")
                & (env.segment == "all")].set_index("pitch_type").reindex(PITCH_ORDER)
        ax.bar(x + (i - .5) * .38, 100 * e.share, .36, color=col, label=lg)
    ax.set_xticks(x, PITCH_ORDER)
    _style(ax, "Pitch usage share — CPBL vs AAA (2024-25)", ylabel="% of classified pitches")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(fig_dir / "fig2_pitch_usage.png", dpi=150); plt.close(fig)

    # 3. four-seam velocity distribution — in km/h with 1 km/h bins centred on integers,
    #    because REBAS records whole km/h (mph bins would alias into a comb pattern)
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=SURFACE)
    bins = np.arange(127.5, 163.5, 1.0)
    for lg, d, col in (("CPBL", cpbl, CPBL_COLOR), ("AAA", aaa, AAA_COLOR)):
        v = d[(d.pitch_type == "FF")].velo_mph.dropna() * 1.609344
        ax.hist(v, bins=bins, density=True, histtype="step", lw=2, color=col,
                label=f"{lg} (n={len(v):,}, mean {v.mean():.1f} km/h = {v.mean()/1.609344:.1f} mph)")
    _style(ax, "Four-seam velocity (CPBL broadcast radar, whole km/h; AAA Statcast)", xlabel="km/h",
           ylabel="density")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(fig_dir / "fig3_ff_velocity.png", dpi=150); plt.close(fig)

    # 4. centred contrast by season pair — the H1 picture
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=SURFACE)
    cols = {"2024": "#1baf7a", "2025": "#6250d6", "pooled": INK}
    for j, (pair, sub) in enumerate(wc.groupby("pair", sort=False)):
        sub = sub.set_index("pitch_type").reindex(PITCH_ORDER)
        off = (j - 1) * 0.22
        ax.errorbar(x + off, sub.centered_contrast, yerr=1.96 * sub.adj_se, fmt="o", ms=5,
                    color=cols.get(pair, INK2), lw=1.2, capsize=2, label=pair)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(x, PITCH_ORDER)
    _style(ax, "CPBL-specific whiff weakness by pitch type (adjusted log-odds, centred)",
           ylabel="+ = CPBL hitters whiff relatively MORE")
    ax.legend(frameon=False, fontsize=9, title="CPBL vs AAA season", title_fontsize=8)
    fig.tight_layout(); fig.savefig(fig_dir / "fig4_cpbl_contrast.png", dpi=150); plt.close(fig)


def run() -> dict:
    cpbl = add_bands(rebas_loader.load())
    aaa = add_bands(milb_loader.load())
    prof_c = batter_profile(cpbl)
    prof_a = batter_profile(aaa)
    env_c = pitching_environment(cpbl)
    env_a = pitching_environment(aaa)
    prof_c.to_parquet(processed("cpbl_batter_profile.parquet"), index=False)
    prof_a.to_parquet(processed("aaa_batter_profile.parquet"), index=False)
    env_c.to_parquet(processed("cpbl_pitching_environment.parquet"), index=False)
    env_a.to_parquet(processed("aaa_pitching_environment.parquet"), index=False)
    wc = whiff_contrast(cpbl, aaa)
    dc = damage_contrast(cpbl, aaa)
    wc.to_parquet(processed("env_whiff_contrast.parquet"), index=False)
    dc.to_parquet(processed("env_damage_contrast.parquet"), index=False)
    figures(pd.concat([prof_c, prof_a]), pd.concat([env_c, env_a]), wc, cpbl, aaa)
    return {"prof": pd.concat([prof_c, prof_a]), "env": pd.concat([env_c, env_a]), "wc": wc, "dc": dc}


if __name__ == "__main__":
    r = run()
    pd.set_option("display.width", 220)
    print(r["wc"][["pair", "pitch_type", "cpbl_whiff_pct", "aaa_whiff_pct", "raw_logit_diff", "adj_logit_diff",
                   "adj_se", "centered_contrast", "interaction_p"]].round(3).to_string())
    print(r["dc"][["pair", "pitch_type", "pa_cpbl", "woba_cpbl", "pa_aaa", "woba_aaa", "centered_contrast",
                   "se"]].round(3).to_string())
