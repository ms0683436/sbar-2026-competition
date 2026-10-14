"""Amendment 8 — AAA Stuff+ modelled on FanGraphs Stuff+, as a validation only.

FanGraphs' public description: a model of a pitch's run value from its physical
characteristics alone (no location, count or batter), separate models by pitch family, scaled
so 100 is average. Here (config `amendment8`, committed before any model was trained):

  inputs   velocity, spin rate, spin axis, pfx_x, pfx_z, release height z0, release side x0,
           extension (mirrored so arm side is positive); non-fastballs also velocity / pfx
           minus the pitcher-season mean of his primary fastball
  park     every input but spin axis minus a home-club offset estimated within the season
           from deviations around each pitcher x pitch type x season mean (altitude parks)
  target   pitch run value from 2025 AAA count values (value after - value before; terminal
           pitches take the PA's wOBA event value), divided by the wOBA scale
  model    HistGradientBoostingRegressor per family, fixed settings; 2025 values out-of-fold
           (folds grouped by pitcher), 2023-24 values from the model fit on all of 2025
  scale    pitcher Stuff+ = 100 - 10 * z(mean predicted run value per pitch), z over
           pitchers with >= 300 pitches in the season

Validation S1-S3 against the amendment-5 baseline on 2025 -> 2026 AAA; S4 (CPBL cohort) is
reported by src.backtest_2026. Nothing here changes the ranking or the picks.

Usage:  python -m src.stuff_plus
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from . import backtest, baseline, baseline_weights, milb_candidates, milb_loader, park, statsapi_release
from .common import config, md_table, path, processed

WOBA_SCALE = 1.20          # runs per wOBA point divisor (target scale only; Stuff+ is a z-score)
BASE = ["velo", "spin", "axis", "pfx_x", "pfx_z", "z0", "x0", "ext"]
DIFF = ["d_velo", "d_pfx_x", "d_pfx_z"]
PARK_ADJ = ["velo", "spin", "pfx_x", "pfx_z", "z0", "x0", "ext"]


def cfg() -> dict:
    return config()["amendment8"]


# ------------------------------------------------------------------ data
def pitches(seasons) -> pd.DataFrame:
    """AAA pitches (outcomes, count, pitch type) joined to the feed's release / spin-axis fields."""
    a = milb_loader.load()
    if min(seasons) < a.season.min():
        a = pd.concat([backtest.aaa_extra(), a], ignore_index=True)
    a = a[a.season.isin(seasons)].copy()
    a["game_pk"] = a.game_id.astype(int)
    a["at_bat_number"] = a.pa_id.str.split("-").str[1].astype(int)
    a["pitch_number"] = a.pitch_seq.astype(int)
    r = statsapi_release.load(seasons).drop(columns=["season", "pitcher"])
    r = r.rename(columns={"release_speed": "velo", "release_spin_rate": "spin", "spin_axis": "axis",
                          "release_extension": "ext", "pfx_x": "pfx_x_feed", "pfx_z": "pfx_z_feed"})
    a = a.merge(r, on=["game_pk", "at_bat_number", "pitch_number"], how="inner")
    a["pfx_x"], a["pfx_z"] = a.pfx_x_feed, a.pfx_z_feed
    lhp = a.p_throws.eq("L")
    a.loc[~lhp, "pfx_x"] = -a.loc[~lhp, "pfx_x"]          # arm side positive for both hands
    a.loc[~lhp, "x0"] = -a.loc[~lhp, "x0"]
    a.loc[~lhp, "axis"] = 360 - a.loc[~lhp, "axis"]
    fam = {t: f for f, ts in cfg()["families"].items() for t in ts}
    a["family"] = a.pitch_type_raw.map(fam)
    return a[a.family.notna() & a.velo.notna()]


def _home(a: pd.DataFrame) -> pd.Series:
    h = pd.concat([park.home_clubs(int(s)) for s in a.season.unique()]).drop_duplicates("game_pk")
    return a.game_pk.map(h.set_index("game_pk").home_id)


def park_offsets(a: pd.DataFrame) -> pd.DataFrame:
    """Home-club offset of each input, within season, from deviations around the pitcher x
    pitch type x season mean."""
    dev = a[PARK_ADJ] - a.groupby(["pitcher_id", "pitch_type_raw", "season"])[PARK_ADJ].transform("mean")
    off = dev.groupby([a.season, _home(a).rename("home_club")]).mean()
    off["n"] = dev.groupby([a.season, _home(a).rename("home_club")]).size()
    return off


def adjust(a: pd.DataFrame, off: pd.DataFrame) -> pd.DataFrame:
    a = a.assign(home_club=_home(a))
    o = off[PARK_ADJ].reindex(pd.MultiIndex.from_arrays([a.season, a.home_club])).fillna(0).to_numpy()
    a[PARK_ADJ] = a[PARK_ADJ].to_numpy() - o
    return a


def fastball_diffs(a: pd.DataFrame) -> pd.DataFrame:
    fb = a[a.pitch_type_raw.isin(["FF", "SI"])]
    prim = fb.groupby(["pitcher_id", "season"]).pitch_type_raw.agg(lambda s: s.value_counts().index[0])
    fc = a[a.pitch_type_raw.eq("FC")].groupby(["pitcher_id", "season"]).size()
    prim = prim.reindex(prim.index.union(fc.index)).fillna("FC")
    a = a.join(prim.rename("primary_fb"), on=["pitcher_id", "season"])
    ref = (a[a.pitch_type_raw.eq(a.primary_fb)].groupby(["pitcher_id", "season"])[["velo", "pfx_x", "pfx_z"]].mean()
           .add_prefix("fb_"))
    a = a.join(ref, on=["pitcher_id", "season"])
    for c in ("velo", "pfx_x", "pfx_z"):
        a[f"d_{c}"] = a[c] - a[f"fb_{c}"]
    return a


def run_values(a: pd.DataFrame) -> pd.Series:
    """Pitch run value from the training season's count values (wOBA scale)."""
    pa_end = a[a.pa_end.astype(bool)]
    val = pa_end.set_index("pa_id").woba_value.where(pa_end.set_index("pa_id").woba_denom > 0)
    v = a.pa_id.map(val)
    ok = v.notna() & ~a.groupby("pa_id").is_bunt.transform("any")
    cv = a[ok].assign(v=v[ok]).groupby(["balls", "strikes"]).v.mean()
    before = cv.reindex(pd.MultiIndex.from_arrays([a.balls, a.strikes])).to_numpy()
    nxt = a.groupby("pa_id")[["balls", "strikes"]].shift(-1)
    after = cv.reindex(pd.MultiIndex.from_arrays([nxt.balls.fillna(0).astype(int), nxt.strikes.fillna(0).astype(int)])).to_numpy()
    after = np.where(a.pa_end.astype(bool), v, after)
    rv = pd.Series((after - before) / WOBA_SCALE, index=a.index)
    return rv.where(ok)


def features(fam: str) -> list[str]:
    return BASE if fam == "FB" else BASE + DIFF


def _model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(**cfg()["model"])


def fit_predict() -> tuple[pd.DataFrame, dict]:
    """Per-pitch predicted run value for every predict season (2025 out-of-fold)."""
    c = cfg()
    tr_y = c["train_season"]
    seasons = sorted(set(c["predict_seasons"]) | {tr_y})
    a = pitches(seasons)
    off = park_offsets(a)
    a = fastball_diffs(adjust(a, off))
    a["rv"] = np.nan
    tr = a.season.eq(tr_y)
    a.loc[tr, "rv"] = run_values(a[tr])
    a["pred"] = np.nan
    diag = {}
    for fam in c["families"]:
        X = features(fam)
        t = a[tr & a.family.eq(fam) & a.rv.notna()]
        oof = pd.Series(np.nan, index=t.index)
        for k, (i, j) in enumerate(GroupKFold(c["folds"]).split(t, groups=t.pitcher_id)):
            m = _model().fit(t.iloc[i][X], t.iloc[i].rv)
            oof.iloc[j] = m.predict(t.iloc[j][X])
        a.loc[oof.index, "pred"] = oof
        full = _model().fit(t[X], t.rv)
        rest = a.family.eq(fam) & ~tr
        a.loc[rest, "pred"] = full.predict(a.loc[rest, X])
        tr_other = tr & a.family.eq(fam) & a.rv.isna()             # bunt / IBB PAs: predict with full model
        a.loc[tr_other, "pred"] = full.predict(a.loc[tr_other, X])
        diag[fam] = {"n_train": int(len(t)), "oof_corr": float(np.corrcoef(oof, t.rv)[0, 1])}
    return a, {"models": diag, "park_offsets": off.reset_index().to_dict("records")}


def pitcher_stuff(a: pd.DataFrame) -> pd.DataFrame:
    g = a.groupby(["pitcher_id", "season"]).agg(n=("pred", "size"), pred_rv=("pred", "mean")).reset_index()
    out = []
    for s, d in g.groupby("season"):
        ref = d[d.n >= cfg()["min_pitches_scale"]]
        z = (d.pred_rv - ref.pred_rv.mean()) / ref.pred_rv.std(ddof=0)
        out.append(d.assign(stuff_plus=100 - 10 * z))
    return pd.concat(out, ignore_index=True)


# ------------------------------------------------------------------ validation
def _pool(year: int, min_bf: int) -> pd.DataFrame:
    pf = park.factors_for([2025, 2026], milb_loader.load())
    s = baseline_weights.fetch("pitching", year, pf=pf)
    df = milb_candidates._combine(s, [c for c in milb_candidates.PITCH_SUM if c in s] + ["outs"], "pitching_BF")
    df["IP"] = df.outs / 3
    df = df[df.pitching_BF >= min_bf]
    empty = pd.DataFrame(columns=["player_id", "pitching_BF", "pitching_SO", "pitching_BB"])
    return baseline.pitcher_metrics(df, year, s, empty)


def _spear(x, y) -> float:
    return float(pd.Series(x).rank().corr(pd.Series(y).rank()))


def validate(stuff: pd.DataFrame) -> dict:
    v, b = cfg()["validation"], cfg()["bootstrap"]
    p25 = baseline.score(_pool(2025, v["min_bf"]), "pitcher", baseline.judgment_weights("pitcher"))
    p26 = _pool(v["target_season"], v["min_bf"])
    d = p25[["player_id", "baseline_score"]].merge(
        p26[["player_id", "fip", "k_pct_adj"]].rename(columns={"fip": "fip26", "k_pct_adj": "k26"}), on="player_id")
    st = stuff[(stuff.season == 2025) & (stuff.n >= cfg()["min_pitches_scale"])]
    d["pitcher_id"] = d.player_id.astype(int).astype(str)
    d = d.merge(st[["pitcher_id", "stuff_plus"]], on="pitcher_id")
    d["neg_fip26"] = -d.fip26
    rng = np.random.default_rng(b["seed"])
    a_ = (1 - b["ci"]) / 2
    res = {"n": len(d)}
    for name, tgt in (("S1 -FIP 2026", "neg_fip26"), ("S2 K% 2026", "k26")):
        rs, rb = _spear(d.stuff_plus, d[tgt]), _spear(d.baseline_score, d[tgt])
        diffs = []
        for _ in range(b["draws"]):
            i = rng.integers(0, len(d), len(d))
            x = d.iloc[i]
            diffs.append(_spear(x.stuff_plus.values, x[tgt].values) - _spear(x.baseline_score.values, x[tgt].values))
        lo, hi = np.quantile(diffs, [a_, 1 - a_])
        res[name] = {"rho_stuff": rs, "rho_baseline": rb, "diff": rs - rb, "ci": [float(lo), float(hi)],
                     "stuff_wins": bool(lo > 0)}
    k, c0 = np.polyfit(d.baseline_score, d.neg_fip26, 1)
    d["resid"] = d.neg_fip26 - (c0 + k * d.baseline_score)
    est, lo, hi = backtest.boot_ci(d, "stuff_plus", "resid", b["draws"], b["ci"], rng)
    res["S3 Stuff+ vs -FIP residual after baseline"] = {"rho": est, "ci": [lo, hi], "supports": bool(lo > 0)}
    res["stuff_vs_baseline_2025"] = _spear(d.stuff_plus, d.baseline_score)
    d.to_csv(processed("stuff_plus_validation.csv"), index=False)
    return res


def run() -> dict:
    a, diag = fit_predict()
    stuff = pitcher_stuff(a)
    stuff.to_csv(processed("stuff_plus_pitchers.csv"), index=False)
    res = validate(stuff)
    off = pd.DataFrame(diag["park_offsets"])
    out = {"models": diag["models"], "validation": res}
    processed("stuff_plus_summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float))
    big = off.assign(mag=off[["pfx_x", "pfx_z"]].abs().max(axis=1)).sort_values("mag", ascending=False).head(10)
    lines = ["# AAA Stuff+ validation (amendment 8; generated by `python -m src.stuff_plus`)", "",
             "Modelled on FanGraphs Stuff+ (physical characteristics only); AAA statsapi game feeds, trained on 2025.", "",
             "## Models (out-of-fold correlation with pitch run value)", "",
             md_table(pd.DataFrame(diag["models"]).T.reset_index(names="family"), floatfmt=".3f"), "",
             f"## Validation: 2025 Stuff+ vs 2026 AAA (n = {res['n']} pitchers)", "",
             "```", json.dumps({k: v for k, v in res.items() if k != "n"}, indent=2, default=float), "```", "",
             "## Largest park offsets (inches for pfx, mph, rpm, feet)", "",
             md_table(big.drop(columns="mag"), floatfmt=".2f"), ""]
    (path("reports") / "stuff_plus.md").write_text("\n".join(lines), encoding="utf-8")
    return out


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, ensure_ascii=False, default=float))
