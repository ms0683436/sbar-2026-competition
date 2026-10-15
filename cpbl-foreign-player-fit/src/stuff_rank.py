"""Amendment 13 — Stuff+ in the pitcher ranking (rules in config/scoring.yaml, amendment13).

Candidate Stuff+ = pitch-weighted mean of the pitcher's 2025 (out-of-fold) and 2026 (model fit on
all of 2025) Stuff+, on the amendment-8 scale. Per-game sums of the predicted run value are cached,
so the evaluation bootstrap can re-weight AAA games without refitting the model.

Usage:  python -m src.stuff_rank      (after evaluate: writes reports/stuff_rank.md)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .common import config, md_table, path, processed, zscore


def cfg() -> dict:
    return config()["amendment13"]


def sum_seasons() -> list[int]:
    """Seasons with per-game sums: amendment 13's, plus amendment 18's Marcel seasons and the
    pre-arrival seasons of its backtest (each season is predicted on its own, so adding one does
    not change the others)."""
    s = set(cfg()["seasons"])
    if "amendment18" in config():
        s |= set(config()["amendment18"]["stuff"]["seasons"]) | {2023, 2024, 2025}
    return sorted(s)


def _key() -> dict:
    """Everything the cached sums depend on (the run-value target uses the wOBA weights)."""
    c = config()
    return {"amendment8": c["amendment8"], "woba_weights": c["woba_weights"], "seasons": sum_seasons()}


def game_sums(force: bool = False) -> pd.DataFrame:
    out, meta = processed("stuff_plus_game_sums.parquet"), processed("stuff_plus_game_sums.json")
    key = json.loads(json.dumps(_key()))
    if out.exists() and meta.exists() and not force and json.loads(meta.read_text()) == key:
        return pd.read_parquet(out)
    from . import stuff_plus
    a, _ = stuff_plus.fit_predict(sum_seasons())
    a = a[a.season.isin(sum_seasons())]
    g = (a.groupby(["pitcher_id", "season", "game_pk"]).agg(n=("pred", "size"), s=("pred", "sum"))
         .reset_index())
    g.to_parquet(out, index=False)
    meta.write_text(json.dumps(key))
    return g


def season_stuff(g: pd.DataFrame, w: pd.Series | None = None, key: str = "season") -> pd.DataFrame:
    """Stuff+ per pitcher x `key` (season, or a split-half label), each group scaled so its
    >= 300-pitch pitchers average 100 with SD 10 of predicted run value."""
    wt = 1.0 if w is None else g.game_pk.astype(str).map(w).fillna(1.0).to_numpy()
    ps = g.assign(n=g.n * wt, s=g.s * wt).groupby(["pitcher_id", key])[["n", "s"]].sum()
    ps = ps[ps.n > 0].reset_index()
    ps["pred_rv"] = ps.s / ps.n
    minp = config()["amendment8"]["min_pitches_scale"]
    out = []
    for _, d in ps.groupby(key):
        ref = d[d.n >= minp]
        sd = ref.pred_rv.std(ddof=0)
        if len(ref) < 2 or not sd > 0:
            continue                                  # no calibration for this group: it contributes nothing
        out.append(d.assign(stuff_plus=100 - 10 * (d.pred_rv - ref.pred_rv.mean()) / sd))
    cols = ["pitcher_id", key, "n", "s", "pred_rv", "stuff_plus"]
    return pd.concat(out) if out else pd.DataFrame(columns=cols)


def reliability(g: pd.DataFrame | None = None) -> dict:
    """Amendment 18 rule S: split-half (game_pk parity) reliability of season Stuff+ -> R."""
    c = config()["amendment18"]["stuff"]["reliability"]
    g = game_sums() if g is None else g
    h = g[g.season == c["season"]].assign(half=lambda d: d.game_pk % 2)
    ps = season_stuff(h, key="half").pivot_table(index="pitcher_id", columns="half", values=["stuff_plus", "n"])
    if not {("n", 0), ("n", 1)} <= set(ps.columns):
        return {"r": np.nan, "nbar": np.nan, "R": float(c["no_signal_R"]), "n_pitchers": 0}
    ok = ps[(ps[("n", 0)] >= c["min_pitches_each_half"]) & (ps[("n", 1)] >= c["min_pitches_each_half"])]
    if len(ok) < 3:
        return {"r": np.nan, "nbar": np.nan, "R": float(c["no_signal_R"]), "n_pitchers": int(len(ok))}
    r = float(np.corrcoef(ok[("stuff_plus", 0)], ok[("stuff_plus", 1)])[0, 1])
    nbar = float(ok[["n"]].to_numpy().mean())
    R = nbar * (1 - r) / r if r > 0 else float(c["no_signal_R"])
    return {"r": r, "nbar": nbar, "R": R, "n_pitchers": int(len(ok))}


_R: dict = {}


def candidate_stuff(g: pd.DataFrame, w: pd.Series | None = None, seasons: list[int] | None = None) -> pd.DataFrame:
    """Stuff+ per pitcher. Amendment 18 (rule S): seasons weighted 3 / 2 / 1 (latest first) and
    regressed toward 100 by R pitches, no minimum; `seasons` = the three seasons, latest first
    (default: amendment18.stuff.seasons). Without amendment 18: amendment 13 (pitch-weighted mean
    of its seasons, >= 300 pitches). w = optional weight per game id (str)."""
    if "amendment18" not in config():
        return candidate_stuff_a13(g, w)
    c = config()["amendment18"]["stuff"]
    seasons = seasons or c["seasons"]
    if "R" not in _R:
        _R.update(reliability(game_sums()))
    wts = dict(zip(seasons, c["weights"]))
    gs = g[g.season.isin(seasons)]
    ps = season_stuff(gs, w)
    ps["wn"] = ps.season.map(wts) * ps.n
    a = ps.assign(ws=ps.stuff_plus * ps.wn).groupby("pitcher_id")[["ws", "wn", "n"]].sum()
    # every pitcher with a tracked pitch in these seasons keeps a row: zero (re-)weighted exposure = 100
    a = a.reindex(pd.Index(gs.pitcher_id.unique(), name="pitcher_id"), fill_value=0.0)
    a["stuff_plus"] = (a.ws + _R["R"] * 100) / (a.wn + _R["R"])
    a = a[["stuff_plus", "n"]].rename(columns={"n": "stuff_pitches"}).reset_index()
    a["pitcher_id"] = a.pitcher_id.astype(int)
    return a


def candidate_stuff_a13(g: pd.DataFrame, w: pd.Series | None = None) -> pd.DataFrame:
    """Amendment 13: pitch-weighted mean over its seasons, NaN below 300 pitches."""
    ps = season_stuff(g[g.season.isin(cfg()["seasons"])], w)
    c = ps.assign(ws=ps.stuff_plus * ps.n).groupby("pitcher_id")[["ws", "n"]].sum()
    c["stuff_plus"] = np.where(c.n >= cfg()["min_pitches"], c.ws / c.n, np.nan)
    c = c[["stuff_plus", "n"]].rename(columns={"n": "stuff_pitches"}).reset_index()
    c["pitcher_id"] = c.pitcher_id.astype(int)          # MLBAM ids (stored as text in the pitch data)
    return c


def combined(base: pd.Series, stuff: pd.Series, fit: pd.Series, w_stuff: float | None = None) -> pd.Series:
    """0.5 z(baseline) + 0.2 z(Stuff+) + 0.3 Fit over the given (Fit-available) candidates; a
    different Stuff+ weight takes its share from the baseline (registered sensitivity)."""
    w = dict(cfg()["weights"])
    if w_stuff is not None:
        w["baseline"] += w["stuff"] - w_stuff
        w["stuff"] = w_stuff
    zs = zscore(stuff, cfg()["stuff_clip"]).fillna(0.0)
    return w["baseline"] * zscore(base) + w["stuff"] * zs + w["fit"] * np.nan_to_num(fit)


def report() -> dict:
    P = pd.read_csv(processed("pitcher_fit_results.csv"))
    P = P[P.fit_available].copy()
    sig = P.signable.fillna(False).astype(bool)
    out = {"n_ranked": len(P), "n_with_stuff": int(P.stuff_plus.notna().sum()),
           "n_signable": int(sig.sum()), "n_signable_with_stuff": int(P[sig].stuff_plus.notna().sum())}
    rows = []
    from . import marcel
    a15 = "marcel_score" in P and marcel.active("pitcher")   # amendment 15: Marcel in the baseline slot
    base = P.marcel_score if a15 else P.baseline_score
    a18 = "amendment18" in config() and "stuff_plus_a13" in P  # amendment 18: Marcel-style Stuff+
    bname = "Marcel" if a15 else "old baseline"
    runs = [(f"amendment {18 if a18 else 15 if a15 else 13} ({bname} 0.5 / "
             f"{'Marcel-style ' if a18 else ''}Stuff+ 0.2 / Fit 0.3)", P.combined_heuristic)]
    if a18 and a15:
        runs.append(("amendment 15 (Marcel 0.5 / amendment-13 Stuff+ 0.2 / Fit 0.3)", P.combined_a15))
    if a15 or a18:
        runs.append(("amendment 13 (old baseline 0.5 / Stuff+ 0.2 / Fit 0.3)", P.combined_a13))
    for label, score in (*runs,
                         ("amendment 6 (0.7 old baseline / 0.3 Fit)", P.combined_a6),
                         *[(f"sensitivity Stuff+ {ws:.1f} ({'Marcel' if a15 else 'baseline'} {0.7 - ws:.1f})",
                            combined(base, P.stuff_plus, P.fit_score, ws)) for ws in cfg()["sensitivity_stuff_weights"]]):
        top = P.loc[score.index, "name"].reindex(score.sort_values(ascending=False).index)
        rows.append({"ranking": label, "signable top 3": ", ".join(top[sig.reindex(top.index)].head(3)),
                     "overall top 3": ", ".join(top.head(3))})
    out["picks"] = rows
    if a18:
        out["stuff_reliability"] = reliability()
        out["stuff_rank_corr_a18_vs_a13"] = float(P.stuff_plus.corr(P.stuff_plus_a13, method="spearman"))
        prev = "combined_rank_a15" if a15 else "combined_rank_a13"
        out["combined_rank_corr_a18_vs_previous_signable"] = float(P[sig].combined_rank.corr(P[sig][prev],
                                                                                             method="spearman"))
        out["previous_ranking"] = "amendment 15" if a15 else "amendment 13"
        b = pd.read_csv(processed("bootstrap_ranks_pitchers.csv"))
        if "comba15_p_top1_signable" in b:          # same bootstrap draws, old and new Stuff+
            b = b.set_index("mlbam_id")
            names = list(dict.fromkeys(list(P[sig].nsmallest(3, "combined_rank").mlbam_id)
                                       + list(P[sig].nsmallest(3, prev).mlbam_id)))
            out["p_top1_before_after"] = [{"name": P.set_index("mlbam_id").name.get(i),
                                           "rank new": int(P.set_index("mlbam_id")[sig.to_numpy()].combined_rank.rank().get(i)),
                                           "rank previous": int(P.set_index("mlbam_id")[sig.to_numpy()][prev].rank().get(i)),
                                           "P(top 1) new": float(b.comb_p_top1_signable.get(i, np.nan)),
                                           "P(top 1) previous": float(b.comba15_p_top1_signable.get(i, np.nan))}
                                          for i in names]
    cols = ["name", "sign_tier", "combined_rank", *(["combined_rank_a15"] if a18 and a15 else []),
            *(["combined_rank_a13"] if a15 or a18 else []), *(["marcel_rank_tracked"] if a15 else []),
            "combined_rank_a6", "baseline_rank_tracked", "fit_rank", "stuff_plus",
            *(["stuff_plus_a13"] if a18 else []), "stuff_pitches"]
    st = P[sig].sort_values("combined_rank").head(10)[cols]
    sh = pd.read_csv(processed("recommendation_shortlist.csv"))
    pk = sh[(sh.role == "pitcher") & (sh.rule_rank_signable == 1)]
    out["primary_pick"] = pk.name.iat[0] if len(pk) else None
    out["primary_pick_p_top1_signable"] = float(pk.boot_p_top1_signable.iat[0]) if len(pk) else None
    L = ["# Stuff+ in the pitcher ranking (amendment 13; generated by `python -m src.stuff_rank`)", "",
         f"combined = 0.5 z({'Marcel K-BB% (amendment 15)' if a15 else 'baseline'}) + 0.2 z(Stuff+) + 0.3 Fit over the Fit-available AAA candidates; Stuff+ from the "
         "amendment-8 model (2025 out-of-fold, other seasons from the model fit on 2025); "
         + ("amendment 18: seasons 2026 / 2025 / 2024 weighted 3 / 2 / 1 and regressed toward 100 (below)."
            if a18 else "pitch-weighted over 2025-26, missing (< 300 tracked pitches) = average."), "",
         f"Ranked pitchers {out['n_ranked']} (with Stuff+ {out['n_with_stuff']}); signable {out['n_signable']} "
         f"(with Stuff+ {out['n_signable_with_stuff']}).", "",
         f"Primary pick (signable top): **{out['primary_pick']}** (bootstrap P(top 1 among signable) "
         f"{out['primary_pick_p_top1_signable']:.2f}).", "",
         *([f"Amendment 18: Stuff+ = seasons 2026 / 2025 / 2024 weighted 3 / 2 / 1, regressed toward 100 by "
            f"R = {out['stuff_reliability']['R']:.0f} pitches (split-half r {out['stuff_reliability']['r']:.3f} at "
            f"{out['stuff_reliability']['nbar']:.0f} pitches per half, {out['stuff_reliability']['n_pitchers']} pitchers), no "
            f"minimum. Rank correlation with the amendment-13 Stuff+: {out['stuff_rank_corr_a18_vs_a13']:.3f}; signable "
            f"combined-rank correlation with {out['previous_ranking']}: "
            f"{out['combined_rank_corr_a18_vs_previous_signable']:.3f}.", "",
            *(["Signable top 3 before and after rule S, bootstrap P(top 1 among signable) from the same draws "
               "(new = rule S, previous = amendment-13 Stuff+):", "",
               md_table(pd.DataFrame(out["p_top1_before_after"]), floatfmt=".2f"), ""]
              if out.get("p_top1_before_after") else [])]
           if a18 else []),
         "## Picks under each ranking", "", md_table(pd.DataFrame(rows)), "",
         "## Signable top 10 (new ranking)", "", md_table(st, floatfmt=".1f"), ""]
    (path("reports") / "stuff_rank.md").write_text("\n".join(L), encoding="utf-8")
    processed("stuff_rank_summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    return out


if __name__ == "__main__":
    print(json.dumps(report(), indent=2, ensure_ascii=False, default=str))
