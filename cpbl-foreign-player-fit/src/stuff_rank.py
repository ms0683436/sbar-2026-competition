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


def _key() -> dict:
    """Everything the cached sums depend on (the run-value target uses the wOBA weights)."""
    c = config()
    return {"amendment8": c["amendment8"], "woba_weights": c["woba_weights"], "seasons": cfg()["seasons"]}


def game_sums(force: bool = False) -> pd.DataFrame:
    out, meta = processed("stuff_plus_game_sums.parquet"), processed("stuff_plus_game_sums.json")
    key = json.loads(json.dumps(_key()))
    if out.exists() and meta.exists() and not force and json.loads(meta.read_text()) == key:
        return pd.read_parquet(out)
    from . import stuff_plus
    a, _ = stuff_plus.fit_predict(cfg()["seasons"])
    a = a[a.season.isin(cfg()["seasons"])]
    g = (a.groupby(["pitcher_id", "season", "game_pk"]).agg(n=("pred", "size"), s=("pred", "sum"))
         .reset_index())
    g.to_parquet(out, index=False)
    meta.write_text(json.dumps(key))
    return g


def candidate_stuff(g: pd.DataFrame, w: pd.Series | None = None) -> pd.DataFrame:
    """Stuff+ per pitcher over the amendment-13 seasons; w = optional weight per game id (str)."""
    wt = 1.0 if w is None else g.game_pk.astype(str).map(w).fillna(1.0).to_numpy()
    ps = g.assign(n=g.n * wt, s=g.s * wt).groupby(["pitcher_id", "season"])[["n", "s"]].sum()
    ps = ps[ps.n > 0].reset_index()
    ps["pred_rv"] = ps.s / ps.n
    minp = config()["amendment8"]["min_pitches_scale"]
    out = []
    for _, d in ps.groupby("season"):
        ref = d[d.n >= minp]
        out.append(d.assign(stuff_plus=100 - 10 * (d.pred_rv - ref.pred_rv.mean()) / ref.pred_rv.std(ddof=0)))
    ps = pd.concat(out)
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
    a15 = "combined_a13" in P           # amendment 15: the Marcel score sits in the baseline slot
    base = P.marcel_score if a15 else P.baseline_score
    runs = ([("amendment 15 (Marcel 0.5 / Stuff+ 0.2 / Fit 0.3)", P.combined_heuristic),
             ("amendment 13 (old baseline 0.5 / Stuff+ 0.2 / Fit 0.3)", P.combined_a13)] if a15 else
            [("amendment 13 (0.5 / 0.2 / 0.3)", P.combined_heuristic)])
    for label, score in (*runs,
                         ("amendment 6 (0.7 old baseline / 0.3 Fit)", P.combined_a6),
                         *[(f"sensitivity Stuff+ {ws:.1f} ({'Marcel' if a15 else 'baseline'} {0.7 - ws:.1f})",
                            combined(base, P.stuff_plus, P.fit_score, ws)) for ws in cfg()["sensitivity_stuff_weights"]]):
        top = P.loc[score.index, "name"].reindex(score.sort_values(ascending=False).index)
        rows.append({"ranking": label, "signable top 3": ", ".join(top[sig.reindex(top.index)].head(3)),
                     "overall top 3": ", ".join(top.head(3))})
    out["picks"] = rows
    cols = ["name", "sign_tier", "combined_rank", *(["combined_rank_a13", "marcel_rank_tracked"] if a15 else []),
            "combined_rank_a6", "baseline_rank_tracked", "fit_rank", "stuff_plus", "stuff_pitches"]
    st = P[sig].sort_values("combined_rank").head(10)[cols]
    sh = pd.read_csv(processed("recommendation_shortlist.csv"))
    pk = sh[(sh.role == "pitcher") & (sh.rule_rank_signable == 1)]
    out["primary_pick"] = pk.name.iat[0] if len(pk) else None
    out["primary_pick_p_top1_signable"] = float(pk.boot_p_top1_signable.iat[0]) if len(pk) else None
    L = ["# Stuff+ in the pitcher ranking (amendment 13; generated by `python -m src.stuff_rank`)", "",
         f"combined = 0.5 z({'Marcel K-BB% (amendment 15)' if a15 else 'baseline'}) + 0.2 z(Stuff+) + 0.3 Fit over the Fit-available AAA candidates; Stuff+ from the "
         "amendment-8 model (2025 out-of-fold, 2026 from the model fit on 2025), pitch-weighted over 2025-26, "
         "missing (< 300 tracked pitches) = average.", "",
         f"Ranked pitchers {out['n_ranked']} (with Stuff+ {out['n_with_stuff']}); signable {out['n_signable']} "
         f"(with Stuff+ {out['n_signable_with_stuff']}).", "",
         f"Primary pick (signable top): **{out['primary_pick']}** (bootstrap P(top 1 among signable) "
         f"{out['primary_pick_p_top1_signable']:.2f}).", "",
         "## Picks under each ranking", "", md_table(pd.DataFrame(rows)), "",
         "## Signable top 10 (new ranking)", "", md_table(st, floatfmt=".1f"), ""]
    (path("reports") / "stuff_rank.md").write_text("\n".join(L), encoding="utf-8")
    processed("stuff_rank_summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    return out


if __name__ == "__main__":
    print(json.dumps(report(), indent=2, ensure_ascii=False, default=str))
