"""Amendment 14 — two-strike "battle" metrics, CPBL (REBAS) vs AAA, and the pick appendix.

Rules in config/scoring.yaml (amendment14). Descriptive / exploratory: nothing enters a score.

Part A  league comparison 2024-25 (complete, audited PAs; bunt / IBB / pitchout / interference PAs dropped)
Part B  exploratory: does a pitcher's pre-arrival AAA two-strike put-away rate say anything about
        his CPBL K-BB% beyond his AAA K-BB%? (amendment-11 cohort; tiny n, outcomes already known)
Part C  appendix for the final primary pitcher pick: AAA usage and results by pitch type x batter side

A continuation foul is a non-terminal, non-bunt swing without a whiff at two strikes (a foul tip
with two strikes is a strikeout, so it is terminal); this needs no raw call codes in either source.
CPBL outcomes: 資料來源：野球革命 Rebas Open Data.

Usage:  python -m src.battle          (Parts A and B; reports/battle.md)
        python -m src.battle appendix (Part C for the current primary pick; reports/pick_appendix.md)
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from . import backtest, milb_loader, rebas_loader
from .common import config, md_table, path, processed

CELL = ["balls", "stand", "p_throws", "pitcher_role"]
METRICS = {   # name: (numerator, denominator, level)
    "A1 continuation-foul rate": ("foul2", "n2", "pitch"),
    "A2 foul per swing": ("foul2", "sw2", "swing"),
    "A3 contact": ("contact2", "sw2", "swing"),
    "A4 put-away (pitch)": ("so2", "n2", "pitch"),
    "A5 conversion (PA)": ("so_pa", "pa2", "pa"),
    "A7 two-strike pitches per PA reaching two strikes": ("n2_pa", "pa2", "pa"),
}


def cfg() -> dict:
    return config()["amendment14"]


# ------------------------------------------------------------------ data
BALL_CODES = {"B", "*B", "I", "P"}          # AAA call codes (REBAS: "B"); I / P drop the PA anyway
DROP_CODES = {"I", "P", "Q", "R"}           # intentional ball, pitchouts (AAA only; REBAS records none)


def _audit(d: pd.DataFrame) -> pd.DataFrame:
    """Per PA: complete, no special play, consistent sequence. Consistent = pitches numbered 1..n,
    first pitch at 0-0, every next count equal to the count implied by the previous pitch (a jump,
    e.g. from an automatic ball or strike that the loader drops, excludes the PA), no non-terminal
    foul tip at two strikes, strikeouts / walks ending at a full count of strikes / balls."""
    d = d.sort_values(["pa_id", "pitch_seq"])
    g = d.groupby("pa_id", sort=False)
    last = g.tail(1).set_index("pa_id")
    code = d.pitch_code.astype(str)
    term = d.pa_end.astype(bool)
    ball = code.isin(BALL_CODES)
    foul = code.eq("F")
    strike = d.called_strike.astype(bool) | d.whiff.astype(bool) | code.isin(["T", "FT"]) | (foul & d.strikes.lt(2))
    pa = pd.DataFrame(index=last.index)
    pa["complete"] = last.pa_end.astype(bool)
    pa["special"] = (g.is_bunt.any() | last.pa_event.isin(["IBB", "SH", "CI", "OTHER"]).reindex(pa.index)
                     | code.isin(DROP_CODES).groupby(d.pa_id).any().reindex(pa.index))
    nb, ns = d.balls + ball.astype(int), d.strikes + strike.astype(int)
    same = d.pa_id.eq(d.pa_id.shift(-1))
    jump = same & ((d.balls.shift(-1) != nb) | (d.strikes.shift(-1) != ns.clip(upper=2)))
    first = d.pitch_seq.eq(1) & d.balls.eq(0) & d.strikes.eq(0)
    misnumbered = d.pitch_seq.ne(g.cumcount() + 1)
    bad_ft = code.isin(["T", "FT"]) & d.strikes.eq(2) & ~term
    bad = (jump | bad_ft | misnumbered).groupby(d.pa_id).any() | ~first.groupby(d.pa_id).first()
    so_ok = ~last.pa_event.eq("SO") | last.strikes.eq(2)
    bb_ok = ~last.pa_event.eq("uBB") | (last.balls.eq(3) & code.groupby(d.pa_id).last().isin(BALL_CODES))
    pa["consistent"] = ~bad.reindex(pa.index) & so_ok & bb_ok
    pa["keep"] = pa.complete & ~pa.special & pa.consistent
    return pa


def load(seasons=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pitch rows of kept PAs (both leagues) and the audit table."""
    seasons = seasons or cfg()["seasons"]
    cols = ["league", "season", "game_id", "pa_id", "pitch_seq", "pitcher_id", "batter_id", "p_throws", "stand",
            "pitcher_role", "balls", "strikes", "pitch_type_raw", "pitch_group", "swing", "whiff", "is_bunt",
            "called_strike", "pa_end", "pa_event", "pitch_code"]
    c = rebas_loader.load()[cols]
    a = milb_loader.load()[cols]
    d = pd.concat([c[c.season.isin(seasons)], a[a.season.isin(seasons)]], ignore_index=True)
    d["pa_id"] = d.league + "-" + d.pa_id.astype(str)
    aud = _audit(d)
    lg = d.drop_duplicates("pa_id").set_index("pa_id")[["league", "season"]]   # before dropping PAs
    audit = aud.join(lg, how="left")
    audit.attrs["box"] = box_check(d)
    d = d[d.pa_id.map(aud.keep)].copy()
    return d, audit


def _flags(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    two = d.strikes.eq(2)
    term = d.pa_end.astype(bool)
    sw = d.swing.astype(bool) & ~d.is_bunt.astype(bool)
    d["n2"] = two.astype(int)
    d["sw2"] = (two & sw).astype(int)
    d["foul2"] = (two & d.pitch_code.astype(str).eq("F") & ~term).astype(int)    # ordinary foul, not a tip
    d["contact2"] = (two & sw & ~d.whiff.astype(bool)).astype(int)
    d["so2"] = (two & term & d.pa_event.eq("SO")).astype(int)
    d["whiff1"] = (d.strikes.eq(1) & sw & d.whiff.astype(bool)).astype(int)
    return d


def pa_table(d: pd.DataFrame) -> pd.DataFrame:
    """One row per PA: pitches, whether / at which ball count it reached two strikes, outcome."""
    d = d.sort_values(["pa_id", "pitch_seq"])
    g = d.groupby("pa_id", sort=False)
    p = g.agg(league=("league", "first"), season=("season", "first"), game_id=("game_id", "first"),
              pitcher_id=("pitcher_id", "first"), stand=("stand", "first"), p_throws=("p_throws", "first"),
              pitcher_role=("pitcher_role", "first"), pitches=("pitch_seq", "size"), event=("pa_event", "last"),
              n2_pa=("n2", "sum"))
    first2 = d[d.strikes.eq(2)].groupby("pa_id").balls.first()
    p["balls"] = first2.reindex(p.index)
    p["pa2"] = p.balls.notna().astype(int)
    p["so_pa"] = (p.pa2.astype(bool) & p.event.eq("SO")).astype(int)
    return p


# ------------------------------------------------------------------ part A
def _game_cells(d: pd.DataFrame, p: pd.DataFrame) -> dict:
    """Per league-season: game x cell matrices of every numerator / denominator."""
    out = {}
    pitch_q = ["n2", "sw2", "foul2", "contact2", "so2"]
    pa_q = ["pa2", "so_pa", "n2_pa"]
    two = d[d.strikes.eq(2)].copy()
    p2 = p[p.pa2.eq(1)].copy()
    for df in (two, p2):
        # a missing hand (1.5% of CPBL pitches) is its own cell, so it falls outside common support
        df["cell"] = df[CELL].astype(object).fillna("NA").astype(str).agg("|".join, axis=1)
    for (lg, yr), g in d.groupby(["league", "season"]):
        games = pd.Index(sorted(g.game_id.unique()))
        tp = two[(two.league == lg) & (two.season == yr)]
        pp = p2[(p2.league == lg) & (p2.season == yr)]
        allp = p[(p.league == lg) & (p.season == yr)]
        mats = {}
        for q, src in [(q, tp) for q in pitch_q] + [(q, pp) for q in pa_q]:
            mats[q] = src.pivot_table(index="game_id", columns="cell", values=q, aggfunc="sum").reindex(games).fillna(0)
        mats["pitches"] = allp.groupby("game_id").pitches.sum().reindex(games).fillna(0)
        mats["pas"] = allp.groupby("game_id").size().reindex(games).fillna(0)
        mats["pa2_all"] = allp.groupby("game_id").pa2.sum().reindex(games).fillna(0)
        for lo, hi in ((1, 3), (4, 5), (6, 7), (8, 99)):
            mats[f"len_{lo}"] = allp[allp.pitches.between(lo, hi)].groupby("game_id").size().reindex(games).fillna(0)
        out[(lg, int(yr))] = mats
    return out


def _std_rate(num: pd.Series, den: pd.Series, target: pd.Series) -> float:
    """Standardised rate: this league's cell rates weighted by the common target shares."""
    return float((num / den * target).sum() / target.sum())


def _season_stats(cpbl: dict, aaa: dict) -> dict:
    """Raw and standardised metrics of both leagues for one season from total vectors. Target =
    the equal-weighted average of the two leagues' full cell distributions, restricted to cells
    with a positive denominator in both leagues (common support); the target mass outside is
    reported as excluded weight."""
    res = {}
    for name, (num, den, _) in METRICS.items():
        cells = cpbl[den].index.union(aaa[den].index)
        cd, ad = cpbl[den].reindex(cells, fill_value=0), aaa[den].reindex(cells, fill_value=0)
        target = (cd / cd.sum() + ad / ad.sum()) / 2
        common = (cd > 0) & (ad > 0)
        t = target[common]
        cn, an = cpbl[num].reindex(cells, fill_value=0)[common], aaa[num].reindex(cells, fill_value=0)[common]
        res[name] = {"cpbl_raw": float(cpbl[num].sum() / cpbl[den].sum()), "aaa_raw": float(aaa[num].sum() / aaa[den].sum()),
                     "cpbl_std": _std_rate(cn, cd[common], t), "aaa_std": _std_rate(an, ad[common], t),
                     "excluded_weight": float(1 - t.sum())}
        res[name]["diff"] = res[name]["cpbl_std"] - res[name]["aaa_std"]
    for lg, t in (("cpbl", cpbl), ("aaa", aaa)):
        res.setdefault("A6 pitches per PA", {})[f"{lg}_raw"] = float(t["pitches"] / t["pas"])
        res.setdefault("A8 share of PAs reaching two strikes", {})[f"{lg}_raw"] = float(t["pa2_all"] / t["pas"])
        for lo in (1, 4, 6, 8):
            res.setdefault(f"A6 PA length {lo}{'+' if lo == 8 else '-' + str({1: 3, 4: 5, 6: 7}[lo])}", {})[f"{lg}_raw"] = float(t[f"len_{lo}"] / t["pas"])
    for k in [k for k in res if k.startswith(("A6", "A8"))]:
        res[k]["diff"] = res[k]["cpbl_raw"] - res[k]["aaa_raw"]
    return res


def _totals(m: dict, k: np.ndarray | None = None) -> dict:
    out = {}
    for q, v in m.items():
        if isinstance(v, pd.DataFrame):
            out[q] = pd.Series(v.to_numpy().T @ k if k is not None else v.to_numpy().sum(axis=0), index=v.columns)
        else:
            out[q] = float(v.to_numpy() @ k) if k is not None else float(v.sum())
    return out


def part_a(d: pd.DataFrame, p: pd.DataFrame) -> dict:
    b = cfg()["bootstrap"]
    rng = np.random.default_rng(b["seed"])
    mats = _game_cells(d, p)
    seasons = sorted({s for _, s in mats})
    est = {s: _season_stats(_totals(mats[("CPBL", s)]), _totals(mats[("AAA", s)])) for s in seasons}
    names = list(est[seasons[0]])
    draws = {n: {s: [] for s in seasons} for n in names}
    for _ in range(b["draws"]):
        for s in seasons:
            tc, ta = [_totals(mats[(lg, s)], np.bincount(rng.integers(0, len(mats[(lg, s)]["pas"]), len(mats[(lg, s)]["pas"])),
                                                         minlength=len(mats[(lg, s)]["pas"])).astype(float))
                      for lg in ("CPBL", "AAA")]
            st = _season_stats(tc, ta)
            for n in names:
                draws[n][s].append(st[n]["diff"])
    q = (1 - b["ci"]) / 2
    rows = []
    for n in names:
        D = np.column_stack([draws[n][s] for s in seasons])
        r = {"metric": n}
        for i, s in enumerate(seasons):
            e = est[s][n]
            r.update({f"cpbl_{s}": e.get("cpbl_std", e.get("cpbl_raw")), f"aaa_{s}": e.get("aaa_std", e.get("aaa_raw")),
                      f"diff_{s}": e["diff"], f"ci_{s}": [float(np.nanquantile(D[:, i], q)), float(np.nanquantile(D[:, i], 1 - q))]})
            if "cpbl_std" in e:
                r[f"cpbl_raw_{s}"], r[f"aaa_raw_{s}"] = e["cpbl_raw"], e["aaa_raw"]
                r[f"excluded_weight_{s}"] = e["excluded_weight"]
        mean = D.mean(axis=1)
        r["diff_mean"] = float(np.mean([est[s][n]["diff"] for s in seasons]))
        r["ci_mean"] = [float(np.nanquantile(mean, q)), float(np.nanquantile(mean, 1 - q))]
        rows.append(r)
    a1 = rows[0]
    th = cfg()["a1_threshold_pp"] / 100
    both_pos = all(a1[f"diff_{s}"] > 0 for s in seasons)
    verdict = ("CPBL hitters battle more" if a1["ci_mean"][0] > th and both_pos else
               "small difference" if a1["ci_mean"][0] > -th and a1["ci_mean"][1] < th else "unresolved")
    return {"rows": rows, "a1_verdict": verdict, "seasons": seasons}


def within_pitcher(d: pd.DataFrame) -> dict:
    """Supplement: two-strike minus one-strike continuation-foul rate (A1, per pitch) and whiff per
    swing, within pitcher-season x balls x pitch family x batter hand; cell weight = harmonic mean
    of the two counts (descriptive)."""
    s = d[d.strikes.isin([1, 2]) & ~d.is_bunt.astype(bool)].copy()
    sw = s.swing.astype(bool)
    s["foul"] = (s.pitch_code.astype(str).eq("F") & ~s.pa_end.astype(bool)).astype(int)
    s["wh"] = (sw & s.whiff.astype(bool)).astype(int)
    s["sw"] = sw.astype(int)
    keys = ["pitcher_id", "balls", "pitch_group", "stand"]
    out = {}
    for (lg, yr), g in s.groupby(["league", "season"]):
        a = g.groupby(keys + ["strikes"]).agg(n=("foul", "size"), foul=("foul", "sum"), sw=("sw", "sum"),
                                               wh=("wh", "sum")).unstack("strikes").dropna()
        w = 2 / (1 / a[("n", 1)] + 1 / a[("n", 2)])
        f = a[("foul", 2)] / a[("n", 2)] - a[("foul", 1)] / a[("n", 1)]
        b = a[(a[("sw", 1)] > 0) & (a[("sw", 2)] > 0)]
        ws = 2 / (1 / b[("sw", 1)] + 1 / b[("sw", 2)])
        wc = b[("wh", 2)] / b[("sw", 2)] - b[("wh", 1)] / b[("sw", 1)]
        out[f"{lg}_{int(yr)}"] = {"foul_change": float(np.average(f, weights=w)),
                                  "whiff_change": float(np.average(wc, weights=ws)), "cells": int(len(a))}
    return out


def box_check(d_all: pd.DataFrame) -> pd.DataFrame:
    """Audit: PA / SO / BB in the pitch sequences vs season box totals (AAA: statsapi season stats,
    no park adjustment; CPBL: no league box totals in hand, see the amendment-12 per-pitcher check),
    plus PAs whose pitch numbering does not start at 1 or has gaps."""
    from . import marcel
    last = d_all.sort_values(["pa_id", "pitch_seq"]).groupby("pa_id").tail(1)
    seq = d_all.groupby("pa_id").pitch_seq.agg(["min", "max", "size"])
    bad = ((seq["min"] != 1) | (seq["max"] != seq["size"])).rename("bad_seq")
    last = last.join(bad, on="pa_id")
    rows = []
    for (lg, yr), g in last.groupby(["league", "season"]):
        r = {"league": lg, "season": int(yr), "games": g.game_id.nunique(), "pa": int(g.pa_end.sum()),
             "so": int(g.pa_event.eq("SO").sum()), "bb": int(g.pa_event.isin(["uBB", "IBB"]).sum()),
             "pa_bad_numbering": int(g.bad_seq.sum())}
        if lg == "AAA":
            b = marcel.stints("pitching", int(yr), neutral=True)
            r.update({"box_bf": int(b.pitching_BF.sum()), "box_so": int(b.pitching_SO.sum()),
                      "box_bb": int(b.pitching_BB.sum())})
        rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ part B
def part_b() -> dict:
    c = cfg()["part_b"]
    coh = pd.read_csv(processed("backtest_chrono_cohort.csv"))
    a2 = pd.read_csv(processed("backtest_cohort.csv"))
    first = {(int(r.mlbam_id), int(r.season)): r.cpbl_first_date for r in a2.itertuples(index=False)}
    aaa = pd.concat([backtest.aaa_extra(), milb_loader.load()], ignore_index=True)
    rows = []
    for r in coh.itertuples(index=False):
        if (int(r.mlbam_id), int(r.season)) in first:            # amendment-2 window: 365 days before the first game
            f = pd.Timestamp(first[(int(r.mlbam_id), int(r.season))])
            lo = f - pd.Timedelta(days=backtest.cfg()["aaa_window_days"])
            w = aaa[(aaa.pitcher_id == str(r.mlbam_id)) & (aaa.date >= lo) & (aaa.date < f)]
        else:                                                     # amendment 7: AAA season Y-1
            w = aaa[(aaa.pitcher_id == str(r.mlbam_id)) & (aaa.season == r.season - 1)]
        w = w.assign(league="AAA", pa_id="AAA-" + w.pa_id.astype(str))
        w = w[w.pa_id.map(_audit(w).keep)]
        fl = _flags(w)
        pt = pa_table(fl)
        rows.append({"mlbam_id": r.mlbam_id, "season": r.season, "name": r.english_name, "n2": int(fl.n2.sum()),
                     "pa2": int(pt.pa2.sum()), "putaway": fl.so2.sum() / max(fl.n2.sum(), 1),
                     "foul_rate": fl.foul2.sum() / max(fl.n2.sum(), 1),
                     "aaa_kbb": (pt.event.eq("SO").sum() - pt.event.eq("uBB").sum()) / max(len(pt), 1),
                     "cpbl_kbb_c": r.cpbl_kbb_c})
    t = pd.DataFrame(rows)
    t["eligible"] = (t.n2 >= c["min_two_strike_pitches"]) & (t.pa2 >= c["min_two_strike_pa"])
    e = t[t.eligible]
    res = {"n_eligible": int(len(e)), "n_cohort": int(len(t))}
    if len(e) >= 5:
        b = cfg()["bootstrap"]
        rng = np.random.default_rng(b["seed"] + 1)
        est = _partial(e)
        draws = [_partial(e.iloc[rng.integers(0, len(e), len(e))]) for _ in range(b["draws"])]
        q = (1 - b["ci"]) / 2
        loo = [_partial(e.drop(i)) for i in e.index]
        res.update({"partial_rho": est, "ci": [float(np.nanquantile(draws, q)), float(np.nanquantile(draws, 1 - q))],
                    "leave_one_out": [float(np.nanmin(loo)), float(np.nanmax(loo))],
                    "foul_rate_rho": backtest.spearman(e.foul_rate, e.cpbl_kbb_c)})
        sig = est >= c["min_rho"] and res["ci"][0] > 0 and min(np.sign(loo)) > 0
        res["verdict"] = ("descriptive only (n below minimum)" if len(e) < c["min_n"]
                          else "tentative signal" if sig else "no signal")
    t.to_csv(processed("battle_part_b.csv"), index=False)
    return res


def _partial(e: pd.DataFrame) -> float:
    """Spearman of put-away and CPBL K-BB%, both ranks residualised on the rank of AAA K-BB%."""
    rk = e[["putaway", "cpbl_kbb_c", "aaa_kbb"]].rank()
    if rk.aaa_kbb.std() == 0:
        return np.nan
    res = {c: rk[c] - np.polyval(np.polyfit(rk.aaa_kbb, rk[c], 1), rk.aaa_kbb) for c in ("putaway", "cpbl_kbb_c")}
    return float(np.corrcoef(res["putaway"], res["cpbl_kbb_c"])[0, 1])


def reliability() -> dict:
    """2024 -> 2025 AAA year-to-year Spearman of put-away and continuation-foul rate (>= 100 two-strike pitches)."""
    a = milb_loader.load()
    a = a[a.season.isin([2024, 2025])].assign(league="AAA")
    a = a.assign(pa_id="AAA-" + a.pa_id.astype(str))
    a = a[a.pa_id.map(_audit(a).keep)]
    fl = _flags(a)
    g = fl.groupby(["pitcher_id", "season"])[["n2", "so2", "foul2"]].sum()
    g = g[g.n2 >= cfg()["part_b"]["min_two_strike_pitches"]]
    g["putaway"], g["foul"] = g.so2 / g.n2, g.foul2 / g.n2
    w = g.unstack("season").dropna()
    return {"n": int(len(w)), "putaway": backtest.spearman(w[("putaway", 2024)], w[("putaway", 2025)]),
            "foul_rate": backtest.spearman(w[("foul", 2024)], w[("foul", 2025)])}


# ------------------------------------------------------------------ run / report
def run() -> dict:
    d, audit = load()
    d = _flags(d)
    p = pa_table(d)
    aud = audit.groupby(["league", "season"]).agg(pas=("keep", "size"), kept=("keep", "sum"),
                                                  incomplete=("complete", lambda s: int((~s).sum())),
                                                  special=("special", "sum"),
                                                  inconsistent=("consistent", lambda s: int((~s).sum()))).reset_index()
    aud["season"] = aud.season.astype(int)
    out = {"audit": aud.to_dict("records"), "box_check": audit.attrs["box"].to_dict("records"), "part_a": part_a(d, p), "within_pitcher": within_pitcher(d),
           "part_b": part_b(), "reliability": reliability()}
    processed("battle_summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float))
    report(out)
    return out


def report(out: dict) -> None:
    A = out["part_a"]
    ss = A["seasons"]
    rows = []
    for r in A["rows"]:
        row = {"metric": r["metric"]}
        for s in ss:
            row[f"CPBL {s}"], row[f"AAA {s}"] = r[f"cpbl_{s}"], r[f"aaa_{s}"]
            row[f"diff {s} (95% CI)"] = f"{r[f'diff_{s}']:+.4f} ({r[f'ci_{s}'][0]:+.4f} to {r[f'ci_{s}'][1]:+.4f})"
        row["mean diff (95% CI)"] = f"{r['diff_mean']:+.4f} ({r['ci_mean'][0]:+.4f} to {r['ci_mean'][1]:+.4f})"
        rows.append(row)
    raw = [{"metric": r["metric"], **{f"CPBL raw {s}": r.get(f"cpbl_raw_{s}") for s in ss},
            **{f"AAA raw {s}": r.get(f"aaa_raw_{s}") for s in ss}, **{f"excluded {s}": r.get(f"excluded_weight_{s}") for s in ss}}
           for r in A["rows"] if f"cpbl_raw_{ss[0]}" in r]
    B = out["part_b"]
    L = ["# Two-strike battle metrics (amendment 14; generated by `python -m src.battle`)", "",
         "CPBL = REBAS 2024-25 (資料來源：野球革命 Rebas Open Data); AAA = statsapi / Statcast 2024-25. Complete, audited "
         "PAs only; bunt, intentional-walk, pitchout and interference PAs dropped. Two strikes = count before the pitch. "
         "Rates A1-A5 and A7 are standardised over balls x batter hand x pitcher hand x SP/RP to the average of the two "
         "leagues' distributions; A6 and A8 are raw. 95% CI from 2,000 game-cluster bootstrap draws.", "",
         f"**A1 verdict (registered): {A['a1_verdict']}.**", "",
         "## Sequence audit (PAs)", "", md_table(pd.DataFrame(out["audit"])), "",
         "Pitch-sequence totals vs season box totals (AAA: statsapi season stats; CPBL: no league box totals in "
         "hand, REBAS matched Baseball-Reference per pitcher in the amendment-12 check; CPBL has 360 regular-season "
         "games a year):", "", md_table(pd.DataFrame(out.get("box_check", []))), "",
         "## League comparison", "", md_table(pd.DataFrame(rows), ".4f"), "",
         "Raw rates of the standardised metrics, and the weight outside the common cells:", "",
         md_table(pd.DataFrame(raw), ".4f"), "",
         "## Within pitcher: two-strike minus one-strike (A1 per pitch, whiff per swing; descriptive)", "",
         md_table(pd.DataFrame(out["within_pitcher"]).T.reset_index(names="league_season"), ".4f"), "",
         "## Part B (exploratory): AAA two-strike put-away rate and CPBL K-BB%", "",
         f"Cohort {B['n_cohort']} (amendment 11), eligible {B['n_eligible']} (>= 100 two-strike pitches and >= 50 PAs "
         f"reaching two strikes in the pre-arrival AAA window). "
         + (f"Partial Spearman controlling AAA K-BB%: {B['partial_rho']:+.2f} (95% CI {B['ci'][0]:+.2f} to {B['ci'][1]:+.2f}; "
            f"leave-one-out {B['leave_one_out'][0]:+.2f} to {B['leave_one_out'][1]:+.2f}); verdict: {B['verdict']}. "
            f"Secondary, continuation-foul rate vs CPBL K-BB%: {B['foul_rate_rho']:+.2f}." if "partial_rho" in B else "Too few."),
         "", f"AAA 2024 -> 2025 reliability (pitchers with >= 100 two-strike pitches both years, n {out['reliability']['n']}): "
         f"put-away {out['reliability']['putaway']:.2f}, continuation-foul rate {out['reliability']['foul_rate']:.2f}.", ""]
    (path("reports") / "battle.md").write_text("\n".join(L), encoding="utf-8")


# ------------------------------------------------------------------ part C
def appendix(mlbam_id: int | None = None) -> pd.DataFrame:
    from . import stuff_plus
    m = cfg()["appendix_min"]
    if mlbam_id is None:
        sh = pd.read_csv(processed("recommendation_shortlist.csv"))
        r = sh[(sh.role == "pitcher") & (sh.rule_rank_signable == 1)].iloc[0]
        mlbam_id, name = int(r.mlbam_id), r["name"]
    else:
        name = str(mlbam_id)
    a = milb_loader.load()
    out, two = [], []
    for yr in (2025, 2026):
        s = a[a.season == yr].copy()
        s["rv"] = stuff_plus.run_values(s)                 # count values of that AAA season, full sequences
        p = s[s.pitcher_id == str(mlbam_id)].copy()
        if p.empty:
            continue
        p["ptype"] = p.pitch_type_raw.fillna("unknown")
        p["sw"] = p.swing.astype(bool) & ~p.is_bunt.astype(bool)
        for (pt, st), g in p.groupby(["ptype", "stand"]):
            n, sw, games, pas = len(g), int(g.sw.sum()), g.game_id.nunique(), int(g.pa_end.sum())
            row = {"season": yr, "pitch": pt, "batter": st, "pitches": n,
                   "usage": n / (p.stand == st).sum(), "swings": sw, "games": games, "terminal PAs": pas,
                   "velo median": g.velo_mph.median(), "velo IQR": g.velo_mph.quantile(.75) - g.velo_mph.quantile(.25)}
            w, rv = m["whiff"], m["run_value"]
            row["whiff/swing"] = (g.whiff.sum() / sw) if n >= w["pitches"] and sw >= w["swings"] and games >= w["games"] else "insufficient sample"
            row["RV/100 (+ = worse)"] = (100 * g.rv.mean()) if n >= rv["pitches"] and pas >= rv["pas"] and games >= rv["games"] else "insufficient sample"
            row["flag (< 50 pitches)"] = n < m["usage_flag"]
            out.append(row)
        t2 = p[p.strikes.eq(2)]
        mt = m["two_strike"]
        for (pt, st), g in t2.groupby(["ptype", "stand"]):
            ok = len(g) >= mt["pitches"] and g.game_id.nunique() >= mt["games"]
            two.append({"season": yr, "pitch": pt, "batter": st, "two-strike pitches": len(g),
                        "usage at two strikes": len(g) / (t2.stand == st).sum(),
                        "put-away": (g.pa_end.astype(bool) & g.pa_event.eq("SO")).mean() if ok else "insufficient sample",
                        "continuation foul": (g.pitch_code.astype(str).eq("F") & ~g.pa_end.astype(bool)).mean() if ok else "insufficient sample",
                        "whiff/swing": (g.whiff.sum() / g.sw.sum()) if ok and g.sw.sum() >= mt["whiff_swings"] else "insufficient sample"})
    # LHB share faced by RHP / LHP, both leagues 2024-25 (PA-ending pitches)
    c = rebas_loader.load()
    lhb = []
    for lg, d in (("CPBL", c[c.season.isin([2024, 2025])]), ("AAA", a[a.season.isin([2024, 2025])])):
        e = d[d.pa_end.astype(bool)]
        for h in ("R", "L"):
            x = e[e.p_throws == h]
            lhb.append({"league": lg, "pitcher hand": h, "LHB share": float((x.stand == "L").mean()), "PAs": int(len(x))})
    T, T2 = pd.DataFrame(out), pd.DataFrame(two)
    L = [f"# Appendix: {name}'s AAA pitch mix (amendment 14 part C; generated by `python -m src.battle appendix`)", "",
         "**Untested hypothesis, not a recommendation.** Observed AAA usage and results reflect his pitch selection and "
         "execution against AAA hitters; they do not estimate what would happen if he changed his mix in the CPBL. "
         "Run value is the project's count-based proxy (amendment 8), positive = worse for the pitcher, not Statcast RV. "
         "Cells below the registered minimums show counts only.", "",
         "## By pitch type and batter side", "", md_table(T, ".3f"), "",
         "## At two strikes", "", md_table(T2, ".3f") if len(T2) else "none", "",
         "## Batter-hand mix by pitcher hand, 2024-25", "", md_table(pd.DataFrame(lhb), ".3f"), "",
         "CPBL environment (findings 1.2-1.4): pitch-type differences become small and unstable across seasons once "
         "pitcher quality is controlled; CPBL lineups have more left-handed batters but similar platoon gaps; fastball "
         "velocity bands are confounded with pitcher quality.", ""]
    (path("reports") / "pick_appendix.md").write_text("\n".join(L), encoding="utf-8")
    T.to_csv(processed("pick_appendix.csv"), index=False)
    return T


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "appendix":
        print(appendix().to_string())
    else:
        print(json.dumps(run(), indent=1, ensure_ascii=False, default=float)[:4000])
