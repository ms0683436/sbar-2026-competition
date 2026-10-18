"""Amendment 20 — how much a strikeout and a walk are worth in the CPBL, and whether weighting
Marcel K% and BB% by that (after translation) predicts first-season CPBL runs allowed better than
the current 1:1 K-BB% (rules in config/scoring.yaml, amendment20). Nothing here changes the score:
a pass is a recommendation for Johnny.

Usage:  python -m src.kbb_weights      (after cpbl_census and marcel)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from . import cpbl_census
from .common import config, md_table, path, processed

EVENTS = ["1B", "2B", "3B", "HR", "BB", "HBP"]          # reference: outs (strikeouts included)
COLS = ["PA", "AB", "R", "H", "2B", "3B", "HR", "BB", "IBB", "HBP", "SO", "SH", "SF", "ER", "IP", "batters_faced"]


def cfg() -> dict:
    return config()["amendment20"]


def _n(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    for c in COLS:
        if c in d:
            d[c] = pd.to_numeric(d[c].replace("", np.nan), errors="coerce")
    return d


# ------------------------------------------------------------------ data
def team_lines() -> pd.DataFrame:
    """CPBL team-season batting totals (sum of the B-R player rows) with event shares per PA."""
    rows = []
    for y in range(2015, 2027):
        b = _n(cpbl_census.team_tables(y)["batting"])
        b = b[b.bbref_id.notna()]
        g = b.groupby("team_id")[["AB", "R", "H", "2B", "3B", "HR", "BB", "HBP", "SO", "SH", "SF"]].sum(min_count=1).fillna(0)
        g["season"] = y
        rows.append(g.reset_index())
    t = pd.concat(rows, ignore_index=True)
    t["1B"] = t.H - t["2B"] - t["3B"] - t.HR
    t["OUT"] = t.AB - t.H + t.SH + t.SF
    t["PA"] = t.AB + t.BB + t.HBP + t.SH + t.SF
    for e in EVENTS + ["OUT", "SO"]:
        t[f"p_{e}"] = t[e] / t.PA
    t["p_OUTNK"] = (t.OUT - t.SO) / t.PA
    t["rpa"] = t.R / t.PA
    return t


def pitcher_lines() -> pd.DataFrame:
    """CPBL pitcher-season lines (summed over teams) and league totals, 2015-2026."""
    rows = []
    for y in range(2015, 2027):
        p = _n(cpbl_census.team_tables(y)["pitching"])
        p = p[p.bbref_id.notna()].copy()
        p["ip"] = cpbl_census._ip(p.IP)
        g = p.groupby("bbref_id")[["R", "ER", "ip", "batters_faced", "SO", "BB", "HBP", "HR", "H"]].sum(min_count=1)
        g["ip_known"] = p.groupby("bbref_id").ip.apply(lambda s: s.notna().all())
        g["season"] = y
        lg_ip = p.ip.sum() if p.ip.notna().mean() > 0.95 else np.nan
        g["lg_ra9"] = p.R.sum() / lg_ip * 9
        g["lg_era"] = p.ER.sum() / lg_ip * 9
        g["lg_rbf"] = p.R.sum() / p.batters_faced.sum()
        rows.append(g.reset_index())
    return pd.concat(rows, ignore_index=True)


# ------------------------------------------------------------------ I. importance
def _values(b: dict, pbar: dict, out_key: str = "OUT") -> dict:
    """Run values relative to an average PA from coefficients relative to the reference event."""
    b = {**b, out_key: 0.0}
    avg = sum(pbar[e] * b[e] for e in pbar)
    v = {e: b[e] - avg for e in b}
    return {"w_K": -v[out_key] if "SO" not in b else -v["SO"], "w_BB": v["BB"], "values": v}


def importance(t: pd.DataFrame, fe: bool = False, k_separate: bool = False) -> dict:
    ev = EVENTS + (["SO"] if k_separate else [])
    ref = "OUTNK" if k_separate else "OUT"
    X = t[[f"p_{e}" for e in ev]].to_numpy()
    if fe:
        X = np.column_stack([X, pd.get_dummies(t.season, drop_first=True).to_numpy(float), np.ones(len(t))])
    else:
        X = np.column_stack([X, np.ones(len(t))])
    coef, *_ = np.linalg.lstsq(X, t.rpa.to_numpy(), rcond=None)
    b = dict(zip(ev, coef[:len(ev)]))
    w = t.PA.to_numpy()
    pbar = {e: float(np.average(t[f"p_{e}"], weights=w)) for e in ev + [ref]}
    r = _values(b, pbar, ref)
    return {"w_K": float(r["w_K"]), "w_BB": float(r["w_BB"]), "b": {k: float(v) for k, v in b.items()},
            "n_team_seasons": int(len(t))}


def importance_ci(t: pd.DataFrame, rng, **kw) -> dict:
    est = importance(t, **kw)
    seasons = t.season.unique()
    bs = []
    for _ in range(cfg()["importance"]["bootstrap"]["draws"]):
        pick = rng.choice(seasons, len(seasons))
        d = pd.concat([t[t.season == s] for s in pick], ignore_index=True)
        try:
            r = importance(d, **kw)
            bs.append((r["w_K"], r["w_BB"], r["w_K"] / r["w_BB"] if r["w_BB"] > 0 else np.nan))
        except np.linalg.LinAlgError:
            continue
    a = np.array(bs)
    q = (1 - cfg()["importance"]["bootstrap"]["ci"]) / 2
    ci = lambda x: [float(np.nanquantile(x, q)), float(np.nanquantile(x, 1 - q))]
    return est | {"w_K_ci": ci(a[:, 0]), "w_BB_ci": ci(a[:, 1]), "ratio": est["w_K"] / est["w_BB"],
                  "ratio_ci": ci(a[:, 2]), "draws": len(a)}


def baseruns(t: pd.DataFrame) -> dict:
    """BaseRuns at pooled CPBL totals, B multiplier calibrated to league runs; marginal values
    of one extra event relative to an average PA (K = any out)."""
    s = t[["R", "H", "2B", "3B", "HR", "BB", "HBP", "AB", "SH", "SF", "1B", "OUT", "PA"]].sum()

    def runs(c: dict, m: float) -> float:
        h = c["1B"] + c["2B"] + c["3B"] + c["HR"]
        tb = c["1B"] + 2 * c["2B"] + 3 * c["3B"] + 4 * c["HR"]
        A = h + c["BB"] + c["HBP"] - c["HR"]
        B = (1.4 * tb - 0.6 * h - 3 * c["HR"] + 0.1 * (c["BB"] + c["HBP"])) * m
        C = c["OUT"]
        return A * B / (B + C) + c["HR"]
    base = {e: float(s[e]) for e in EVENTS + ["OUT"]}
    m = brentq(lambda x: runs(base, x) - s.R, 0.3, 3.0)
    d = {e: runs({**base, e: base[e] + 1}, m) - runs(base, m) for e in base}
    pbar = {e: base[e] / s.PA for e in base}
    avg = sum(pbar[e] * d[e] for e in base)
    v = {e: d[e] - avg for e in base}
    return {"w_K": float(-v["OUT"]), "w_BB": float(v["BB"]), "ratio": float(-v["OUT"] / v["BB"]), "B_multiplier": float(m)}


def pitcher_regression(pl: pd.DataFrame) -> dict:
    lo, hi = cfg()["importance"]["seasons"]
    d = pl[pl.season.between(lo, hi) & (pl.batters_faced >= cfg()["importance"]["pitcher_min_bf"])].dropna(
        subset=["R", "batters_faced", "SO", "BB", "HBP", "HR", "H"])
    bf = d.batters_faced
    sh = {"SO": d.SO / bf, "BB": d.BB / bf, "HBP": d.HBP / bf, "HR": d.HR / bf, "HIT": (d.H - d.HR) / bf}
    X = np.column_stack([*sh.values(), np.ones(len(d))])
    coef, *_ = np.linalg.lstsq(X, (d.R / bf).to_numpy(), rcond=None)
    b = dict(zip(sh, coef[:5]))
    pbar = {k: float(np.average(v, weights=bf)) for k, v in sh.items()}
    pbar["OUT"] = 1 - sum(pbar.values())                 # in-play outs: the reference
    b["OUT"] = 0.0
    avg = sum(pbar[k] * b[k] for k in pbar)
    return {"w_K": float(-(b["SO"] - avg)), "w_BB": float(b["BB"] - avg), "ratio": float(-(b["SO"] - avg) / (b["BB"] - avg)),
            "n_pitcher_seasons": int(len(d))}


# ------------------------------------------------------------------ III. validation
def cohort(pl: pd.DataFrame) -> pd.DataFrame:
    cen = pd.read_csv(processed("marcel_v2_census.csv"))
    first = pl.rename(columns={"season": "season_p"})
    c = cen.merge(first, left_on=["bbref_id", "season"], right_on=["bbref_id", "season_p"], how="left")
    ok = c.ip_known & (c.ip > 0) & c.lg_ra9.notna()
    c["ra9_rel"] = np.where(ok, c.R / c.ip * 9 - c.lg_ra9, np.nan)
    c["era_rel"] = np.where(ok, c.ER / c.ip * 9 - c.lg_era, np.nan)
    c["rbf_rel"] = c.R / c.batters_faced - c.lg_rbf
    return c


def _fit(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    k, a = np.polyfit(x.astype(float), y.astype(float), 1)
    return float(a), float(k)


def rolling(c: pd.DataFrame, t: pd.DataFrame, outcome: str, arrivals: tuple[int, int]) -> pd.DataFrame:
    v = cfg()["validation"]
    d = c[c.season.between(*arrivals)].dropna(subset=[outcome, "marcel_k", "marcel_bb", "cpbl_k_c", "cpbl_bb_c"])
    rows, fallback = [], 0
    for Y in sorted(d.season.unique()):
        tr, te = d[d.season < Y], d[d.season == Y]
        if len(tr) < v["min_train"] or not len(te):
            continue
        aK, bK = _fit(tr.marcel_k, tr.cpbl_k_c)
        aB, bB = _fit(tr.marcel_bb, tr.cpbl_bb_c)
        tt = t[t.season.between(cfg()["importance"]["seasons"][0], min(Y - 1, cfg()["importance"]["seasons"][1]))]
        imp = importance(tt)
        wK, wB = imp["w_K"], imp["w_BB"]
        if wK <= 0 or wB <= 0:
            wK = wB = 1.0
            fallback += 1
        scores = {"A1": lambda f: f.marcel_k - f.marcel_bb,
                  "A2": lambda f: (aK + bK * f.marcel_k) - (aB + bB * f.marcel_bb),
                  "A3": lambda f: wK * (aK + bK * f.marcel_k) - wB * (aB + bB * f.marcel_bb)}
        pred = {"M0": pd.Series(tr[outcome].mean(), index=te.index)}
        for k, s in scores.items():
            c0, d0 = _fit(s(tr), tr[outcome])
            pred[k] = c0 + d0 * s(te)
        for i in te.index:
            rows.append({"name": te.name[i], "season": int(Y), "bf": float(te.batters_faced[i]), "actual": float(te[outcome][i]),
                         **{f"pred_{k}": float(p[i]) for k, p in pred.items()}, "w_K": wK, "w_BB": wB,
                         "b_K": bK, "b_BB": bB})
    f = pd.DataFrame(rows)
    f.attrs["fallback_years"] = fallback
    return f


def compare(f: pd.DataFrame, rng, weighted: bool = False) -> dict:
    b = cfg()["validation"]["bootstrap"]
    e = {k: (f.actual - f[f"pred_{k}"]).abs().to_numpy() for k in ("M0", "A1", "A2", "A3")}
    w = f.bf.to_numpy() if weighted else np.ones(len(f))
    mae = lambda k, i: float(np.average(e[k][i], weights=w[i]))
    idx = np.arange(len(f))
    draws = [rng.integers(0, len(f), len(f)) for _ in range(b["draws"])]
    q = (1 - b["ci"]) / 2
    out = {"n": int(len(f)), "mae": {k: mae(k, idx) for k in e}, "diff": {}}
    for a_, c_ in (("A3", "A1"), ("A2", "A1"), ("A3", "A2"), ("A1", "M0"), ("A2", "M0"), ("A3", "M0")):
        ds = [mae(a_, i) - mae(c_, i) for i in draws]
        out["diff"][f"{a_}-{c_}"] = {"est": mae(a_, idx) - mae(c_, idx),
                                     "ci": [float(np.quantile(ds, q)), float(np.quantile(ds, 1 - q))]}
    return out


def decision(r: dict) -> str:
    v = cfg()["validation"]
    d = r["diff"]["A3-A1"]
    if r["n"] < config()["amendment12"]["min_n_for_conclusion"]:
        return "樣本不足，維持現在的 K−BB%"
    if d["ci"][1] < 0 and d["est"] <= -v["min_gain_ra9"]:
        imp = r["diff"]["A3-A2"]["ci"][1] < 0
        return "建議換成 A3（待 Johnny 決定）" + ("；重要性本身也有貢獻" if imp else "；但改善主要來自換算，不是重要性")
    return "未達門檻，維持現在的 K−BB%"


def run() -> dict:
    rng = np.random.default_rng(cfg()["importance"]["bootstrap"]["seed"])
    t = team_lines()
    lo, hi = cfg()["importance"]["seasons"]
    ti = t[t.season.between(lo, hi)]
    pl = pitcher_lines()
    out = {"importance": importance_ci(ti, rng),
           "sensitivity": {"season_fixed_effects": importance_ci(ti, rng, fe=True),
                           "k_separate": importance_ci(ti, rng, k_separate=True),
                           "baseruns": baseruns(ti), "pitcher_seasons": pitcher_regression(pl),
                           "seasons_2023_on": importance(t[t.season.between(2023, hi)])}}
    c = cohort(pl)
    out["translation_all"] = {"K": _fit(c.marcel_k, c.cpbl_k_c), "BB": _fit(c.marcel_bb, c.cpbl_bb_c)}
    rv = np.random.default_rng(cfg()["validation"]["bootstrap"]["seed"])
    arr = tuple(cfg()["validation"]["arrivals"])
    f = rolling(c, t, "ra9_rel", arr)
    out["validation"] = compare(f, rv) | {"fallback_years": f.attrs["fallback_years"],
                                          "held_out_by_year": f.season.value_counts().sort_index().to_dict()}
    out["decision"] = decision(out["validation"])
    sens = {}
    for lab, oc, a in (("R/BF（含 2026）", "rbf_rel", (arr[0], 2026)), ("ERA", "era_rel", arr)):
        g = rolling(c, t, oc, a)
        sens[lab] = compare(g, rv)
    sens["RA9，依打席數加權"] = compare(f, rv, weighted=True)
    late = f[f.season >= 2023]
    sens["RA9，只看 2023 年以後來台（描述）"] = compare(late, rv) if len(late) >= 3 else {"n": len(late)}
    out["validation_sensitivity"] = sens
    f.to_csv(processed("kbb_weights_validation.csv"), index=False)
    t.to_csv(processed("cpbl_team_seasons.csv"), index=False)
    processed("kbb_weights_summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float))
    report(out, f)
    return out


def _diff_rows(r: dict) -> pd.DataFrame:
    lab = {"A3-A1": "A3 − A1（主要）", "A2-A1": "A2 − A1", "A3-A2": "A3 − A2", "A1-M0": "A1 − 猜平均",
           "A2-M0": "A2 − 猜平均", "A3-M0": "A3 − 猜平均"}
    return pd.DataFrame([{"比較": lab[k], "誤差差": v["est"], "90% 下界": v["ci"][0], "90% 上界": v["ci"][1]}
                         for k, v in r["diff"].items()])


def report(out: dict, f: pd.DataFrame) -> None:
    i, s = out["importance"], out["sensitivity"]
    v = out["validation"]
    tr = out["translation_all"]
    imp_rows = [{"做法": "主要：中職球隊得分迴歸（2015–2025）", "三振價值": i["w_K"], "保送價值": i["w_BB"], "K:BB": i["ratio"],
                 "說明": f"{i['n_team_seasons']} 個球隊球季；90% CI 三振 {i['w_K_ci'][0]:.2f}–{i['w_K_ci'][1]:.2f}、保送 "
                         f"{i['w_BB_ci'][0]:.2f}–{i['w_BB_ci'][1]:.2f}、比值 {i['ratio_ci'][0]:.2f}–{i['ratio_ci'][1]:.2f}"}]
    for lab, key in (("加入球季固定效果", "season_fixed_effects"), ("三振和其他出局分開", "k_separate")):
        x = s[key]
        imp_rows.append({"做法": lab, "三振價值": x["w_K"], "保送價值": x["w_BB"], "K:BB": x["ratio"],
                         "說明": f"比值 90% CI {x['ratio_ci'][0]:.2f}–{x['ratio_ci'][1]:.2f}"})
    imp_rows += [{"做法": "BaseRuns 公式（中職總數校準）", "三振價值": s["baseruns"]["w_K"], "保送價值": s["baseruns"]["w_BB"],
                  "K:BB": s["baseruns"]["ratio"], "說明": "公式本身有外部假設，三振等同一般出局"},
                 {"做法": "中職投手個人球季迴歸", "三振價值": s["pitcher_seasons"]["w_K"], "保送價值": s["pitcher_seasons"]["w_BB"],
                  "K:BB": s["pitcher_seasons"]["ratio"], "說明": f"{s['pitcher_seasons']['n_pitcher_seasons']} 個投手球季（≥ 100 打席），條件相關，只當佐證"},
                 {"做法": "只用 2023–2025（描述）", "三振價值": s["seasons_2023_on"]["w_K"], "保送價值": s["seasons_2023_on"]["w_BB"],
                  "K:BB": s["seasons_2023_on"]["w_K"] / s["seasons_2023_on"]["w_BB"],
                  "說明": f"{s['seasons_2023_on']['n_team_seasons']} 個球隊球季，參數多、樣本少"}]
    eff = i["w_K"] * tr["K"][1] / (i["w_BB"] * tr["BB"][1])
    L = ["# 中職三振、保送的得分價值與權重驗證（修正 20；由 `python -m src.kbb_weights` 產生）", "",
         "Johnny：「那各占多少？該怎麼決定？」規則在計算前登錄。**重要性** = 在中職，一個三振、一個保送，和一個平均打席相比"
         "各少給、多給幾分（只用中職資料）。**換算** = 猴子方法的 K%、BB% 優勢到中職還剩多少（普查）。**驗證** = 用來台第一季的"
         "失分率（RA9，減去當年聯盟平均）比較三種做法，失分本身不含任何 K、BB 權重。這份報告不改變排名，結果由 Johnny 決定。", "",
         "## 一、重要性：中職一個三振、一個保送值幾分", "", md_table(pd.DataFrame(imp_rows), ".3f"), "",
         "價值的單位是「分／次」，和一個平均打席比較。K:BB 大於 1 代表三振比較值錢。", "",
         "## 二、換算：猴子方法到中職", "",
         f"全部 53 人：中職 K% = {tr['K'][0]:+.4f} + {tr['K'][1]:.2f} × 猴子 K%；中職 BB% = {tr['BB'][0]:+.4f} + "
         f"{tr['BB'][1]:.2f} × 猴子 BB%（驗證時每一年只用更早來台的人重估）。", "",
         f"重要性 × 換算之後，猴子方法 K% 對 BB% 的實際比重約 **{eff:.2f} : 1**（現在的 K−BB% 是 1 : 1）。", "",
         "## 三、驗證：哪一種做法比較能預測來台第一季的失分", "",
         "- A1 現在：猴子 K% − 猴子 BB%（1 : 1）",
         "- A2 換算後 1 : 1：預測中職 K% − 預測中職 BB%",
         "- A3 換算後乘上中職重要性：三振價值 × 預測中職 K% − 保送價值 × 預測中職 BB%",
         "- 每個做法都用更早來台的人把分數換成失分預測；誤差 = 預測和實際 RA9 差幾分（平均絕對誤差）。", "",
         f"驗證人數 {v['n']}（各年 {v['held_out_by_year']}）；重要性估不出正值而改用 1 : 1 的年數：{v['fallback_years']}。", "",
         md_table(pd.DataFrame([{"做法": k, "平均誤差（RA9）": x} for k, x in v["mae"].items()]).replace(
             {"做法": {"M0": "猜平均", "A1": "A1 現在", "A2": "A2 換算 1:1", "A3": "A3 換算 + 重要性"}}), ".3f"), "",
         "成對比較（負數 = 前者誤差較小；bootstrap 2,000 次）：", "", md_table(_diff_rows(v), "+.3f"), "",
         f"**判定：{out['decision']}**（門檻：A3 − A1 的 90% 上界 < 0，且至少少 {cfg()['validation']['min_gain_ra9']:.2f} 分）。", "",
         "### 敏感度（只報告，不拿來挑結果）", ""]
    for lab, r in out["validation_sensitivity"].items():
        if "diff" in r:
            d = r["diff"]
            m = 100 if lab.startswith("R/BF") else 1          # R/BF shown per 100 batters faced
            L.append(f"- {lab}（n {r['n']}）：A3 − A1 {m * d['A3-A1']['est']:+.3f}（{m * d['A3-A1']['ci'][0]:+.3f} 至 "
                     f"{m * d['A3-A1']['ci'][1]:+.3f}）；A2 − A1 {m * d['A2-A1']['est']:+.3f}；A1 − 猜平均 "
                     f"{m * d['A1-M0']['est']:+.3f}（{m * d['A1-M0']['ci'][0]:+.3f} 至 {m * d['A1-M0']['ci'][1]:+.3f}）。")
        else:
            L.append(f"- {lab}：n {r['n']}，太少。")
    L += ["", "R/BF 的單位是每 100 打席失分；ERA 只算自責分。", "",
          "## 限制", "",
          "- 53 人普查在設計這個方法前就看過，這不是全新的驗證。",
          "- 球隊得分還受守備、連打時機、球場影響；中職每年只有 4–6 隊，迴歸參數多、樣本少。",
          "- 失分率雜訊很大，一季只有幾十局；所以比不出差異不代表沒有差異。",
          "- 2026 年 Baseball-Reference 沒有局數，RA9 只到 2025 年來台的人；R/BF 敏感度有包含 2026。", "",
          "資料來源：Baseball-Reference（中職球隊與投手成績）；AAA 來自 MLB Stats API。", ""]
    (path("reports") / "kbb_weights.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    print(json.dumps(run(), indent=1, ensure_ascii=False, default=float)[:3000])
