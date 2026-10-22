"""Amendment 21 — does Stuff+ add to Marcel K-BB% for AAA -> KBO and AAA -> CPBL (2024-26 arrivals)?
Rules in config/scoring.yaml, amendment21. A report only: nothing here changes the score, the
ranking or the picks.

  S0  z(Marcel K-BB%)
  S1  (0.5 z(Marcel K-BB%) + 0.2 z(Stuff+)) / 0.7      (the main ranking without Fit)
  each z against amendment 18's AAA D-1 reference pool R(D); Stuff+ = rule S, missing = 100

Usage:  python -m src.stuff_signal      (after kbo and marcel)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import a18_checks, backtest, backtest_chrono, marcel, stuff_rank
from .common import config, md_table, path, processed

LEAGUES = {"kbo": "AAA → 韓職", "cpbl": "AAA → 中職"}


def cfg() -> dict:
    return config()["amendment21"]


def _verdict(n: int, lo: float, hi: float) -> str:
    if n < backtest_chrono.cfg()["min_n_for_conclusion"]:
        return "僅描述（n 不足）"
    return "Stuff+ 有加分" if lo > 0 else "Stuff+ 扣分" if hi < 0 else "無定論"


def samples() -> pd.DataFrame:
    """One row per pitcher and league (first transition in the window): mlbam_id, name, league,
    D, Marcel K-BB% (centred), outcome K-BB% and K% (centred)."""
    k0, k1 = cfg()["kbo_seasons"]
    k = pd.read_csv(processed("kbo_t2_pitchers.csv"))
    k = k[k.dest.between(k0, k1)].sort_values("dest").drop_duplicates("player_id")
    k = pd.DataFrame({"mlbam_id": k.player_id.astype(int), "name": k.name, "league": "kbo", "D": k.dest.astype(int),
                      "marcel": k.aaa_m, "y_kbb": k.kbb_c, "y_k": k.k_c})
    c0, c1 = cfg()["cpbl_arrivals"]
    c = pd.read_csv(processed("marcel_v2_census.csv"))
    c = c[c.season.between(c0, c1)].sort_values("season").drop_duplicates("mlbam_id")
    c = pd.DataFrame({"mlbam_id": c.mlbam_id.astype(int), "name": c.name, "league": "cpbl", "D": c.season.astype(int),
                      "marcel": c.marcel, "y_kbb": c.cpbl_kbb_c, "y_k": c.cpbl_k_c})
    return pd.concat([k, c], ignore_index=True)


def scored(s: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """z(Marcel) and z(Stuff+) against R(D); Marcel of the pool from the same projection as the
    league's sample (KBO park-adjusted, CPBL PF = 1)."""
    clip = config()["amendment13"]["stuff_clip"]
    g = stuff_rank.game_sums()
    parts, info = [], {}
    for (lg, D), d in s.groupby(["league", "D"]):
        D = int(D)
        seasons = [D - 1, D - 2, D - 3]
        pool = a18_checks._pool(D)
        m = marcel.project("pitcher", D, seasons=seasons, neutral=(lg == "cpbl")).set_index("player_id").score_c
        st = stuff_rank.candidate_stuff(g, seasons=seasons).set_index("pitcher_id").stuff_plus
        pm, ps = pool.mlbam_id.map(m), pool.mlbam_id.map(st).fillna(100.0)
        d = d.copy()
        d["stuff"] = d.mlbam_id.map(st)
        d["stuff_missing"] = d.stuff.isna()
        d["stuff"] = d.stuff.fillna(100.0)
        d["z_marcel"] = backtest_chrono._z(d.marcel, pm.dropna())
        d["z_stuff"] = backtest_chrono._z(d.stuff, ps, clip)
        info[f"{lg} {D}"] = {"n": int(len(d)), "pool": int(len(pool)), "pool_with_marcel": int(pm.notna().sum()),
                             "stuff_missing": int(d.stuff_missing.sum()),
                             "stuff_seasons_with_data": sorted(int(x) for x in g.season.unique() if x in seasons)}
        parts.append(d)
    out = pd.concat(parts, ignore_index=True)
    w = cfg()["weights"]
    tot = w["marcel"] + w["stuff"]
    out["s0"] = out.z_marcel
    out["s1"] = (w["marcel"] * out.z_marcel + w["stuff"] * out.z_stuff) / tot
    for ws in cfg()["sensitivity_stuff_weights"]:
        out[f"s1_w{ws}"] = ((tot - ws) * out.z_marcel + ws * out.z_stuff) / tot
    return out, info


def _boot(d: pd.DataFrame, stat, k: int) -> tuple[float, float, float]:
    """Bootstrap over pitchers (clusters of mlbam_id, so a pitcher in both leagues moves as one)."""
    b = cfg()["bootstrap"]
    rng = np.random.default_rng(b["seed"] + k)
    groups = [g.index.to_numpy() for _, g in d.groupby("mlbam_id")]
    vals = []
    for _ in range(b["draws"]):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        v = stat(d.loc[idx])
        if not np.isnan(v):
            vals.append(v)
    q = (1 - b["ci"]) / 2
    return float(stat(d)), float(np.quantile(vals, q)), float(np.quantile(vals, 1 - q))


def _rho(d, x, y):
    return backtest.spearman(d[x], d[y])


def tests(d: pd.DataFrame, y: str, k: int) -> dict:
    d = d.dropna(subset=["s0", "s1", y]).reset_index(drop=True)
    n = int(d.mlbam_id.nunique())
    res = {"n_rows": int(len(d)), "n_pitchers": n}
    for j, (key, x) in enumerate([("marcel", "s0"), ("marcel_stuff", "s1"), ("stuff", "z_stuff")]):
        est, lo, hi = _boot(d, lambda e, x=x: _rho(e, x, y), k + j)
        res[key] = {"rho": est, "ci_lo": lo, "ci_hi": hi}
    est, lo, hi = _boot(d, lambda e: _rho(e, "s1", y) - _rho(e, "s0", y), k + 10)
    res["paired"] = {"diff": est, "ci_lo": lo, "ci_hi": hi, "verdict": _verdict(n, lo, hi)}
    res["sensitivity"] = {}
    for j, ws in enumerate(cfg()["sensitivity_stuff_weights"]):
        est, lo, hi = _boot(d, lambda e, c=f"s1_w{ws}": _rho(e, c, y) - _rho(e, "s0", y), k + 20 + j)
        res["sensitivity"][str(ws)] = {"diff": est, "ci_lo": lo, "ci_hi": hi}
    return res


def run() -> dict:
    d, info = scored(samples())
    out = {"pools": info, "primary": {}, "secondary": {}}
    for i, lg in enumerate(LEAGUES):
        out["primary"][lg] = tests(d[d.league == lg], "y_kbb", 100 * i)
        out["secondary"][f"{lg} K%"] = tests(d[d.league == lg], "y_k", 100 * i + 50)
    out["secondary"]["pooled"] = tests(d, "y_kbb", 300)
    out["secondary"]["pooled K%"] = tests(d, "y_k", 350)
    d.to_csv(processed("stuff_signal_cohort.csv"), index=False)
    with open(processed("stuff_signal_summary.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    report(d, out)
    return out


def _f(r: dict, k: str = "rho") -> str:
    return f"{r[k]:+.2f}（{r['ci_lo']:+.2f} 至 {r['ci_hi']:+.2f}）"


def report(d: pd.DataFrame, out: dict) -> None:
    rows = []
    for lab, r in [(LEAGUES["kbo"] + " K−BB%（主要）", out["primary"]["kbo"]),
                   (LEAGUES["cpbl"] + " K−BB%（主要）", out["primary"]["cpbl"]),
                   ("兩聯盟合併 K−BB%", out["secondary"]["pooled"]),
                   (LEAGUES["kbo"] + " K%", out["secondary"]["kbo K%"]),
                   (LEAGUES["cpbl"] + " K%", out["secondary"]["cpbl K%"]),
                   ("兩聯盟合併 K%", out["secondary"]["pooled K%"])]:
        rows.append({"樣本": lab, "n": r["n_pitchers"], "只看 K−BB%": _f(r["marcel"]),
                     "K−BB% + Stuff+": _f(r["marcel_stuff"]), "只看 Stuff+": _f(r["stuff"]),
                     "差（加 Stuff+ 減只看 K−BB%）": _f(r["paired"], "diff"), "判定": r["paired"]["verdict"]})
    sens = [{"樣本": lab, **{f"Stuff+ 權重 {w}": _f(r["sensitivity"][w], "diff") for w in r["sensitivity"]},
             "Stuff+ 權重 0.2（主要）": _f(r["paired"], "diff")}
            for lab, r in [(LEAGUES["kbo"], out["primary"]["kbo"]), (LEAGUES["cpbl"], out["primary"]["cpbl"]),
                           ("合併", out["secondary"]["pooled"])]]
    pools = pd.DataFrame([{"樣本": k, **v, "stuff_seasons_with_data": ", ".join(map(str, v["stuff_seasons_with_data"]))}
                          for k, v in out["pools"].items()])
    roster = d.sort_values(["league", "D", "s1"], ascending=[True, True, False])[
        ["league", "D", "name", "marcel", "stuff", "stuff_missing", "z_marcel", "z_stuff", "s1", "y_kbb"]].copy()
    roster["marcel"] *= 100
    roster["y_kbb"] *= 100
    roster.columns = ["聯盟", "季", "投手", "Marcel K−BB%（置中，%）", "Stuff+", "無追蹤", "z Marcel", "z Stuff+", "S1",
                      "實際 K−BB%（置中，%）"]

    def verdict_line(lg):
        p = out["primary"][lg]["paired"]
        return (f"- **{LEAGUES[lg]}**（{out['primary'][lg]['n_pitchers']} 人）：加入 Stuff+ 後的等級相關"
                f"差 {p['diff']:+.2f}，90% 信賴區間 {p['ci_lo']:+.2f} 至 {p['ci_hi']:+.2f}，判定「{p['verdict']}」。")

    md = f"""# 修正 21：K−BB% 加 Stuff+，訊號有比只看 K−BB% 強嗎？

規則在 config/scoring.yaml 的 amendment21，於計算任何數字前提交（4b13422）。這只是報告：
不改分數、排序或推薦名單。

## 結論

{verdict_line("kbo")}
{verdict_line("cpbl")}
- 兩聯盟合併（{out['secondary']['pooled']['n_pitchers']} 人，依投手聚類）：差 {out['secondary']['pooled']['paired']['diff']:+.2f}，90% 信賴區間 {out['secondary']['pooled']['paired']['ci_lo']:+.2f} 至 {out['secondary']['pooled']['paired']['ci_hi']:+.2f}，判定「{out['secondary']['pooled']['paired']['verdict']}」。

判定規則：差的 90% 信賴區間整段大於 0 才算「Stuff+ 有加分」，整段小於 0 算「扣分」，其餘為「無定論」；
n 少於 15 只做描述。

## 主要與次要檢定

數字是 Spearman 等級相關（括號內為 90% 信賴區間，2,000 次依投手重抽，種子 20261022）。

{md_table(pd.DataFrame(rows))}

## 敏感度（只報告，不用來選）

Stuff+ 權重改為 0.1 或 0.3（從 Marcel 的 0.5 挪過去，總和維持 0.7），與只看 K−BB% 的差：

{md_table(pd.DataFrame(sens))}

## 樣本與參考池

- 韓職：修正 16 的 T2 世代中，韓職季 D 在 2024–2026、AAA D−1 與韓職 D 各 ≥ 150 打席的投手，每人取第一次。
  Marcel 依修正 16 做球場調整。結果是該季置中後的韓職 K−BB%（資料來源：FanGraphs）。
- 中職：修正 12 普查中 2024–2026 來台的投手，每人取第一季（2026 用 2026-10-22 重抓的
  Baseball-Reference 頁面，總打席 54,182 與先前相同）。Marcel 依修正 15 的 V2，球場係數 = 1。
- Stuff+ 用修正 18 的規則 S（AAA D−1、D−2、D−3 有追蹤資料的季，權重 3/2/1，向 100 收斂），沒有追蹤資料的投手給 100。
- 兩個 z 都以修正 18 的 AAA D−1 參考池 R(D)（≥ 100 打席、≥ 400 顆追蹤球）為基準；Stuff+ z 截在 ±2.5。

{md_table(pools)}

## 揭露

- Stuff+ 模型以 AAA 2025 訓練，所以 2024–2025 移籍的投手用了看過較晚 AAA 資料的模型。
- 修正 18 已看過中職 17 人的描述性結果（M2 0.35 對 Marcel 0.29，未做配對檢定），修正 16 已看過韓職 Marcel 的結果；
  本修正的樣本因此不是完全沒看過的資料。
- 中職 2026 數字來自 Baseball-Reference（請引用），2026 中職局數在 B-R 上仍為空白。

## 名單

{md_table(roster, floatfmt=".2f")}
"""
    (path("reports") / "stuff_signal.md").write_text(md)


if __name__ == "__main__":
    r = run()
    for lg in LEAGUES:
        print(lg, r["primary"][lg]["n_pitchers"], r["primary"][lg]["marcel"]["rho"], r["primary"][lg]["marcel_stuff"]["rho"], r["primary"][lg]["paired"])
    print("pooled", r["secondary"]["pooled"]["paired"])
