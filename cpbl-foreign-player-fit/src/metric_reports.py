"""Amendment 19 — K%, BB% and K-BB% reported one by one (rules in config/scoring.yaml,
amendment19). A report only: the three metrics are never combined or weighted, and nothing here
changes the score, the ranking or the picks.

Usage:  python -m src.metric_reports      (after evaluate, cpbl_census, marcel and kbo)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import backtest
from .common import ROOT, config, md_table, path, processed

# metric -> (label, Marcel column in the candidate table, census / a11 Marcel column, CPBL column,
#            census single-season test / slope label, V3 key, KBO T1 test prefix, T2 row)
METRICS = {
    "k": ("K%", "marcel_k_pct", "marcel_k", "cpbl_k_c", "AAA K%", "k", "K1b", "AAA Marcel k vs KBO k"),
    "bb": ("BB%", "marcel_bb_pct", "marcel_bb", "cpbl_bb_c", "AAA BB%", "bb", "K1c", "AAA Marcel bb vs KBO bb"),
    "kbb": ("K−BB%", "marcel_score", "marcel", "cpbl_kbb_c", "AAA K-BB%", "k-bb", "K1 ", "AAA Marcel vs KBO next season"),
}


def cfg() -> dict:
    return config()["amendment19"]


def _verdict(n: int, lo: float, hi: float) -> str:
    if n < config()["amendment12"]["min_n_for_conclusion"]:
        return "樣本不足，只能描述"
    return "支持" if lo > 0 else "相反" if hi < 0 else "無法判定"


def _test(label: str, d: pd.DataFrame, x: str, y: str, rng) -> dict:
    b = cfg()["bootstrap"]
    d = d[[x, y]].astype(float).dropna()
    if len(d) < 3:
        return {"檢定": label, "n": len(d), "ρ": np.nan, "下界": np.nan, "上界": np.nan, "判定": "樣本太少"}
    est, lo, hi = backtest.boot_ci(d, x, y, b["draws"], b["ci"], rng)
    return {"檢定": label, "n": len(d), "ρ": est, "下界": lo, "上界": hi, "判定": _verdict(len(d), lo, hi)}


def _slope(label: str, d: pd.DataFrame, x: str, y: str, rng) -> dict:
    b = cfg()["bootstrap"]
    m = d[[x, y]].astype(float).dropna().to_numpy()
    k = np.polyfit(m[:, 0], m[:, 1], 1)[0]
    bs = [np.polyfit(m[i, 0], m[i, 1], 1)[0] for i in (rng.integers(0, len(m), len(m)) for _ in range(b["draws"]))
          if np.ptp(m[i, 0]) > 0]
    q = (1 - b["ci"]) / 2
    return {"斜率": label, "n": len(m), "估計": float(k), "下界": float(np.quantile(bs, q)), "上界": float(np.quantile(bs, 1 - q))}


def _existing(row: dict, label: str | None = None) -> dict:
    """An already-published test row in this report's column names."""
    verdict = {"supports": "支持", "contradicts": "相反", "inconclusive": "無法判定"}.get(row.get("verdict"), row.get("verdict"))
    if str(verdict).startswith("descriptive"):
        verdict = "樣本不足，只能描述"
    return {"檢定": label or row["test"], "n": row["n"], "ρ": row["rho"], "下界": row["ci_lo"], "上界": row["ci_hi"],
            "判定": verdict}


def candidates() -> pd.DataFrame:
    P = pd.read_csv(processed("pitcher_fit_results.csv"))
    return P[P.fit_available & P.signable.fillna(False).astype(bool)].copy()


def run() -> dict:
    rng = np.random.default_rng(cfg()["bootstrap"]["seed"])
    cen = pd.read_csv(processed("marcel_v2_census.csv"))
    a11 = pd.read_csv(processed("marcel_v2_a11.csv"))
    census = json.loads(processed("cpbl_census_summary.json").read_text())["pitchers"]
    v3 = json.loads(processed("marcel_summary.json").read_text())["v2_v3"]["v3"]
    kbo = json.loads(processed("kbo_summary.json").read_text())
    out = {"metrics": {}}
    for m in cfg()["metrics"]:
        lab, _, mc, yc, aaa, v3k, k1, t2 = METRICS[m]
        old = {"k": "CP2", "bb": "CP3", "kbb": "CP1"}[m]
        single = next(r for r in census["tests"] if r["test"].startswith(old))
        sl = next(r for r in census["slopes"] if r["metric"] == aaa)
        r = {"census": [_test(f"猴子方法 {lab} vs 中職 {lab}（新）", cen, mc, yc, rng),
                        _existing(single, f"單季 AAA {lab} vs 中職 {lab}（既有，{old}）")],
             "slopes": [_slope(f"中職 {lab} 對 猴子方法 {lab}（新）", cen, mc, yc, rng),
                        {"斜率": f"中職 {lab} 對 單季 AAA {lab}（既有）", "n": sl["n"], "估計": sl["slope"],
                         "下界": sl["ci_lo"], "上界": sl["ci_hi"]}],
             "v3": v3[f"census_{v3k}"],
             "a11": _test(f"猴子方法 {lab} vs 中職 {lab}，修正 11 世代（{'既有數字重算' if m == 'kbb' else '新'}）",
                          a11, mc, yc, rng),
             "kbo_t1": _existing(next(x for x in kbo["t1"]["k1"] if x["test"].startswith(k1)),
                                 f"韓職猴子方法 {lab} vs 中職 {lab}（既有）"),
             "kbo_t2": _existing(kbo["t2"]["pitchers"]["rho"] if m == "kbb" else
                                 next(x for x in kbo["t2"]["pitchers"]["components"] if x["test"] == t2),
                                 f"AAA 猴子方法 {lab} vs 韓職 {lab}（既有）")}
        out["metrics"][m] = r
    cen["neg_marcel_bb"] = -cen.marcel_bb                     # fewer walks = better, so the sign is flipped
    out["cross"] = [_test("猴子方法 K% vs 中職 K−BB%", cen, "marcel_k", "cpbl_kbb_c", rng),
                    _test("猴子方法 BB%（取負號，少保送為正）vs 中職 K−BB%", cen, "neg_marcel_bb", "cpbl_kbb_c", rng),
                    _test("猴子方法 K−BB% vs 中職 K−BB%（參照）", cen, "marcel", "cpbl_kbb_c", rng)]
    processed("metric_reports_summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float))
    report(out, cen)
    return out


def _ranking(C: pd.DataFrame, m: str) -> pd.DataFrame:
    col = METRICS[m][1]
    d = C.sort_values(col, ascending=(m == "bb")).copy()
    d[f"{METRICS[m][0]} 名次"] = np.arange(1, len(d) + 1)
    d["K%"], d["BB%"], d["K−BB%"] = 100 * d.marcel_k_pct, 100 * d.marcel_bb_pct, 100 * d.marcel_score
    d["主排名（可簽者）"] = d.combined_rank.rank(method="first").astype(int)
    return d[[f"{METRICS[m][0]} 名次", "name", "sign_tier", "K%", "BB%", "K−BB%", "主排名（可簽者）"]].rename(
        columns={"name": "球員", "sign_tier": "可簽性"})


def cpbl_names() -> pd.Series:
    """中職登錄名 by MLBAM id where a reference table has it (REBAS 2024-25 pitchers, 2026 site list)."""
    ref = ROOT / "data" / "reference"
    a = pd.read_csv(ref / "cpbl_foreign_pitchers.csv", comment="#")[["mlbam_id", "rebas_name"]]
    b = pd.read_csv(ref / "cpbl_imports_2026_site.csv", comment="#")[["mlbam_id", "site_name"]]
    s = pd.concat([b.rename(columns={"site_name": "n"}), a.rename(columns={"rebas_name": "n"})]).dropna()
    s = s.drop_duplicates("mlbam_id")
    return s.set_index(s.mlbam_id.astype(int)).n


def report(out: dict, cen: pd.DataFrame) -> None:
    C = candidates()
    b = cfg()["bootstrap"]
    L = ["# K%、BB%、K−BB% 分開的報告（修正 19；由 `python -m src.metric_reports` 產生）", "",
         "Johnny：「先不管權重，先分別做 K%、BB% 跟 K−BB% 分開的報告」。三個指標各一節，**不加權、不合併**，"
         "這份報告不改變分數、排名或首選。數字都是猴子方法（Marcel，修正 15）的預測；「新」是這次登錄後才算的，"
         f"「既有」是之前報告已公布的。新檢定：bootstrap {b['draws']} 次、90% CI、n ≥ 15 才下判定。", "",
         "名詞：中職、AAA 的數字都已減去當年聯盟平均（比平均高多少）；BB% 越低越好，所以 BB% 的 ρ 為正代表"
         "「AAA 少保送的人，到中職也少保送」。斜率 = AAA 比平均好 1 個百分點，到中職還剩多少。", "",
         f"排名對象：主排名裡有 Fit 資料的可簽洋投（T1 + T2）{len(C)} 人，和主排名同一批人；這些人還沒有中職登錄名，"
         "所以用英文名。「主排名（可簽者）」是現在正式排名在這 57 人中的名次。", ""]
    for m in cfg()["metrics"]:
        lab = METRICS[m][0]
        r = out["metrics"][m]
        v = r["v3"]
        L += [f"## {lab}", "",
              f"### 可簽洋投依猴子方法 {lab} 排名（{'越低越好' if m == 'bb' else '越高越好'}）", "",
              md_table(_ranking(C, m).head(15), ".1f"), "",
              f"### {lab} 的實際資料驗證", "",
              "**中職洋將普查 2018–2026（53 人）：排序相關**", "", md_table(pd.DataFrame(r["census"]), ".2f"), "",
              "**轉換斜率**", "", md_table(pd.DataFrame(r["slopes"]), ".2f"), "",
              "猴子方法本身已經把成績往平均拉回，所以它的斜率接近 1 才代表「預測多少，中職就差不多多少」；"
              "單季 AAA 沒有拉回，斜率小於 1 是正常的。兩者不能直接比大小。", "",
              f"**滾動預測（只用更早來台的人訓練，{v['n_held_out']} 人）**：預測誤差 {100 * v['mae_marcel']:.2f} 個百分點，"
              f"直接猜平均 {100 * v['mae_mean']:.2f}，差 {100 * v['diff']:+.2f}（90% CI {100 * v['ci'][0]:+.2f} 至 "
              f"{100 * v['ci'][1]:+.2f}）：" + ("**比猜平均準**。" if v["ci"][1] < 0 else "沒有確定比猜平均準。"), "",
              "**其他樣本**", "",
              md_table(pd.DataFrame([r["a11"], r["kbo_t1"], r["kbo_t2"]]), ".2f"), "",
              "修正 11 世代只有 17 人；韓職→中職 16 人；AAA→韓職 95 人（同一人多季以人為單位抽樣）。", ""]
    L += ["## 跨指標（描述性）：哪一項能排出來台後的 K−BB%", "",
          "Q&A 串事前看過的探索檢查，這裡用登錄的亂數種子重算。", "",
          md_table(pd.DataFrame(out["cross"]), ".2f"), ""]
    n = cfg()["appendix_top_n"]
    top = cen.dropna(subset=["marcel"]).sort_values("marcel", ascending=False).head(n).copy()
    med = cen.cpbl_kbb_c.median()
    top["猴子 K−BB%"], top["猴子 K%"], top["猴子 BB%"] = 100 * top.marcel, 100 * top.marcel_k, 100 * top.marcel_bb
    top["中職 K−BB%"] = 100 * top.cpbl_kbb_c
    top["中職後半段"] = np.where(top.cpbl_kbb_c < med, "是", "")
    top["中職登錄名"] = top.mlbam_id.astype(int).map(cpbl_names()).fillna("")
    bad = top[top.cpbl_kbb_c < med]
    L += [f"## 附錄（描述性，登錄前已看過）：普查猴子方法 K−BB% 前 {n} 名", "",
          f"所有數字都是比聯盟平均高多少（百分點）。「中職後半段」= 來台後 K−BB% 低於 53 人的中位數。前 {n} 名中有 "
          f"{len(bad)} 人落在後半段，其中猴子方法 K% 低於平均的有 {int((bad.marcel_k < 0).sum())} 人；"
          f"K% 高於平均的 {int((top.marcel_k > 0).sum())} 人中，落在後半段的有 {int(((top.marcel_k > 0) & (top.cpbl_kbb_c < med)).sum())} 人。"
          "這是事後挑出來看的，不是檢定。中職登錄名只有 2024 年以後的洋投有對照表，其他人空白。", "",
          md_table(top[["中職登錄名", "name", "season", "猴子 K−BB%", "猴子 K%", "猴子 BB%", "中職 K−BB%", "cpbl_bf", "中職後半段"]]
                   .rename(columns={"name": "球員", "season": "來台年", "cpbl_bf": "中職打席"}), ".1f"), "",
          "資料來源：中職成績 Baseball-Reference；韓職 FanGraphs；AAA MLB Stats API。", ""]
    (path("reports") / "metric_reports.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    print(json.dumps(run(), indent=1, ensure_ascii=False, default=float)[:2500])
