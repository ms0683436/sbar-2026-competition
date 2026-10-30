"""Amendment 22 — AAA -> KBO hitter components: does AAA Marcel K%, BB% or ISO carry over to the
next KBO season? Rules in config/scoring.yaml, amendment22. A report only: nothing here changes the
score, the ranking or the picks.

Sample = amendment 16's T2 hitter cohort (AAA D-1 -> KBO D, D 2016-2026, >= 150 PA both sides).

Usage:  python -m src.hitter_kbo      (after kbo)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import kbo, marcel
from .common import config, md_table, path, processed

LABEL = {"woba": "wOBA", "k": "K%", "bb": "BB%", "iso": "ISO"}


def cfg() -> dict:
    return config()["amendment22"]


def _kbo_rates(D: int) -> pd.DataFrame:
    """KBO season-D K%, BB%, ISO per player, each centred on that season's KBO average."""
    s = kbo.stints("hitting", D)
    iso = s.batting_TB - s.batting_H
    r = pd.DataFrame({"player_id": s.player_id,
                      "kbo_k_c": s.batting_SO / s.batting_PA - s.batting_SO.sum() / s.batting_PA.sum(),
                      "kbo_bb_c": s.batting_BB / s.batting_PA - s.batting_BB.sum() / s.batting_PA.sum(),
                      "kbo_iso_c": iso / s.batting_AB - iso.sum() / s.batting_AB.sum()})
    return r[r.player_id > 0].drop_duplicates("player_id").set_index("player_id")


def cohort() -> pd.DataFrame:
    H = kbo.t2_cohort("hitter")
    rows = []
    for D, g in H.groupby("dest"):
        m = marcel.project("hitter", D, seasons=[D - 1, D - 2, D - 3]).set_index("player_id")
        k = _kbo_rates(D)
        g = g.copy()
        for c in cfg()["components"]:
            g[f"aaa_m_{c}_c"] = g.player_id.map(m[c] - m[f"lg_{c}"]).to_numpy()
            g[f"kbo_{c}_c"] = g.player_id.map(k[f"kbo_{c}_c"]).to_numpy()
        rows.append(g)
    d = pd.concat(rows, ignore_index=True)
    d["aaa_m_woba_c"], d["kbo_woba_c"] = d.aaa_m, d.y
    return d


def run() -> dict:
    d = cohort()
    rng = np.random.default_rng(cfg()["bootstrap"]["seed"])
    out = {"rows": len(d), "players": int(d.player_id.nunique()), "tests": []}
    for c in ["woba", *cfg()["components"]]:
        x, y = f"aaa_m_{c}_c", f"kbo_{c}_c"
        out["tests"].append({"component": c,
                             "rho": kbo.rho_test(f"AAA Marcel {LABEL[c]} vs KBO {LABEL[c]}", d, x, y, rng, "player_id"),
                             "line": kbo.ols_ci(f"KBO_c = a + b * AAA Marcel_c ({LABEL[c]})", d, x, y, rng, "player_id"),
                             "mean_change": kbo.mean_ci(f"KBO_c - AAA Marcel_c ({LABEL[c]})", d[y] - d[x], rng, d.player_id)})
    keep = ["name", "player_id", "dest", "pa", "aaa_exposure"] + [f"{p}_{c}_c" for c in ["woba", *cfg()["components"]]
                                                                   for p in ("aaa_m", "kbo")]
    d[keep].to_csv(processed("hitter_kbo_cohort.csv"), index=False)
    processed("hitter_kbo_summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float))
    report(out, d)
    return out


def report(out: dict, d: pd.DataFrame) -> None:
    pts = lambda c, v: f"{v * 1000:+.0f}" if c in ("woba", "iso") else f"{v * 100:+.1f}"
    unit = lambda c: "（千分位）" if c in ("woba", "iso") else "（百分點）"
    rows = []
    for t in out["tests"]:
        c, r, l, m = t["component"], t["rho"], t["line"], t["mean_change"]
        rows.append([LABEL[c], r["n"], f"{r['rho']:.2f}（{r['ci_lo']:.2f} 至 {r['ci_hi']:.2f}）", r["verdict"],
                     f"{l['slope']:.2f}（{l['slope_lo']:.2f} 至 {l['slope_hi']:.2f}）",
                     f"{pts(c, m['mean'])}（{pts(c, m['ci_lo'])} 至 {pts(c, m['ci_hi'])}）{unit(c)}"])
    L = ["# 修正 22：AAA → 韓職打者的分項指標", "",
         f"樣本：修正 16 的 T2 打者（AAA 前一季 → 韓職，2016–2026，兩邊都 ≥ 150 打席），{out['players']} 人、{out['rows']} 筆。",
         "預測值：AAA Marcel（權重 5/4/3、收斂 1,200 打席、年齡調整、球場校正），各項減 AAA 平均；結果：韓職當季同項指標減韓職平均。",
         "Spearman 以球員為單位重抽 2,000 次，90% 信賴區間；n ≥ 15 且區間全在 0 以上才判「支持」。只做報告，不改分數、排名或人選。", "",
         md_table(pd.DataFrame(rows, columns=["指標", "人數", "排序相關（90% CI）", "判定", "斜率（90% CI）", "韓職相對 AAA 的平均變化（90% CI）"])), "",
         "wOBA 一列是重算修正 16 的結果，作為核對。", "資料來源：FanGraphs（韓職）、MLB statsapi（AAA）。"]
    (path("reports") / "hitter_kbo.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    run()
