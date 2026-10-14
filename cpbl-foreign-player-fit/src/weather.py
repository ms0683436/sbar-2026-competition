"""Amendment 10 — do AAA pitchers have a personal hot-weather split that repeats year to year?

Weather per AAA game from the statsapi schedule (hydrate=weather: temperature in F, condition,
wind; no humidity), cached in data/raw/statsapi_season/weather_{season}.csv. Indoor games
(condition Dome / Roof Closed) are dropped. For each pitcher-season the hot (>= 85 F) minus
other split of K%, whiff% and FF + SI velocity is computed on outcomes centred by home club
and calendar month (config `amendment10`, committed before any weather value was read).
The test is the 2025-vs-2026 Spearman of the split; nothing here enters the score.

Usage:  python -m src.weather
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import backtest, milb_loader, park, statsapi_pbp, statsapi_release
from .common import config, md_table, path, processed, raw


def cfg() -> dict:
    return config()["amendment10"]


# ------------------------------------------------------------------ weather
def fetch(season: int) -> pd.DataFrame:
    f = raw("statsapi_season", f"weather_{season}.csv")
    if f.exists():
        return pd.read_csv(f)
    lo, hi = statsapi_release.SEASONS[season]
    rows = []
    lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
    for m in pd.period_range(lo, hi, freq="M"):               # one request per month
        a, b = max(m.start_time, lo), min(m.end_time.normalize(), hi)
        r = statsapi_pbp._get(f"{statsapi_pbp.API}/v1/schedule",
                              {"sportId": 11, "startDate": a.date().isoformat(), "endDate": b.date().isoformat(),
                               "gameType": "R", "hydrate": "weather"})
        for d in r.get("dates", []):
            for g in d["games"]:
                if g.get("status", {}).get("abstractGameState") != "Final":
                    continue
                w = g.get("weather") or {}
                rows.append({"game_pk": g["gamePk"], "season": season, "temp_f": w.get("temp"),
                             "condition": w.get("condition"), "wind": w.get("wind")})
    df = pd.DataFrame(rows).drop_duplicates("game_pk")
    df["temp_f"] = pd.to_numeric(df.temp_f, errors="coerce")
    f.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(f, index=False)
    return df


def games() -> pd.DataFrame:
    w = pd.concat([fetch(s) for s in cfg()["seasons"]], ignore_index=True)
    indoor = w.condition.fillna("").str.contains("|".join(cfg()["indoor_conditions"]), case=False)
    w["outdoor"] = ~indoor & w.temp_f.notna()
    return w


# ------------------------------------------------------------------ splits
def _centre(d: pd.DataFrame, col: str) -> pd.Series:
    """value - season x club mean - season x month mean + season mean."""
    s = d.groupby("season")[col].transform("mean")
    c = d.groupby(["season", "home_club"])[col].transform("mean")
    m = d.groupby(["season", "month"])[col].transform("mean")
    return d[col] - c - m + s


def pitches(a: pd.DataFrame, w: pd.DataFrame) -> pd.DataFrame:
    a = a[a.season.isin(cfg()["seasons"])].copy()
    a["game_pk"] = a.game_id.astype(int)
    a = a.merge(w[w.outdoor][["game_pk", "temp_f"]], on="game_pk", how="inner")
    h = pd.concat([park.home_clubs(int(s)) for s in cfg()["seasons"]]).drop_duplicates("game_pk")
    a["home_club"] = a.game_pk.map(h.set_index("game_pk").home_id)
    a["month"] = pd.to_datetime(a.date).dt.month
    return a[a.home_club.notna()]


def splits(a: pd.DataFrame, hot_f: float) -> pd.DataFrame:
    """Per pitcher-season: hot minus other of centred K%, whiff% and FF + SI velocity, with
    the group sizes used for the minimums."""
    c, mins = cfg(), cfg()["min_each_group"]
    a = a.assign(hot=a.temp_f >= hot_f)
    pa = a[a.pa_end.astype(bool) & a.pa_event.ne("OTHER")].assign(so=lambda x: x.pa_event.eq("SO").astype(float))
    sw = a[a.swing.astype(bool) & ~a.is_bunt.astype(bool)].assign(wh=lambda x: x.whiff.astype(float))
    fb = a[a.pitch_type_raw.isin(c["fastballs"]) & a.velo_mph.notna()]
    out = []
    for name, d, col, nmin in (("k_pct", pa, "so", mins["bf"]), ("whiff_pct", sw, "wh", mins["swings"]),
                               ("velo", fb, "velo_mph", mins["fastballs"])):
        d = d.assign(v=_centre(d, col))
        g = (d.groupby(["season", "pitcher_id", "hot"]).v.agg(["mean", "size"]).unstack("hot")
             .reindex(columns=pd.MultiIndex.from_product([["mean", "size"], [False, True]])))
        ok = (g[("size", True)] >= nmin) & (g[("size", False)] >= nmin)
        sp = (g[("mean", True)] - g[("mean", False)])[ok].rename(name)
        out.append(sp)
    return pd.concat(out, axis=1).reset_index()


def reliability(sp: pd.DataFrame, rng) -> list[dict]:
    b, c = cfg()["bootstrap"], cfg()
    y0, y1 = c["seasons"]
    res = []
    for m in ("k_pct", "whiff_pct", "velo"):
        w = sp.pivot_table(index="pitcher_id", columns="season", values=m).dropna()
        if len(w) < 3:
            res.append({"split": m, "n": len(w), "rho": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "verdict": "too few"})
            continue
        df = w.rename(columns={y0: "a", y1: "b"})
        est, lo, hi = backtest.boot_ci(df, "a", "b", b["draws"], b["ci"], rng)
        res.append({"split": m, "n": len(w), "rho": est, "ci_lo": lo, "ci_hi": hi,
                    "verdict": "pass" if est >= c["pass_rho"] else "fail",
                    f">= {c['report_rho']}": bool(est >= c["report_rho"])})
    return res


def league_effect(sp: pd.DataFrame) -> dict:
    return {m: {str(s): float(g[m].mean()) for s, g in sp.groupby("season")} for m in ("k_pct", "whiff_pct", "velo")}


def run() -> dict:
    rng = np.random.default_rng(cfg()["bootstrap"]["seed"])
    w = games()
    p = pitches(milb_loader.load(), w)
    out = {"games": {str(s): {"total": int((w.season == s).sum()), "outdoor_with_temp": int((w.outdoor & (w.season == s)).sum()),
                              "hot": int((w.outdoor & (w.season == s) & (w.temp_f >= cfg()["hot_f"])).sum())}
                     for s in cfg()["seasons"]},
           "temp_f_quantiles": {str(q): float(w.loc[w.outdoor, "temp_f"].quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}}
    for hot in [cfg()["hot_f"], *cfg()["sensitivity_hot_f"]]:
        sp = splits(p, hot)
        if hot == cfg()["hot_f"]:
            sp.to_csv(processed("weather_splits.csv"), index=False)
        out[f"hot_{hot}F"] = {"reliability": reliability(sp, rng), "league_mean_split": league_effect(sp)}
    processed("weather_summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    report(out)
    return out


def report(out: dict) -> None:
    c = cfg()
    L = ["# Hot-weather split (amendment 10; generated by `python -m src.weather`)", "",
         "AAA 2025 and 2026, outdoor games with a temperature (statsapi schedule weather; no humidity). "
         f"Hot = >= {c['hot_f']} F. Outcomes centred by home club and month before splitting. "
         f"Pass bar: 2025-vs-2026 Spearman >= {c['pass_rho']} (also reported: >= {c['report_rho']}). "
         "Nothing here enters the score.", "",
         md_table(pd.DataFrame(out["games"]).T.reset_index(names="season"), ".0f"), "",
         "Outdoor temperature quantiles (F): " + ", ".join(f"{k}: {v:.0f}" for k, v in out["temp_f_quantiles"].items()), ""]
    for k, v in out.items():
        if not k.startswith("hot_"):
            continue
        L += [f"## {k.replace('_', ' ')}{' (primary)' if k == f'hot_{c['hot_f']}F' else ' (sensitivity)'}", "",
              md_table(pd.DataFrame(v["reliability"]), ".2f"), "",
              "League mean split (hot minus other, centred; K% and whiff% as fractions, velocity in mph):", "",
              md_table(pd.DataFrame(v["league_mean_split"]).T.reset_index(names="split"), ".4f"), ""]
    (path("reports") / "weather.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    print(json.dumps(run(), indent=1, ensure_ascii=False))
