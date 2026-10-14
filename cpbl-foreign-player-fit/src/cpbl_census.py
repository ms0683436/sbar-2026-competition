"""Amendment 12 — census of CPBL imports 2018-2026 (Baseball-Reference register) and a
larger test of whether AAA performance carries over to the CPBL, for pitchers and hitters.

Source: baseball-reference.com register pages (CPBL team-season pages and each import's own
register page), fetched at most one request per `request_interval_s` and cached under
data/raw/bbref/ (git-ignored, not redistributed). AAA predictors and AAA league values come from
statsapi season stats; CPBL league values are the totals of every player on that season's team
pages. Rules in config `amendment12`, committed before any census page was downloaded.
Traditional stats only (no Fit). Nothing here changes the ranking or the picks.

Usage:  python -m src.cpbl_census
"""
from __future__ import annotations

import glob
import html
import json
import re
import time
import urllib.request

import numpy as np
import pandas as pd

from . import backtest, baseline_weights
from .common import ROOT, config, md_table, path, processed, raw

BR = "https://www.baseball-reference.com"
UA = "Mozilla/5.0 (research; sbar-2026 CPBL import census; <= 20 requests/min)"
AAA_LEAGUES = {"IL", "PCL", "AAAE", "AAAW"}
NPB_LEAGUES = {"CL", "PL", "NPB", "JPCL", "JPPL", "JCL", "JPL"}
_last = [0.0]


def cfg() -> dict:
    return config()["amendment12"]


# ------------------------------------------------------------------ fetching
def _get(rel: str, key: str) -> str:
    f = raw("bbref", f"{key}.html")
    if f.exists() and f.stat().st_size > 1000:
        return f.read_text(encoding="utf-8")
    wait = cfg()["request_interval_s"] - (time.time() - _last[0])
    if wait > 0:
        time.sleep(wait)
    req = urllib.request.Request(BR + rel, headers={"User-Agent": UA})
    for i in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                t = r.read().decode("utf-8", errors="ignore")
            break
        except Exception:
            if i == 3:
                raise
            time.sleep(30 * (i + 1))
    _last[0] = time.time()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(t, encoding="utf-8")
    return t


def _clean(t: str) -> str:
    return t.replace("<!--", "").replace("-->", "")


def _cell(v: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", v)).strip()


def _table(t: str, tid: str) -> pd.DataFrame:
    i = t.find(f'id="{tid}"')
    if i < 0:
        return pd.DataFrame()
    tb = t[i:t.find("</table>", i)]
    rows = []
    for r in re.findall(r"<tr([^>]*)>(.*?)</tr>", tb, re.S):
        cls, body = r
        d = {k: _cell(v) for k, v in re.findall(r'data-stat="([^"]+)"[^>]*>(.*?)</t[dh]>', body, re.S)}
        m = re.search(r"/register/player\.fcgi\?id=([\w-]+)", body)
        if m:
            d["bbref_id"] = m.group(1)
        d["row_class"] = cls
        rows.append(d)
    df = pd.DataFrame(rows)
    if len(df) and "ranker" in df:
        df = df[df.ranker.ne("Rk")]
    return df


def league_id(year: int) -> str:
    t = _clean(_get(f"/register/league.cgi?year={year}", f"leagues_{year}"))
    ids = {i for i, n in re.findall(r'href="/register/league\.cgi\?id=(\w+)">([^<]+)</a>', t)
           if "Chinese Professional" in n}
    assert len(ids) == 1, (year, ids)
    return ids.pop()


def team_ids(year: int) -> list[str]:
    lid = league_id(year)
    t = _clean(_get(f"/register/league.cgi?id={lid}", f"league_{year}_{lid}"))
    return sorted(set(re.findall(r"/register/team\.cgi\?id=(\w+)", t)))


def team_tables(year: int) -> dict[str, pd.DataFrame]:
    out = {"batting": [], "pitching": [], "roster": []}
    for tid in team_ids(year):
        t = _clean(_get(f"/register/team.cgi?id={tid}", f"team_{year}_{tid}"))
        for k, name in (("batting", "team_batting"), ("pitching", "team_pitching"), ("roster", "standard_roster")):
            d = _table(t, name)
            if len(d):
                out[k].append(d.assign(season=year, team_id=tid))
    return {k: pd.concat(v, ignore_index=True) for k, v in out.items()}


def player_rows(pid: str) -> dict:
    t = _clean(_get(f"/register/player.fcgi?id={pid}", f"player_{pid}"))
    m = re.search(r"Born:</strong>(.*?)</p>", t, re.S) or re.search(r"Born:(.*?)</p>", t, re.S)
    born = " ".join(_cell(m.group(1)).split()) if m else ""
    return {"born": born, "pitching": _table(t, "standard_pitching"), "batting": _table(t, "standard_batting")}


# ------------------------------------------------------------------ census
NUM = ["batters_faced", "SO", "BB", "HBP", "HR", "IP", "ER", "H", "PA", "AB", "2B", "3B", "SF", "IBB"]


def _num(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    for c in NUM:
        if c in d:
            d[c] = pd.to_numeric(d[c].replace("", np.nan), errors="coerce")
    return d


def _country(born: str) -> str:
    m = re.search(r"\sin\s.*\s([a-z]{2})\s*$", born.strip())
    return m.group(1).upper() if m else ""


def _category(lg: str, level: str) -> str:
    level, lg = (level or "").strip(), (lg or "").strip()
    if level == "Maj":
        return "MLB"
    if level == "AAA" and lg in AAA_LEAGUES:
        return "AAA"
    if lg == "CPBL":
        return "CPBL"
    if lg == "KBO":
        return "KBO"
    if lg in NPB_LEAGUES or "NPB" in lg:
        return "NPB"
    if level in {"AA", "A+", "A", "A-", "Rk", "FRk", "A--Rk"}:
        return "AA or lower"
    return "other"


def history(p: dict) -> pd.DataFrame:
    """One row per (season, category) with BF and PA, summary rows ('N Lgs') dropped."""
    rows = []
    for kind, col in (("pitching", "batters_faced"), ("batting", "PA")):
        d = p[kind]
        if not len(d) or "year_ID" not in d:
            continue
        lg = d["lg_ID"].astype(str) if "lg_ID" in d else pd.Series("", index=d.index)
        d = _num(d[d.year_ID.str.fullmatch(r"\d{4}", na=False) & ~lg.str.contains("Lgs")])
        d["cat"] = [_category(a, b) for a, b in zip(d.get("lg_ID", ""), d.get("level", ""))]
        g = d.groupby([d.year_ID.astype(int), "cat"])[col].sum(min_count=1).rename(col)
        rows.append(g)
    if not rows:
        return pd.DataFrame(columns=["season", "cat"])
    return pd.concat(rows, axis=1).reset_index().rename(columns={"year_ID": "season"})


def census() -> tuple[pd.DataFrame, dict]:
    c = cfg()
    y0, y1 = c["seasons"]
    tabs = {y: team_tables(y) for y in range(y0, y1 + 1)}
    roster = pd.concat([t["roster"] for t in tabs.values()], ignore_index=True)
    roster["country"] = roster.birth_city.fillna("").map(_country)
    reg = _register()
    roster["mlbam_id"] = roster.bbref_id.map(reg)
    # player pages: everyone born outside Taiwan, and blank birthplaces that have an MLBAM id
    todo = roster[(roster.country.ne("") & roster.country.ne("TW")) | (roster.country.eq("") & roster.mlbam_id.notna())]
    hist, born = {}, {}
    for pid in sorted(todo.bbref_id.dropna().unique()):
        p = player_rows(pid)
        born[pid] = p["born"]
        hist[pid] = history(p)
    leagues = {y: _cpbl_league(t) for y, t in tabs.items()}
    rows = []
    for pid, h in hist.items():
        ctry = _country(born[pid]) or roster.loc[roster.bbref_id == pid, "country"].iat[0]
        cp = h[h.cat.eq("CPBL")]
        if cp.empty:
            continue
        first = int(cp.season.min())
        r = roster[roster.bbref_id == pid].iloc[0]
        prev = h[h.season.eq(first - 1)]
        def top(col):
            if col not in prev:
                return "none"
            q = prev.dropna(subset=[col])
            q = q[q[col] > 0]
            return q.sort_values(col).cat.iat[-1] if len(q) else "none"
        rows.append({"bbref_id": pid, "name": r.player.rstrip("*#").strip(), "mlbam_id": r.mlbam_id, "born": born[pid],
                     "country": ctry, "import": bool(ctry) and ctry != "TW", "first_cpbl": first,
                     "cpbl_seasons": sorted(cp.season.astype(int).tolist()),
                     "origin_pitch": top("batters_faced"), "origin_bat": top("PA"),
                     "returned_next": bool(((cp.season == first + 1)).any())})
    cen = pd.DataFrame(rows)
    return cen, {"tabs": tabs, "leagues": leagues, "roster": roster}


def _register() -> dict:
    ext = (ROOT / config()["paths"]["external"]).resolve()
    fs = glob.glob(str(ext / "register" / "data" / "people-*.csv"))
    d = pd.concat([pd.read_csv(f, usecols=["key_mlbam", "key_bbref_minors"], dtype=str) for f in fs])
    d = d.dropna()
    return dict(zip(d.key_bbref_minors, d.key_mlbam.astype(float).astype(int)))


def _woba(d: pd.DataFrame) -> pd.Series:
    w = config()["woba_weights"]
    ubb = d.BB - d.IBB.fillna(0)
    b1 = d.H - d["2B"] - d["3B"] - d.HR
    num = w["uBB"] * ubb + w["HBP"] * d.HBP + w["1B"] * b1 + w["2B"] * d["2B"] + w["3B"] * d["3B"] + w["HR"] * d.HR
    den = d.AB + ubb + d.SF.fillna(0) + d.HBP
    return num / den


def _bat_rates(d: pd.DataFrame) -> pd.DataFrame:
    d = _num(d)
    return pd.DataFrame({"pa": d.PA, "k_pct": d.SO / d.PA, "bb_pct": d.BB / d.PA,
                         "iso": (d["2B"] + 2 * d["3B"] + 3 * d.HR) / d.AB, "woba": _woba(d)})


def _cpbl_league(t: dict) -> dict:
    p = _num(t["pitching"][t["pitching"].get("bbref_id").notna()])
    b = _num(t["batting"][t["batting"].get("bbref_id").notna()])
    bs = b[NUM[7:] + ["SO", "BB", "HBP", "HR"]].sum().to_frame().T
    lr = _bat_rates(bs).iloc[0]
    return {"k_pct": p.SO.sum() / p.batters_faced.sum(), "bb_pct": p.BB.sum() / p.batters_faced.sum(),
            "bat_k_pct": lr.k_pct, "bat_bb_pct": lr.bb_pct, "iso": lr.iso, "woba": lr.woba}


# ------------------------------------------------------------------ AAA predictors
def aaa_season(y: int) -> int:
    return int(cfg()["aaa_season_override"].get(y, y - 1))


def _aaa_raw(group: str, season: int) -> pd.DataFrame:
    baseline_weights.fetch(group, season)                     # caches the per-club stints
    return pd.read_csv(raw("statsapi_season", f"{season}_{group}_by_team.csv"))


def aaa_pitchers(season: int) -> pd.DataFrame:
    s = _aaa_raw("pitching", season)
    g = s.groupby("player_id")[["pitching_BF", "pitching_SO", "pitching_BB", "pitching_HBP", "pitching_HR",
                                 "pitching_ER", "outs"]].sum()
    lg = g.sum()
    ip, lip = g.outs / 3, lg.outs / 3
    lg_k, lg_bb = lg.pitching_SO / lg.pitching_BF, lg.pitching_BB / lg.pitching_BF
    lg_fip_raw = (13 * lg.pitching_HR + 3 * (lg.pitching_BB + lg.pitching_HBP) - 2 * lg.pitching_SO) / lip
    cfip = 9 * lg.pitching_ER / lip - lg_fip_raw
    d = pd.DataFrame({"aaa_bf": g.pitching_BF,
                      "aaa_k_c": g.pitching_SO / g.pitching_BF - lg_k, "aaa_bb_c": g.pitching_BB / g.pitching_BF - lg_bb})
    d["aaa_kbb_c"] = d.aaa_k_c - d.aaa_bb_c
    d["aaa_fip"] = (13 * g.pitching_HR + 3 * (g.pitching_BB + g.pitching_HBP) - 2 * g.pitching_SO) / ip + cfip
    pool = d[d.aaa_bf >= cfg()["min_aaa_bf"]]
    w, clip = cfg()["baseline_weights"], cfg()["baseline_clip"]
    z = lambda x, sign: (sign * (x - pool[x.name].mean()) / pool[x.name].std(ddof=0)).clip(-clip, clip)
    d["baseline"] = w["k_pct"] * z(d.aaa_k_c, 1) + w["bb_pct"] * z(d.aaa_bb_c, -1) + w["fip"] * z(d.aaa_fip, -1)
    return d


def aaa_hitters(season: int) -> pd.DataFrame:
    s = _aaa_raw("hitting", season)
    s = s[s.player_position != "P"]
    m = {"batting_PA": "PA", "batting_AB": "AB", "batting_H": "H", "batting_2B": "2B", "batting_3B": "3B",
         "batting_HR": "HR", "batting_BB": "BB", "batting_IBB": "IBB", "batting_HBP": "HBP", "batting_SF": "SF",
         "batting_SO": "SO"}
    g = s.groupby("player_id")[[k for k in m if k in s]].sum().rename(columns=m)
    for k in m.values():
        if k not in g:
            g[k] = 0
    r, lr = _bat_rates(g), _bat_rates(g.sum().to_frame().T).iloc[0]
    return pd.DataFrame({"aaa_pa": r.pa, "aaa_woba_c": r.woba - lr.woba, "aaa_k_c": r.k_pct - lr.k_pct,
                         "aaa_bb_c": r.bb_pct - lr.bb_pct, "aaa_iso_c": r.iso - lr.iso})


# ------------------------------------------------------------------ cohorts
def cohorts(cen: pd.DataFrame, ctx: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    c = cfg()
    y0, y1 = c["seasons"]
    imp = cen[cen["import"] & cen.first_cpbl.between(y0, y1) & cen.mlbam_id.notna()]
    P, H = [], []
    for y, g in imp.groupby("first_cpbl"):
        y = int(y)
        a = aaa_season(y)
        ap, ah, lg, t = aaa_pitchers(a), aaa_hitters(a), ctx["leagues"][y], ctx["tabs"][y]
        pt = _num(t["pitching"]).groupby("bbref_id")[["batters_faced", "SO", "BB"]].sum()
        bt = _num(t["batting"]).groupby("bbref_id")[NUM[7:] + ["SO", "BB", "HBP", "HR"]].sum()
        br = _bat_rates(bt)
        for r in g.itertuples(index=False):
            base = {"name": r.name, "bbref_id": r.bbref_id, "mlbam_id": int(r.mlbam_id), "season": y, "aaa_season": a,
                    "origin": None}
            if r.bbref_id in pt.index and pt.loc[r.bbref_id, "batters_faced"] >= c["min_cpbl_bf"] \
                    and int(r.mlbam_id) in ap.index and ap.loc[int(r.mlbam_id), "aaa_bf"] >= c["min_aaa_bf"]:
                q = pt.loc[r.bbref_id]
                k, b = q.SO / q.batters_faced, q.BB / q.batters_faced
                P.append(base | {"origin": r.origin_pitch, "cpbl_bf": q.batters_faced, "cpbl_k_c": k - lg["k_pct"],
                                 "cpbl_bb_c": b - lg["bb_pct"], "cpbl_kbb_c": (k - lg["k_pct"]) - (b - lg["bb_pct"])}
                         | ap.loc[int(r.mlbam_id)].to_dict())
            if r.bbref_id in br.index and br.loc[r.bbref_id, "pa"] >= c["min_cpbl_pa"] \
                    and int(r.mlbam_id) in ah.index and ah.loc[int(r.mlbam_id), "aaa_pa"] >= c["min_aaa_pa"]:
                q = br.loc[r.bbref_id]
                H.append(base | {"origin": r.origin_bat, "cpbl_pa": q.pa, "cpbl_woba_c": q.woba - lg["woba"],
                                 "cpbl_k_c": q.k_pct - lg["bat_k_pct"], "cpbl_bb_c": q.bb_pct - lg["bat_bb_pct"],
                                 "cpbl_iso_c": q.iso - lg["iso"]} | ah.loc[int(r.mlbam_id)].to_dict())
    return pd.DataFrame(P), pd.DataFrame(H)


# ------------------------------------------------------------------ tests
def _verdict(n, lo, hi):
    if n < cfg()["min_n_for_conclusion"]:
        return "descriptive only (n below minimum)"
    return "supports" if lo > 0 else "contradicts" if hi < 0 else "inconclusive"


def _test(label, d, x, y, rng):
    b = cfg()["bootstrap"]
    d = d[[x, y]].astype(float).dropna() if len(d) else pd.DataFrame(columns=[x, y])
    if len(d) < 3:
        return {"test": label, "n": len(d), "rho": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "verdict": "too few"}
    est, lo, hi = backtest.boot_ci(d, x, y, b["draws"], b["ci"], rng)
    return {"test": label, "n": len(d), "rho": est, "ci_lo": lo, "ci_hi": hi, "verdict": _verdict(len(d), lo, hi)}


def _slope(label, d, x, y, rng):
    b = cfg()["bootstrap"]
    m = d[[x, y]].astype(float).dropna().to_numpy()
    if len(m) < 3:
        return {"metric": label, "n": len(m)}
    k = np.polyfit(m[:, 0], m[:, 1], 1)[0]
    bs = [np.polyfit(m[i, 0], m[i, 1], 1)[0] for i in (rng.integers(0, len(m), len(m)) for _ in range(b["draws"]))
          if np.ptp(m[i, 0]) > 0]
    q = (1 - b["ci"]) / 2
    return {"metric": label, "n": len(m), "slope": k, "ci_lo": float(np.quantile(bs, q)), "ci_hi": float(np.quantile(bs, 1 - q))}


def rolling(d, x, y, rng) -> dict:
    b = cfg()["bootstrap"]
    rows = []
    for yr in sorted(d.season.unique()):
        tr, te = d[d.season < yr].dropna(subset=[x, y]), d[d.season == yr].dropna(subset=[x, y])
        if len(tr) < 3 or not len(te) or np.ptp(tr[x]) == 0:
            continue
        k, a = np.polyfit(tr[x], tr[y], 1)
        rows += [{"season": yr, "n_train": len(tr), "actual": v, "m0": tr[y].mean(), "m1": a + k * xv}
                 for xv, v in zip(te[x], te[y])]
    f = pd.DataFrame(rows)
    if f.empty:
        return {"n_held_out": 0}
    e0, e1 = (f.actual - f.m0).abs().to_numpy(), (f.actual - f.m1).abs().to_numpy()
    diffs = [e1[i].mean() - e0[i].mean() for i in (rng.integers(0, len(f), len(f)) for _ in range(b["draws"]))]
    q = (1 - b["ci"]) / 2
    return {"n_held_out": len(f), "mae_m0": float(e0.mean()), "mae_m1": float(e1.mean()),
            "m1_minus_m0": float(e1.mean() - e0.mean()), "ci": [float(np.quantile(diffs, q)), float(np.quantile(diffs, 1 - q))]}


P_TESTS = [("CP1 AAA K-BB% vs CPBL K-BB% (primary)", "aaa_kbb_c", "cpbl_kbb_c"),
           ("CP2 AAA K% vs CPBL K%", "aaa_k_c", "cpbl_k_c"),
           ("CP3 AAA BB% vs CPBL BB%", "aaa_bb_c", "cpbl_bb_c"),
           ("CP4 baseline vs CPBL K-BB%", "baseline", "cpbl_kbb_c")]
H_TESTS = [("CH1 AAA wOBA vs CPBL wOBA (primary)", "aaa_woba_c", "cpbl_woba_c"),
           ("CH2 AAA K% vs CPBL K%", "aaa_k_c", "cpbl_k_c"),
           ("CH3 AAA BB% vs CPBL BB%", "aaa_bb_c", "cpbl_bb_c"),
           ("CH4 AAA ISO vs CPBL ISO", "aaa_iso_c", "cpbl_iso_c")]


def analyse(P: pd.DataFrame, H: pd.DataFrame) -> dict:
    rng = np.random.default_rng(cfg()["bootstrap"]["seed"])
    s0 = cfg()["sensitivity_from"]
    out = {"pitchers": {"n": len(P), "by_season": P.season.value_counts().sort_index().to_dict() if len(P) else {},
                        "tests": [_test(l, P, x, y, rng) for l, x, y in P_TESTS],
                        "sensitivity_2021_on": [_test(l, P[P.season >= s0] if len(P) else P, x, y, rng) for l, x, y in P_TESTS],
                        "slopes": [_slope(l.split(" vs ")[0][4:], P, x, y, rng) for l, x, y in P_TESTS[:3]],
                        "rolling": rolling(P, "aaa_kbb_c", "cpbl_kbb_c", rng) if len(P) else {}},
           "hitters": {"n": len(H), "by_season": H.season.value_counts().sort_index().to_dict() if len(H) else {},
                       "tests": [_test(l, H, x, y, rng) for l, x, y in H_TESTS],
                       "sensitivity_2021_on": [_test(l, H[H.season >= s0] if len(H) else H, x, y, rng) for l, x, y in H_TESTS],
                       "slopes": [_slope(l.split(" vs ")[0][4:], H, x, y, rng) for l, x, y in H_TESTS],
                       "rolling": rolling(H, "aaa_woba_c", "cpbl_woba_c", rng) if len(H) else {}}}
    return out


def describe(cen: pd.DataFrame) -> dict:
    y0, y1 = cfg()["seasons"]
    imp = cen[cen["import"]]
    arr = imp[imp.first_cpbl.between(y0, y1)].copy()
    arr["origin"] = np.where(arr.origin_pitch.ne("none"), arr.origin_pitch, arr.origin_bat)
    by = arr.groupby("origin").agg(n=("name", "size"), returned_next=("returned_next", "mean"))
    return {"imports_seen": int(len(imp)), "arrivals_in_range": int(len(arr)),
            "returnees_first_before_range": int((imp.first_cpbl < y0).sum()),
            "unknown_birthplace": int((cen.country.eq("")).sum()),
            "arrivals_by_season": arr.first_cpbl.value_counts().sort_index().to_dict(),
            "origin": by.reset_index().to_dict(orient="records"),
            "with_mlbam": int(arr.mlbam_id.notna().sum())}


def data_check(P: pd.DataFrame, ctx: dict) -> dict:
    """B-R CPBL K% / BB% vs REBAS for 2024-25 pitchers in the amendment-2 reference."""
    from . import rebas_loader
    ref = backtest.reference().dropna(subset=["mlbam_id"])
    cp = rebas_loader.load()
    rows = []
    for y in (2024, 2025):
        pt = _num(ctx["tabs"][y]["pitching"]).groupby("bbref_id")[["batters_faced", "SO", "BB"]].sum()
        rmap = {v: k for k, v in _register().items()}
        for r in ref.itertuples(index=False):
            bid = rmap.get(int(r.mlbam_id))
            if bid not in pt.index:
                continue
            g = cp[(cp.season == y) & (cp.pitcher_id == r.rebas_id)]
            pa = g[g.pa_end.astype(bool) & g.pa_event.ne("OTHER")]
            if len(pa) < 60:
                continue
            q = pt.loc[bid]
            rows.append({"season": y, "name": r.english_name, "bf_br": q.batters_faced, "bf_rebas": len(pa),
                         "k_br": q.SO / q.batters_faced, "k_rebas": pa.pa_event.eq("SO").mean(),
                         "bb_br": q.BB / q.batters_faced, "bb_rebas": pa.pa_event.eq("uBB").mean()})
    d = pd.DataFrame(rows)
    if d.empty:
        return {"n": 0}
    return {"n": len(d), "mad_k": float((d.k_br - d.k_rebas).abs().mean()), "mad_bb": float((d.bb_br - d.bb_rebas).abs().mean()),
            "mean_bf_ratio": float((d.bf_br / d.bf_rebas).mean())}


# ------------------------------------------------------------------ run
def run() -> dict:
    cen, ctx = census()
    P, H = cohorts(cen, ctx)
    out = {"census": describe(cen), "data_check": data_check(P, ctx)} | analyse(P, H)
    out["cpbl_league"] = {str(k): v for k, v in ctx["leagues"].items()}
    cen.to_csv(processed("cpbl_census.csv"), index=False)
    P.to_csv(processed("cpbl_census_pitchers.csv"), index=False)
    H.to_csv(processed("cpbl_census_hitters.csv"), index=False)
    processed("cpbl_census_summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    report(cen, P, H, out)
    return out


def report(cen, P, H, out) -> None:
    c = out["census"]
    L = ["# CPBL import census 2018-2026 (amendment 12; generated by `python -m src.cpbl_census`)", "",
         "Source: Baseball-Reference register (CPBL team-season pages and player register pages); AAA from "
         "statsapi season stats. Traditional stats only. Only players who reached the CPBL first team appear.", "",
         f"Imports seen on 2018-2026 team pages: {c['imports_seen']}; first CPBL season in 2018-2026: "
         f"{c['arrivals_in_range']} (with an MLBAM id: {c['with_mlbam']}); first seen before 2018: "
         f"{c['returnees_first_before_range']}; players with unknown birthplace (excluded): {c['unknown_birthplace']}.", "",
         "Arrivals by season: " + ", ".join(f"{k}: {v}" for k, v in c["arrivals_by_season"].items()), "",
         "Where arrivals pitched / batted most in the season before (and share back in the CPBL the next season):", "",
         md_table(pd.DataFrame(c["origin"]), ".2f"), "",
         f"Data check vs REBAS 2024-25 (pitchers, n {out['data_check'].get('n', 0)}): "
         + (f"mean |K% diff| {out['data_check']['mad_k']:.4f}, mean |BB% diff| {out['data_check']['mad_bb']:.4f}, "
            f"BF ratio B-R/REBAS {out['data_check']['mean_bf_ratio']:.3f}." if out["data_check"].get("n") else "none."), ""]
    for role in ("pitchers", "hitters"):
        r = out[role]
        L += [f"## {role.capitalize()} (n {r['n']}, by arrival season {r['by_season']})", "",
              md_table(pd.DataFrame(r["tests"]), ".2f"), "", "Sensitivity, arrivals 2021-2026:", "",
              md_table(pd.DataFrame(r["sensitivity_2021_on"]), ".2f"), "",
              "How much of an AAA edge survives (OLS slope of centred CPBL on centred AAA, 90% CI):", "",
              md_table(pd.DataFrame(r["slopes"]), ".2f"), ""]
        f = r["rolling"]
        if f.get("n_held_out"):
            L += [f"Rolling forecast (primary outcome, trained on earlier arrivals): held-out {f['n_held_out']}, "
                  f"MAE training mean {f['mae_m0']:.4f}, MAE OLS on AAA {f['mae_m1']:.4f}, difference "
                  f"{f['m1_minus_m0']:+.4f} (90% CI {f['ci'][0]:+.4f} to {f['ci'][1]:+.4f}).", ""]
    (path("reports") / "cpbl_census.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    r = run()
    print(json.dumps({k: v for k, v in r.items() if k != "cpbl_league"}, indent=1, ensure_ascii=False, default=str))
