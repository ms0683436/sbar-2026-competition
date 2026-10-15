"""Amendment 16 — KBO: a KBO -> CPBL translation for pitchers, an AAA -> KBO line, and a reference
list of 2026 KBO foreign players (rules in config/scoring.yaml, amendment16; primary picks unchanged).

Sources: FanGraphs KBO leaderboards 2015-2026 (season totals, cached under data/raw/fangraphs_kbo,
git-ignored); CPBL lines from the Baseball-Reference census (amendment 12); AAA from statsapi via the
amendment-15 Marcel. Players are linked FanGraphs -> MLBAM with the Chadwick register: its FanGraphs
key first, otherwise a unique name + birth-year match confirmed by the statsapi birth date (deviation
D1 in reports/kbo.md: the register has no FanGraphs key for players who never reached the majors).
Birth country (statsapi) decides who is a foreign player.

Usage:  python -m src.kbo
"""
from __future__ import annotations

import glob
import json
import re
import unicodedata
from functools import lru_cache

import numpy as np
import pandas as pd

from . import backtest, cpbl_census, marcel
from .common import ROOT, config, ip_to_outs, md_table, path, processed, raw

KR = {"Korea", "South Korea", "Republic of Korea", "KOR"}
# FanGraphs names the register cannot match, found by the B-R coverage check (t1_coverage); each one
# confirmed by the B-R register line (same club, same BF) and the birth date.
ALIASES = {"sa3069091": 608571}     # "Mercedes c.c." = Cristopher Crisostomo (Kiwoom 2025, 197 BF on both)
# Deviation D2 (found after the run, in the availability news check): born abroad but a domestic KBO
# player, so not a foreign player; dropped from the candidate list (reported).
DOMESTIC = {"sa3012508": "Kwon Ju (Joo Kwon, KT): born in China, plays as a domestic player"}


def cfg() -> dict:
    return config()["amendment16"]


# ------------------------------------------------------------------ FanGraphs KBO
def _download(kind: str, y: int) -> list:
    import requests
    url = cfg()["fangraphs_url"].format(kind=kind, y=y)
    r = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0 (research; sbar-2026)"})
    r.raise_for_status()
    d = r.json()
    return d["data"] if isinstance(d, dict) else d


@lru_cache(maxsize=None)
def fg(kind: str, y: int) -> pd.DataFrame:
    """One FanGraphs leaderboard season (kind 'pit' / 'bat'); one row per player (multi-team = one row)."""
    f = raw("fangraphs_kbo", f"{kind}_{y}.json")
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(_download(kind, y)), encoding="utf-8")
    d = pd.DataFrame(json.loads(f.read_text(encoding="utf-8")))
    d["fg_id"] = d.playerids.astype(str)
    assert not d.fg_id.duplicated().any(), (kind, y)
    d["season"] = y
    d["age"] = pd.to_numeric(d.Age, errors="coerce")
    d["hangul"] = d.Name.astype(str).str.replace(r"<[^>]+>", "", regex=True).str.extract(r"([가-힣 ]+)$")[0].str.strip()
    return d


def fg_all(kind: str) -> pd.DataFrame:
    y0, y1 = cfg()["kbo_seasons"]
    return pd.concat([fg(kind, y) for y in range(y0, y1 + 1)], ignore_index=True)


# ------------------------------------------------------------------ linking
def _norm(s) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    return re.sub(r"[^a-z]", "", "".join(c for c in s if not unicodedata.combining(c)).lower())


@lru_cache(maxsize=1)
def register() -> pd.DataFrame:
    ext = (ROOT / config()["paths"]["external"]).resolve()
    fs = glob.glob(str(ext / "register" / "data" / "people-*.csv"))
    r = pd.concat([pd.read_csv(f, usecols=["key_mlbam", "key_fangraphs", "key_bbref_minors", "name_first",
                                           "name_last", "birth_year"], dtype=str) for f in fs])
    r = r.dropna(subset=["key_mlbam"]).reset_index(drop=True)
    r["mlbam"] = r.key_mlbam.astype(float).astype(int)
    r["nk"] = (r.name_first.fillna("") + r.name_last.fillna("")).map(_norm)
    r["by"] = pd.to_numeric(r.birth_year, errors="coerce")
    return r


def people(ids) -> pd.DataFrame:
    """statsapi people (birth date, birth country, full name), cached."""
    from .statsapi_pbp import _get
    f = raw("statsapi_people", "kbo_people.json")
    cache = json.loads(f.read_text()) if f.exists() else {}
    need = sorted({int(i) for i in ids} - {int(k) for k in cache})
    for i in range(0, len(need), 100):
        d = _get("https://statsapi.mlb.com/api/v1/people", {"personIds": ",".join(map(str, need[i:i + 100]))})
        got = {p["id"]: {"full_name": p.get("fullName"), "birth_date": p.get("birthDate"),
                         "birth_country": p.get("birthCountry")} for p in d.get("people", [])}
        for j in need[i:i + 100]:
            cache[str(j)] = got.get(j, {"full_name": None, "birth_date": None, "birth_country": None})
    if need:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(cache, indent=0, ensure_ascii=False))
    out = pd.DataFrame.from_dict(cache, orient="index")
    out.index = out.index.astype(int)
    return out.reindex([int(i) for i in ids])


def _age_on(birth: str | None, season: int) -> float:
    """Age on 30 June of the season (FanGraphs' convention; checked on FanGraphs-key matches)."""
    if not birth:
        return np.nan
    b = pd.Timestamp(birth)
    return season - b.year - (1 if (b.month, b.day) > (6, 30) else 0)


@lru_cache(maxsize=1)
def links() -> pd.DataFrame:
    """fg_id -> MLBAM id with the method used ('fg key', 'name + birth year', or why not linked)."""
    rows = pd.concat([fg_all("pit"), fg_all("bat")], ignore_index=True)
    per = rows.groupby("fg_id").agg(name=("PlayerName", "last"), hangul=("hangul", "last"),
                                    seasons=("season", lambda s: sorted(set(s))))
    ages = rows.dropna(subset=["age"]).groupby("fg_id").apply(lambda g: list(zip(g.season, g.age)), include_groups=False)
    per["ages"] = ages.reindex(per.index)
    reg = register()
    key = dict(zip(reg.dropna(subset=["key_fangraphs"]).key_fangraphs, reg.dropna(subset=["key_fangraphs"]).mlbam))
    by_name = reg.groupby("nk").groups
    cand = {}
    for fid, r in per.iterrows():
        if fid in key:
            cand[fid] = ("fg key", [key[fid]])
            continue
        if fid in ALIASES:
            cand[fid] = ("alias", [ALIASES[fid]])
            continue
        c = reg.loc[by_name[_norm(r["name"])]] if _norm(r["name"]) in by_name else reg.iloc[0:0]
        if isinstance(r.ages, list) and r.ages:
            s, a = r.ages[-1]
            c = c[c.by.between(s - a - 1, s - a)]
        else:
            c = c.iloc[0:0]
        cand[fid] = ("name + birth year", c.mlbam.drop_duplicates().tolist())
    ppl = people({i for _, ids in cand.values() for i in ids})
    out = []
    for fid, (how, ids) in cand.items():
        ages = per.at[fid, "ages"] if isinstance(per.at[fid, "ages"], list) else []
        ok = [i for i in ids if how == "fg key" or
              (ages and all(abs(_age_on(ppl.at[i, "birth_date"], s) - a) <= 0 for s, a in ages))]
        if len(ok) == 1:
            m = ok[0]
            out.append({"fg_id": fid, "mlbam_id": m, "link": how, **ppl.loc[m].to_dict()})
        else:
            out.append({"fg_id": fid, "mlbam_id": np.nan,
                        "link": "no register match" if not ids else
                        "birth date does not match" if not ok else f"ambiguous ({len(ok)})"})
    d = pd.DataFrame(out).set_index("fg_id").join(per[["name", "hangul", "seasons"]])
    dup = d.mlbam_id.notna() & d.mlbam_id.duplicated(keep=False)
    d.loc[dup, ["mlbam_id", "link"]] = [np.nan, "two FanGraphs ids for one MLBAM id"]
    d.loc[dup, ["birth_country", "birth_date", "full_name"]] = None
    d["foreign"] = np.where(d.birth_country.isna(), None, ~d.birth_country.isin(KR))
    return d


def pid(fid: str) -> int:
    """Marcel player id: MLBAM when linked, else a negative id from the FanGraphs id."""
    m = links().mlbam_id.get(fid)
    if pd.notna(m):
        return int(m)
    n = int(re.sub(r"\D", "", fid) or 0)
    return -(100_000_000 + n) if fid.startswith("sa") else -n


# ------------------------------------------------------------------ KBO stints for Marcel
@lru_cache(maxsize=None)
def stints(group: str, season: int, neutral: bool = False) -> pd.DataFrame:
    """KBO season totals in the amendment-15 Marcel column names (one league, PF = 1)."""
    y0, y1 = cfg()["kbo_seasons"]
    if not y0 <= season <= y1:
        return pd.DataFrame()
    d = fg("pit" if group == "pitching" else "bat", season)
    num = lambda c: pd.to_numeric(d[c], errors="coerce").fillna(0)
    s = pd.DataFrame({"player_id": d.fg_id.map(pid), "fg_id": d.fg_id, "team_id": 0, "team_league": "KBO",
                      "season": season})
    if group == "pitching":
        outs = d.IP.map(lambda v: ip_to_outs(v) if pd.notna(v) else 0)
        s = s.assign(pitching_BF=num("TBF"), pitching_SO=num("SO"), pitching_BB=num("BB"), pitching_IBB=num("IBB"),
                     pitching_HR=num("HR"), pitching_HBP=num("HBP"), pitching_ER=num("ER"), pitching_GS=num("GS"),
                     outs=outs, pitching_IP=outs / 3)
        s = s[s.pitching_BF > 0]
    else:
        s = s.assign(batting_PA=num("PA"), batting_AB=num("AB"), batting_H=num("H"), batting_2B=num("2B"),
                     batting_3B=num("3B"), batting_HR=num("HR"), batting_BB=num("BB"), batting_IBB=num("IBB"),
                     batting_SO=num("SO"), batting_HBP=num("HBP"), batting_SF=num("SF"))
        s["batting_TB"] = s.batting_H + s.batting_2B + 2 * s.batting_3B + 3 * s.batting_HR
        s = s[s.batting_PA > 0]
    return s.reset_index(drop=True)


@lru_cache(maxsize=1)
def _register_dob() -> pd.DataFrame:
    """Every register person with a full birth date (MLBAM id or not), by normalised name."""
    ext = (ROOT / config()["paths"]["external"]).resolve()
    fs = glob.glob(str(ext / "register" / "data" / "people-*.csv"))
    r = pd.concat([pd.read_csv(f, usecols=["name_first", "name_last", "birth_year", "birth_month", "birth_day"],
                               dtype=str) for f in fs]).dropna().reset_index(drop=True)
    r["nk"] = (r.name_first + r.name_last).map(_norm)
    r["dob"] = r.birth_year + "-" + r.birth_month.str.zfill(2) + "-" + r.birth_day.str.zfill(2)
    return r[["nk", "dob"]].drop_duplicates()


@lru_cache(maxsize=1)
def birth() -> pd.Series:
    """Birth year by Marcel player id: statsapi birth date when linked; else a unique register birth
    date (any person, name + every FanGraphs June-30 age); else season - age - 0.5 (FanGraphs' June-30
    age leaves the year one of two; the midpoint)."""
    L = links()
    rows = pd.concat([fg_all("pit"), fg_all("bat")], ignore_index=True).dropna(subset=["age"])
    ages = rows.groupby("fg_id").apply(lambda g: list(zip(g.season, g.age)), include_groups=False)
    last = rows.sort_values("season").drop_duplicates("fg_id", keep="last").set_index("fg_id")
    dob = _register_dob().groupby("nk").dob.apply(list)
    names = pd.concat([fg_all("pit"), fg_all("bat")]).drop_duplicates("fg_id").set_index("fg_id").PlayerName
    by = {}
    for fid, a in ages.items():
        if pd.notna(L.mlbam_id.get(fid)):
            continue
        c = [d for d in dob.get(_norm(names[fid]), []) if all(_age_on(d, s_) == x for s_, x in a)]
        by[pid(fid)] = int(c[0][:4]) if len({d[:4] for d in c}) == 1 else last.season[fid] - last.age[fid] - 0.5
    b = pd.Series(by, dtype=float)
    lk = L[L.mlbam_id.notna() & L.birth_date.notna()]
    sa = pd.Series(pd.to_datetime(lk.birth_date).dt.year.to_numpy(), index=lk.mlbam_id.astype(int).to_numpy())
    b = pd.concat([sa, b])                        # statsapi first; FanGraphs age only when unlinked
    return b[~b.index.duplicated(keep="first")]


def kbo_marcel(role: str, year: int) -> pd.DataFrame:
    """KBO Marcel for season `year` from KBO seasons year-1..year-3 (amendment 15 rules, PF = 1)."""
    return marcel.project(role, year, seasons=[year - 1, year - 2, year - 3], stints_fn=stints, birth=birth())


def league(group: str, season: int) -> dict:
    s = marcel._prep(stints(group, season), "hitter" if group == "hitting" else "pitcher")
    if group == "pitching":
        return {"k": s.pitching_SO.sum() / s.pitching_BF.sum(), "bb": s.pitching_BB.sum() / s.pitching_BF.sum()}
    w = marcel._stats("hitter")["woba"][0]
    return {"woba": w(s).sum() / s._woba_den.sum()}


def season_lines(group: str, season: int) -> pd.DataFrame:
    """Single-season KBO rates, centred on that season's KBO average, by Marcel player id."""
    s = stints(group, season)
    L = league(group, season)
    if group == "pitching":
        k, bb = s.pitching_SO / s.pitching_BF, s.pitching_BB / s.pitching_BF
        return pd.DataFrame({"player_id": s.player_id, "fg_id": s.fg_id, "bf": s.pitching_BF, "gs": s.pitching_GS,
                             "ip": s.pitching_IP, "k": k, "bb": bb, "k_c": k - L["k"], "bb_c": bb - L["bb"],
                             "kbb_c": (k - L["k"]) - (bb - L["bb"])})
    p = marcel._prep(s, "hitter")
    wo = marcel._stats("hitter")["woba"][0](p) / p._woba_den
    return pd.DataFrame({"player_id": s.player_id, "fg_id": s.fg_id, "pa": s.batting_PA, "woba": wo,
                         "woba_c": wo - L["woba"], "k_c": s.batting_SO / s.batting_PA - (s.batting_SO.sum() / s.batting_PA.sum()),
                         "bb_c": s.batting_BB / s.batting_PA - (s.batting_BB.sum() / s.batting_PA.sum())})


# ------------------------------------------------------------------ coverage check (B-R KBO rosters)
def br_roster(year: int) -> pd.DataFrame:
    """Baseball-Reference KBO team rosters (birthplace) for one season: used only to list foreign
    players the FanGraphs -> MLBAM link misses (cached like the census pages, <= 20 requests/min)."""
    c = cpbl_census
    t = c._clean(c._get(f"/register/league.cgi?year={year}", f"leagues_{year}"))
    ids = {i for i, n in re.findall(r'href="/register/league\.cgi\?id=(\w+)">([^<]+)</a>', t) if n.strip() == "Korean"}
    assert len(ids) == 1, (year, ids)
    lid = ids.pop()
    t = c._clean(c._get(f"/register/league.cgi?id={lid}", f"league_{year}_{lid}"))
    out = []
    for tid in sorted(set(re.findall(r"/register/team\.cgi\?id=(\w+)", t))):
        p = c._clean(c._get(f"/register/team.cgi?id={tid}", f"team_{year}_{tid}"))
        r = c._table(p, "standard_roster")
        if len(r):
            out.append(r[r.player.ne("Name")].assign(team_id=tid))
    r = pd.concat(out, ignore_index=True)
    r["country"] = r.birth_city.astype(str).str.extract(r",\s*([A-Za-z]{2})\s*$")[0].str.upper()
    return r


# ------------------------------------------------------------------ statistics
def _rng(k: int = 0):
    return np.random.default_rng(cfg()["bootstrap"]["seed"] + k)


def _q() -> float:
    return (1 - cfg()["bootstrap"]["ci"]) / 2


def _groups(d: pd.DataFrame, cluster: str | None) -> list[np.ndarray]:
    keys = d[cluster].to_numpy() if cluster else np.arange(len(d))
    return [np.flatnonzero(keys == k) for k in pd.unique(keys)]


def _draws(d: pd.DataFrame, cluster: str | None, rng):
    g = _groups(d, cluster)
    for _ in range(cfg()["bootstrap"]["draws"]):
        yield np.concatenate([g[i] for i in rng.integers(0, len(g), len(g))])


def _verdict(n: int, lo: float, hi: float) -> str:
    if n < cfg()["min_n"]:
        return "descriptive only (n below 15)"
    return "supports" if lo > 0 else "contradicts" if hi < 0 else "inconclusive"


def rho_test(label: str, d: pd.DataFrame, x: str, y: str, rng, cluster: str | None = None) -> dict:
    d = d.dropna(subset=[x, y])
    n = d[cluster].nunique() if cluster else len(d)
    if len(d) < 3:
        return {"test": label, "n": n, "rows": len(d), "rho": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "verdict": "too few"}
    est = backtest.spearman(d[x], d[y])
    v = [backtest.spearman(d[x].to_numpy()[i], d[y].to_numpy()[i]) for i in _draws(d, cluster, rng)]
    v = np.array([a for a in v if not np.isnan(a)])
    if np.isnan(est) or not len(v):                 # a constant input: no rank correlation defined
        return {"test": label, "n": n, "rows": len(d), "rho": est, "ci_lo": np.nan, "ci_hi": np.nan, "verdict": "undefined"}
    lo, hi = float(np.quantile(v, _q())), float(np.quantile(v, 1 - _q()))
    return {"test": label, "n": n, "rows": len(d), "rho": est, "ci_lo": lo, "ci_hi": hi, "verdict": _verdict(n, lo, hi)}


def mean_ci(label: str, v: pd.Series, rng, cluster: pd.Series | None = None) -> dict:
    d = pd.DataFrame({"v": v, "c": cluster if cluster is not None else np.arange(len(v))}).dropna(subset=["v"])
    v = d.v.to_numpy()
    if len(v) < 2:
        return {"metric": label, "n": len(v), "mean": np.nan, "ci_lo": np.nan, "ci_hi": np.nan}
    m = [v[i].mean() for i in _draws(d, "c", rng)]
    return {"metric": label, "n": len(v), "mean": float(v.mean()), "ci_lo": float(np.quantile(m, _q())),
            "ci_hi": float(np.quantile(m, 1 - _q()))}


def ols(d: pd.DataFrame, x: str, y: str) -> tuple[float, float]:
    """(slope, intercept)."""
    d = d.dropna(subset=[x, y])
    k, a = np.polyfit(d[x].astype(float), d[y].astype(float), 1)
    return float(k), float(a)


def ols_ci(label: str, d: pd.DataFrame, x: str, y: str, rng, cluster: str | None = None) -> dict:
    d = d.dropna(subset=[x, y])
    if len(d) < 3 or np.ptp(d[x].astype(float)) == 0:
        return {"line": label, "n": len(d)}
    k, a = ols(d, x, y)
    bs = [ols(d.iloc[i], x, y) for i in _draws(d, cluster, rng) if np.ptp(d[x].to_numpy()[i]) > 0]
    ks = np.array([b[0] for b in bs])
    if not len(ks):
        return {"line": label, "n": len(d)}
    return {"line": label, "n": d[cluster].nunique() if cluster else len(d), "slope": k,
            "slope_lo": float(np.quantile(ks, _q())), "slope_hi": float(np.quantile(ks, 1 - _q())),
            "intercept": a, "x_min": float(d[x].min()), "x_max": float(d[x].max())}


def _mae_diff(f: pd.DataFrame, rng, cluster: str | None = None) -> dict:
    e0, e1 = (f.actual - f.m0).abs().to_numpy(), (f.actual - f.m1).abs().to_numpy()
    diffs = [e1[i].mean() - e0[i].mean() for i in _draws(f, cluster, rng)]
    n_fit = int(f.loc[f.fitted, cluster].nunique()) if cluster else int(f.fitted.sum())   # players, not rows
    lo, hi = float(np.quantile(diffs, _q())), float(np.quantile(diffs, 1 - _q()))
    ok = n_fit >= cfg()["min_n"] and hi < 0
    return {"n_held_out": len(f), "n_fitted": n_fit, "n_fitted_rows": int(f.fitted.sum()), "mae_mean": float(e0.mean()), "mae_line": float(e1.mean()),
            "diff": float(e1.mean() - e0.mean()), "ci": [lo, hi],
            "verdict": "predictive improvement" if ok else "not established"}


def loo(d: pd.DataFrame, x: str, y: str, rng, cluster: str | None = None) -> dict:
    """Leave-one-player-out: OLS line vs the training mean."""
    d = d.dropna(subset=[x, y]).reset_index(drop=True)
    rows = []
    for g in _groups(d, cluster):
        tr, te = d.drop(index=g), d.iloc[g]
        if len(tr) < 3 or np.ptp(tr[x]) == 0:
            continue
        k, a = ols(tr, x, y)
        rows += [{"key": te[cluster].iat[0] if cluster else i, "actual": v, "m0": tr[y].mean(), "m1": a + k * xv,
                  "fitted": True} for i, xv, v in zip(te.index, te[x], te[y])]
    f = pd.DataFrame(rows)
    return _mae_diff(f, rng, "key" if cluster else None) if len(f) else {"n_held_out": 0}


def rolling(d: pd.DataFrame, x: str, y: str, rng, time: str = "dest") -> dict:
    """Train on earlier destination years only (>= min_train pitchers for a line, else the mean)."""
    d = d.dropna(subset=[x, y])
    rows = []
    for t in sorted(d[time].unique()):
        tr, te = d[d[time] < t], d[d[time] == t]
        if not len(tr):
            continue
        fit = len(tr) >= cfg()["t1"]["min_train"] and np.ptp(tr[x]) > 0
        k, a = ols(tr, x, y) if fit else (0.0, tr[y].mean())
        rows += [{"dest": t, "actual": v, "m0": tr[y].mean(), "m1": a + k * xv, "fitted": fit}
                 for xv, v in zip(te[x], te[y])]
    f = pd.DataFrame(rows)
    return _mae_diff(f, rng) if len(f) else {"n_held_out": 0}


# ------------------------------------------------------------------ T1 KBO -> CPBL (pitchers)
def _cpbl_lines(ctx: dict) -> pd.DataFrame:
    rows = []
    for y, t in ctx["tabs"].items():
        p = cpbl_census._num(t["pitching"]).groupby("bbref_id")[["batters_faced", "SO", "BB"]].sum()
        lg = ctx["leagues"][y]
        k, bb = p.SO / p.batters_faced, p.BB / p.batters_faced
        rows.append(pd.DataFrame({"bbref_id": p.index, "season": y, "cpbl_bf": p.batters_faced.to_numpy(),
                                  "cpbl_k": k.to_numpy(), "cpbl_bb": bb.to_numpy(),
                                  "cpbl_k_c": (k - lg["k_pct"]).to_numpy(), "cpbl_bb_c": (bb - lg["bb_pct"]).to_numpy()}))
    d = pd.concat(rows, ignore_index=True)
    d["cpbl_kbb_c"] = d.cpbl_k_c - d.cpbl_bb_c
    return d


def _kbo_lines() -> pd.DataFrame:
    y0, y1 = cfg()["kbo_seasons"]
    d = pd.concat([season_lines("pitching", y).assign(season=y) for y in range(y0, y1 + 1)], ignore_index=True)
    return d.rename(columns={c: f"kbo_{c}" for c in d.columns if c not in ("player_id", "fg_id", "season")})


def transitions(cen: pd.DataFrame, ctx: dict) -> dict[str, pd.DataFrame]:
    """Forward (KBO Y -> CPBL Y+1) and reverse (CPBL Y -> KBO Y+1) pitcher transitions, >= min BF both."""
    c = cfg()["t1"]
    imp = cen[cen["import"] & cen.mlbam_id.notna()][["bbref_id", "name", "mlbam_id"]].copy()
    imp["player_id"] = imp.mlbam_id.astype(float).astype(int)
    cp = _cpbl_lines(ctx).merge(imp, on="bbref_id")
    kb = _kbo_lines()
    m = {}
    for D in range(c["destinations"][0], c["destinations"][1] + 1):
        mm = kbo_marcel("pitcher", D).set_index("player_id")
        m[D] = pd.DataFrame({"kbo_m_kbb_c": mm.score_c, "kbo_m_k_c": mm.k - mm.lg_k, "kbo_m_bb_c": mm.bb - mm.lg_bb})
    fwd = cp.rename(columns={"season": "dest"}).assign(src=lambda d: d.dest - 1)
    fwd = fwd.merge(kb.rename(columns={"season": "src"}), on=["player_id", "src"])
    fwd = fwd[fwd.dest.between(*c["destinations"]) & (fwd.cpbl_bf >= c["min_bf"]) & (fwd.kbo_bf >= c["min_bf"])]
    fwd = pd.concat([g.join(m[int(D)], on="player_id") for D, g in fwd.groupby("dest")], ignore_index=True)
    fwd = fwd.sort_values(["player_id", "dest"]).reset_index(drop=True)
    fwd["first"] = ~fwd.player_id.duplicated()
    # reverse: CPBL seasons before the census range too (B-R team pages from the first KBO season)
    early = {y: cpbl_census.team_tables(y) for y in range(cfg()["kbo_seasons"][0], min(ctx["tabs"]))}
    ctx_all = {"tabs": {**early, **ctx["tabs"]},
               "leagues": {**{y: cpbl_census._cpbl_league(t) for y, t in early.items()}, **ctx["leagues"]}}
    # imports of those early seasons: the census covers 2018-26 only, so the register page's birthplace
    # decides for early-roster players who reappear in KBO (only those can form a reverse pair)
    ro = pd.concat([t["roster"] for t in early.values()], ignore_index=True) if early else pd.DataFrame()
    imp_all = imp
    if len(ro):
        ro = ro.assign(mlbam_id=ro.bbref_id.map(cpbl_census._register())).dropna(subset=["bbref_id", "mlbam_id"])
        ro = ro[ro.mlbam_id.astype(int).isin(set(kb.player_id)) & ~ro.bbref_id.isin(imp.bbref_id)]
        extra = []
        for pid, r in ro.drop_duplicates("bbref_id").set_index("bbref_id").iterrows():
            ctry = cpbl_census._country(cpbl_census.player_rows(pid)["born"])
            if ctry and ctry != "TW":
                extra.append({"bbref_id": pid, "name": r.player.rstrip("*#").strip(), "mlbam_id": r.mlbam_id,
                              "player_id": int(r.mlbam_id)})
        imp_all = pd.concat([imp, pd.DataFrame(extra, columns=imp.columns)], ignore_index=True)
    cp_all = _cpbl_lines(ctx_all).merge(imp_all, on="bbref_id")
    rev = cp_all.rename(columns={"season": "src"}).assign(dest=lambda d: d.src + 1)
    rev = rev.merge(kb.rename(columns={"season": "dest"}), on=["player_id", "dest"])
    rev = rev[(rev.cpbl_bf >= c["min_bf"]) & (rev.kbo_bf >= c["min_bf"])]
    return {"forward": fwd, "reverse": rev.reset_index(drop=True)}


def t1_coverage(cen: pd.DataFrame, ctx: dict, fwd: pd.DataFrame) -> dict:
    """B-R register KBO lines of imports: forward pairs FanGraphs misses, and BF agreement."""
    c = cfg()["t1"]
    cp = _cpbl_lines(ctx)
    rows, chk = [], []
    kb = _kbo_lines()
    for r in cen[cen["import"]].itertuples(index=False):
        h = cpbl_census.history(cpbl_census.player_rows(r.bbref_id))
        if "batters_faced" not in h:
            continue
        k = h[h.cat.eq("KBO")].set_index("season").batters_faced
        mine = cp[cp.bbref_id.eq(r.bbref_id)].set_index("season").cpbl_bf
        for D, bf in mine.items():
            if c["destinations"][0] <= D <= c["destinations"][1] and bf >= c["min_bf"] and k.get(D - 1, 0) >= c["min_bf"]:
                rows.append({"name": r.name, "bbref_id": r.bbref_id, "mlbam_id": r.mlbam_id, "dest": D,
                             "in_sample": bool(((fwd.bbref_id == r.bbref_id) & (fwd.dest == D)).any())})
        if pd.notna(r.mlbam_id):
            fk = kb[kb.player_id.eq(int(r.mlbam_id))].set_index("season").kbo_bf
            chk += [{"season": s, "bf_br": v, "bf_fg": fk.get(s)} for s, v in k.items() if s in fk.index]
    br = pd.DataFrame(rows)
    ck = pd.DataFrame(chk)
    return {"br_pairs": int(len(br)), "in_sample": int(br.in_sample.sum()) if len(br) else 0,
            "missed": br[~br.in_sample].to_dict(orient="records") if len(br) else [],
            "bf_check_n": int(len(ck)), "bf_mean_abs_diff": float((ck.bf_br - ck.bf_fg).abs().mean()) if len(ck) else None}


T1_TESTS = [("K1 KBO Marcel K-BB% vs CPBL K-BB% (primary)", "kbo_m_kbb_c", "cpbl_kbb_c"),
            ("K1b KBO Marcel K% vs CPBL K%", "kbo_m_k_c", "cpbl_k_c"),
            ("K1c KBO Marcel BB% vs CPBL BB%", "kbo_m_bb_c", "cpbl_bb_c")]


def t1(tr: dict) -> dict:
    fwd, rev = tr["forward"], tr["reverse"]
    P = fwd[fwd["first"]]
    rng = _rng()
    out = {"n_pairs": len(fwd), "n_pitchers": int(fwd.player_id.nunique()), "by_dest": P.dest.value_counts().sort_index().to_dict(),
           "k1": [rho_test(l, P, x, y, rng) for l, x, y in T1_TESTS],
           "k2_single_season": [rho_test(f"K2 KBO season {lab} vs CPBL {lab}", P, f"kbo_{k}_c", f"cpbl_{k}_c", rng)
                                for lab, k in (("K-BB%", "kbb"), ("K%", "k"), ("BB%", "bb"))],
           "k2_change": [mean_ci(f"CPBL - KBO {lab}", P[a] - P[b], rng) for lab, a, b in
                         (("centred K-BB%", "cpbl_kbb_c", "kbo_kbb_c"), ("centred K%", "cpbl_k_c", "kbo_k_c"),
                          ("centred BB%", "cpbl_bb_c", "kbo_bb_c"), ("raw K%", "cpbl_k", "kbo_k"),
                          ("raw BB%", "cpbl_bb", "kbo_bb"))]
                         + [mean_ci("CPBL - KBO raw K-BB%", (P.cpbl_k - P.cpbl_bb) - (P.kbo_k - P.kbo_bb), rng)],
           "k3": {comp: {"line": ols_ci(comp, P, x, y, rng), "loo": loo(P, x, y, rng), "rolling": rolling(P, x, y, rng)}
                  for comp, x, y in (("k-bb", "kbo_m_kbb_c", "cpbl_kbb_c"), ("k", "kbo_m_k_c", "cpbl_k_c"),
                                     ("bb", "kbo_m_bb_c", "cpbl_bb_c"))}}
    c = cfg()["t1"]
    rng2 = _rng(1)
    s200 = fwd[(fwd.cpbl_bf >= c["sensitivity_min_bf"]) & (fwd.kbo_bf >= c["sensitivity_min_bf"])]
    s200 = s200[~s200.player_id.duplicated()]
    dlate = fwd[fwd.dest >= c["sensitivity_from"]]          # filter, then each pitcher's first such transition
    dlate = dlate[~dlate.player_id.duplicated()]
    out["sensitivity"] = []
    for l, x, y in T1_TESTS:
        k = l.split(" ")[0]
        lab = l.split(" vs ")[1].replace(" (primary)", "")
        out["sensitivity"] += [rho_test(f"{k} {lab}: >= 200 BF both sides", s200, x, y, rng2),
                               rho_test(f"{k} {lab}: destinations {c['sensitivity_from']}+", dlate, x, y, rng2),
                               rho_test(f"{k} {lab}: all forward pairs (clustered by pitcher)", fwd, x, y, rng2,
                                        cluster="player_id")]
    out["reverse"] = rev[["name", "src", "dest", "cpbl_bf", "kbo_bf", "cpbl_kbb_c", "kbo_kbb_c"]].to_dict(orient="records")
    return out


# ------------------------------------------------------------------ T2 AAA -> KBO
def t2_cohort(role: str) -> pd.DataFrame:
    c = cfg()["t2"]
    group = "pitching" if role == "pitcher" else "hitting"
    expo, kexpo, mn = (("pitching_BF", "bf", c["min_bf"]) if role == "pitcher" else ("batting_PA", "pa", c["min_pa"]))
    rows = []
    for D in range(c["destinations"][0], c["destinations"][1] + 1):
        a = marcel.stints(group, D - 1)
        if a.empty:
            continue                                   # no AAA 2020: no 2019 -> 2021 bridge
        aaa = a.groupby("player_id")[expo].sum()
        k = season_lines(group, D)
        k = k[(k.player_id > 0) & (k[kexpo] >= mn)]
        k = k[k.player_id.map(aaa).fillna(0) >= mn]
        if k.empty:
            continue
        m = marcel.project(role, D, seasons=[D - 1, D - 2, D - 3]).set_index("player_id")
        g = k.assign(dest=D, aaa_exposure=k.player_id.map(aaa).to_numpy(), aaa_m=k.player_id.map(m.score_c).to_numpy())
        if role == "pitcher":
            g["aaa_m_k_c"] = k.player_id.map(m.k - m.lg_k).to_numpy()
            g["aaa_m_bb_c"] = k.player_id.map(m.bb - m.lg_bb).to_numpy()
        rows.append(g)
    d = pd.concat(rows, ignore_index=True)
    d["name"] = d.fg_id.map(links().name)
    d["y"] = d.kbb_c if role == "pitcher" else d.woba_c
    return d


def t2(P: pd.DataFrame, H: pd.DataFrame) -> dict:
    out = {}
    for role, d in (("pitchers", P), ("hitters", H)):
        rng = _rng(2 if role == "pitchers" else 3)
        out[role] = {"rows": len(d), "players": int(d.player_id.nunique()),
                     "by_dest": d.dest.value_counts().sort_index().to_dict(),
                     "rho": rho_test("AAA Marcel vs KBO next season (clustered by player)", d, "aaa_m", "y", rng, "player_id"),
                     "line": ols_ci("KBO_c = a + b * AAA Marcel_c", d, "aaa_m", "y", rng, "player_id"),
                     "loo": loo(d, "aaa_m", "y", rng, "player_id"),
                     "mean_change": mean_ci("KBO_c - AAA Marcel_c", d.y - d.aaa_m, rng, d.player_id)}
        if role == "pitchers":
            out[role]["components"] = [rho_test(f"AAA Marcel {k} vs KBO {k}", d, f"aaa_m_{k}_c", f"{k}_c", rng, "player_id")
                                       for k in ("k", "bb")]
    return out


# ------------------------------------------------------------------ candidates and comparison
def _availability() -> pd.DataFrame:
    f = ROOT / "data" / "reference" / "kbo_availability_2026.csv"
    if not f.exists():
        return pd.DataFrame(columns=["fg_id", "availability", "evidence_date", "source"]).set_index("fg_id")
    return pd.read_csv(f, dtype={"fg_id": str}).set_index("fg_id")


def _unlinked_foreign() -> pd.DataFrame:
    f = ROOT / "data" / "reference" / "kbo_unlinked_foreign_2026.csv"
    return pd.read_csv(f, dtype={"fg_id": str}) if f.exists() else pd.DataFrame(columns=["fg_id", "name", "note"])


def candidates() -> dict[str, pd.DataFrame]:
    c = cfg()["candidates"]
    Y, F = c["season"], cfg()["forecast_season"]
    L, av = links(), _availability()
    out = {}
    for role, group, ex, mn in (("pitcher", "pitching", "bf", c["min_bf"]), ("hitter", "hitting", "pa", c["min_pa"])):
        s = season_lines(group, Y).join(L[["name", "hangul", "foreign", "birth_country", "link", "mlbam_id"]], on="fg_id")
        q = s[(s[ex] >= mn) & s.foreign.eq(True) & s.mlbam_id.notna() & ~s.fg_id.isin(DOMESTIC)].copy()
        m = kbo_marcel(role, F).set_index("player_id")
        q["kbo_m"] = q.player_id.map(m.score_c)
        q["kbo_m_share_data"] = q.player_id.map(m[f"share_data_{'k' if role == 'pitcher' else 'woba'}"])
        q["age_2027"] = q.player_id.map(m.age)
        if role == "pitcher":
            q["kbo_m_k_c"] = q.player_id.map(m.k - m.lg_k)
            q["kbo_m_bb_c"] = q.player_id.map(m.bb - m.lg_bb)
            prev = season_lines(group, Y - 1).set_index("fg_id")
            st = c["starter"]
            gs2 = q.gs + q.fg_id.map(prev.gs).fillna(0)
            ip2 = q.ip + q.fg_id.map(prev.ip).fillna(0)
            q["starter"] = (gs2 >= st["min_gs_2y"]) & (ip2 >= st["min_ip_2y"]) & (q.ip >= st["min_ip_last"])
            q["gs_2y"], q["ip_2y"] = gs2, ip2
        stat = "kbb_c" if role == "pitcher" else "woba_c"
        q["watchlist"] = q[stat] < q[stat].median()           # unweighted median, ties not flagged
        q["availability"] = q.fg_id.map(av.availability).fillna("unknown") if len(av) else "unknown"
        for k in ("evidence_date", "source"):
            q[k] = q.fg_id.map(av[k]) if len(av) else None
        out[role] = q.sort_values("kbo_m", ascending=False).reset_index(drop=True)
    return out


def _line_draws(d: pd.DataFrame, x: str, y: str, rng, cluster: str | None = None) -> np.ndarray:
    d = d.dropna(subset=[x, y])
    return np.array([ols(d.iloc[i], x, y) for i in _draws(d, cluster, rng) if np.ptp(d[x].to_numpy()[i]) > 0])


def _pct(v: float, pool: np.ndarray) -> float:
    pool = pool[np.isfinite(pool)]
    return float((pool <= v).mean() * 100) if np.isfinite(v) and len(pool) else np.nan


def _joint_line_draws(a: pd.DataFrame, ax: str, ay: str, b: pd.DataFrame, bx: str, by: str, rng):
    """Bootstrap two lines jointly: players (MLBAM id) drawn once over the union of both training
    sets, so a pitcher in both (same CPBL arrival) carries his rows into both lines together."""
    a, b = a.dropna(subset=[ax, ay]), b.dropna(subset=[bx, by])
    ka, kb = a.player_id.astype(int).to_numpy(), b.mlbam_id.astype(int).to_numpy()
    keys = np.unique(np.concatenate([ka, kb]))
    ia = {k: np.flatnonzero(ka == k) for k in keys}
    ib = {k: np.flatnonzero(kb == k) for k in keys}
    da, db = [], []
    for _ in range(cfg()["bootstrap"]["draws"]):
        pick = keys[rng.integers(0, len(keys), len(keys))]
        i = np.concatenate([ia[k] for k in pick])
        j = np.concatenate([ib[k] for k in pick])
        if len(i) < 3 or len(j) < 3 or np.ptp(a[ax].to_numpy()[i]) == 0 or np.ptp(b[bx].to_numpy()[j]) == 0:
            continue
        da.append(ols(a.iloc[i], ax, ay))
        db.append(ols(b.iloc[j], bx, by))
    return np.array(da), np.array(db)


def compare(cand: dict, P1: pd.DataFrame, H2: pd.DataFrame) -> dict:
    """KBO candidates placed among the AAA signable pool (baseline only; the AAA ranking is unchanged)."""
    rng = _rng(4)
    F = cfg()["forecast_season"]
    out = {}
    # pitchers: T1 line (KBO Marcel -> CPBL) vs census V3 line (AAA Marcel -> CPBL)
    cen = pd.read_csv(processed("marcel_v2_census.csv"))
    l_kbo, l_aaa = ols(P1, "kbo_m_kbb_c", "cpbl_kbb_c"), ols(cen, "marcel", "cpbl_kbb_c")
    d_kbo, d_aaa = _joint_line_draws(P1, "kbo_m_kbb_c", "cpbl_kbb_c", cen, "marcel", "cpbl_kbb_c", rng)
    n = len(d_kbo)
    fr = pd.read_csv(processed("pitcher_fit_results.csv"))
    pool = fr[fr.signable.fillna(False).astype(bool)].copy()
    am = marcel.project("pitcher", F).set_index("player_id")
    pool["aaa_m"] = pool.mlbam_id.astype(int).map(am.score_c)
    dropped_p = pool[pool.aaa_m.isna()].name.tolist()
    pool = pool.dropna(subset=["aaa_m"])
    pool["proj_cpbl_kbb_c"] = l_aaa[1] + l_aaa[0] * pool.aaa_m
    xa = pool.aaa_m.to_numpy()
    pa_ = np.array([d_aaa[i, 1] + d_aaa[i, 0] * xa for i in range(n)])
    pool["proj_lo"], pool["proj_hi"] = np.quantile(pa_, _q(), axis=0), np.quantile(pa_, 1 - _q(), axis=0)
    cx = cen.dropna(subset=["marcel", "cpbl_kbb_c"]).marcel
    pool["extrapolated"] = (pool.aaa_m < cx.min()) | (pool.aaa_m > cx.max())
    q = cand["pitcher"].copy()
    q["proj_cpbl_kbb_c"] = l_kbo[1] + l_kbo[0] * q.kbo_m
    lo_x, hi_x = P1.kbo_m_kbb_c.min(), P1.kbo_m_kbb_c.max()
    q["extrapolated"] = (q.kbo_m < lo_x) | (q.kbo_m > hi_x)
    base = pool.proj_cpbl_kbb_c.to_numpy()
    q["pct_among_aaa_signable"] = [_pct(v, base) for v in q.proj_cpbl_kbb_c]
    sims = np.array([[_pct(d_kbo[i, 1] + d_kbo[i, 0] * x, d_aaa[i, 1] + d_aaa[i, 0] * xa) for i in range(n)]
                     for x in q.kbo_m])
    pr = np.array([[d_kbo[i, 1] + d_kbo[i, 0] * x for i in range(n)] for x in q.kbo_m])
    q["proj_lo"], q["proj_hi"] = np.quantile(pr, _q(), axis=1), np.quantile(pr, 1 - _q(), axis=1)
    q["pct_lo"], q["pct_hi"] = np.quantile(sims, _q(), axis=1), np.quantile(sims, 1 - _q(), axis=1)
    cand["pitcher"] = q
    top = pool.sort_values("combined_rank").head(5)
    pool[["name", "mlbam_id", "sign_tier", "combined_rank", "aaa_m", "proj_cpbl_kbb_c", "proj_lo", "proj_hi",
          "extrapolated"]].to_csv(processed("kbo_compare_aaa_pitchers.csv"), index=False)
    out["pitchers"] = {"t1_line": l_kbo, "census_v3_line": l_aaa, "aaa_signable_n": len(pool),
                       "aaa_dropped_no_marcel": dropped_p, "aaa_extrapolated": int(pool.extrapolated.sum()),
                       "joint_draws": n, "t1_x_range": [float(lo_x), float(hi_x)],
                       "census_x_range": [float(cx.min()), float(cx.max())],
                       "aaa_top5": [{"name": r.name, "combined_rank": int(r.combined_rank), "aaa_marcel_c": r.aaa_m,
                                     "proj_cpbl_kbb_c": r.proj_cpbl_kbb_c, "proj_lo": r.proj_lo, "proj_hi": r.proj_hi,
                                     "pct": _pct(r.proj_cpbl_kbb_c, base)}
                                    for r in top.itertuples()]}
    # hitters: T2 hitter line puts AAA signable hitters on the KBO scale
    l_h = ols(H2, "aaa_m", "y")
    d_h = _line_draws(H2, "aaa_m", "y", rng, "player_id")
    hr = pd.read_csv(processed("hitter_fit_results.csv"))
    hp = hr[hr.signable.fillna(False).astype(bool)].copy()
    hm = marcel.project("hitter", F).set_index("player_id")
    hp["aaa_m"] = hp.mlbam_id.astype(int).map(hm.score_c)
    dropped_h = hp[hp.aaa_m.isna()].name.tolist()
    hp = hp.dropna(subset=["aaa_m"])
    hp["proj_kbo_woba_c"] = l_h[1] + l_h[0] * hp.aaa_m
    hx = H2.dropna(subset=["aaa_m", "y"]).aaa_m
    hp["extrapolated"] = (hp.aaa_m < hx.min()) | (hp.aaa_m > hx.max())
    xh = hp.aaa_m.to_numpy()
    q = cand["hitter"].copy()
    base = hp.proj_kbo_woba_c.to_numpy()
    q["pct_among_aaa_signable"] = [_pct(v, base) for v in q.kbo_m]
    sims = np.array([[_pct(x, d_h[i, 1] + d_h[i, 0] * xh) for i in range(len(d_h))] for x in q.kbo_m])
    q["pct_lo"], q["pct_hi"] = np.quantile(sims, _q(), axis=1), np.quantile(sims, 1 - _q(), axis=1)
    cand["hitter"] = q
    rank_col = "marcel_rank_tracked" if "marcel_rank_tracked" in hp else "combined_rank"
    top = hp.sort_values(rank_col).head(5)
    out["hitters"] = {"t2_line": l_h, "aaa_signable_n": len(hp), "aaa_dropped_no_marcel": dropped_h,
                      "aaa_extrapolated": int(hp.extrapolated.sum()), "t2_x_range": [float(hx.min()), float(hx.max())],
                      "aaa_top5": [{"name": r.name, "rank": int(getattr(r, rank_col)), "aaa_marcel_c": r.aaa_m,
                                    "proj_kbo_woba_c": r.proj_kbo_woba_c, "extrapolated": bool(r.extrapolated),
                                    "pct": _pct(r.proj_kbo_woba_c, base)}
                                   for r in top.itertuples()]}
    return out


def aaa_pool_with_kbo() -> pd.DataFrame:
    """Descriptive, not registered: AAA-pool candidates who also pitched / batted in the KBO 2024-26."""
    rows = []
    for role, f, group, ex in (("pitcher", "pitcher_fit_results.csv", "pitching", "bf"),
                               ("hitter", "hitter_fit_results.csv", "hitting", "pa")):
        fr = pd.read_csv(processed(f))
        rank = "combined_rank" if role == "pitcher" else "marcel_rank_tracked"
        for y in (2024, 2025, 2026):
            s = season_lines(group, y)
            s = s[s.player_id > 0].merge(fr[["mlbam_id", "name", "signable", "sign_tier", rank]],
                                         left_on="player_id", right_on="mlbam_id")
            stat = "kbb_c" if role == "pitcher" else "woba_c"
            rows += [{"role": role, "name": r.name, "aaa_rank": getattr(r, rank), "signable": r.signable,
                      "sign_tier": r.sign_tier, "kbo_season": y, "kbo_exposure": getattr(r, ex),
                      "kbo_centred": getattr(r, stat)} for r in s.itertuples()]
    return pd.DataFrame(rows).sort_values(["role", "aaa_rank", "kbo_season"])


# ------------------------------------------------------------------ run
def run() -> dict:
    cen, ctx = cpbl_census.census()
    tr = transitions(cen, ctx)
    P2, H2 = t2_cohort("pitcher"), t2_cohort("hitter")
    cand = candidates()
    P1 = tr["forward"][tr["forward"]["first"]]
    L = links()
    Y = cfg()["candidates"]["season"]
    unl = []
    for kind, ex, mn in (("pit", "TBF", cfg()["candidates"]["min_bf"]), ("bat", "PA", cfg()["candidates"]["min_pa"])):
        f = fg(kind, Y)
        unl += f[(f[ex] >= mn) & f.fg_id.map(L.mlbam_id).isna()].fg_id.tolist()
    hand = _unlinked_foreign()
    out = {"links": L.link.value_counts().to_dict(),
           "kbo_league": {str(y): {**league("pitching", y), **league("hitting", y)}
                          for y in range(cfg()["kbo_seasons"][0], cfg()["kbo_seasons"][1] + 1)},
           "t1": t1(tr), "t1_coverage": t1_coverage(cen, ctx, tr["forward"]), "t2": t2(P2, H2),
           "comparison": compare(cand, P1, H2),
           "unlinked_2026_qualified": len(set(unl)),
           "unlinked_foreign_by_hand": hand[hand.fg_id.isin(unl)].to_dict(orient="records")}
    tr["forward"].to_csv(processed("kbo_t1_forward.csv"), index=False)
    tr["reverse"].to_csv(processed("kbo_t1_reverse.csv"), index=False)
    P2.to_csv(processed("kbo_t2_pitchers.csv"), index=False)
    H2.to_csv(processed("kbo_t2_hitters.csv"), index=False)
    for role, d in cand.items():
        d.to_csv(processed(f"kbo_candidates_{role}s.csv"), index=False)
    pool = aaa_pool_with_kbo()
    pool.to_csv(processed("kbo_aaa_pool_overlap.csv"), index=False)
    processed("kbo_summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float))
    report(out, tr, cand, pool)
    return out


def _f(v, fmt="+.3f"):
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else format(v, fmt)


def _pred_line(name: str, r: dict) -> str:
    if not r.get("n_held_out"):
        return f"- {name}: no held-out predictions."
    return (f"- {name}: held-out {r['n_held_out']} (by a fitted line {r['n_fitted']}), MAE training mean "
            f"{r['mae_mean']:.4f}, MAE line {r['mae_line']:.4f}, difference {r['diff']:+.4f} (90% CI {r['ci'][0]:+.4f} "
            f"to {r['ci'][1]:+.4f}): **{r['verdict']}**.")


def report(out: dict, tr: dict, cand: dict, pool: pd.DataFrame) -> None:
    t, cov, c = out["t1"], out["t1_coverage"], out["comparison"]
    ints = lambda d: d.astype({k: "Int64" for k in ("bf", "pa", "kbo_bf", "cpbl_bf", "age_2027", "kbo_exposure",
                                                     "aaa_rank", "src", "dest") if k in d})
    L = ["# KBO: translation to the CPBL and a 2026 KBO foreign-player reference list (amendment 16; generated by "
         "`python -m src.kbo`)", "",
         "Sources: FanGraphs KBO leaderboards 2015-2026 (cite \"FanGraphs\"); CPBL lines from the Baseball-Reference "
         "census (amendment 12, cite \"Baseball-Reference\"); AAA from the MLB Stats API (amendment 15 Marcel). "
         "Rules were committed before any KBO rate was computed. **The primary picks are unchanged**: KBO players are a "
         "separate reference list (Johnny chose 「另列參考」).", "",
         "## Linking FanGraphs to MLBAM", "",
         "Links by method: " + ", ".join(f"{k} {v}" for k, v in out["links"].items()) + ". Birth country comes from "
         "statsapi. Deviation D1 (decided before any result): the Chadwick register has no FanGraphs key for players "
         "who never reached the majors, so a FanGraphs player without one is linked by a unique name + birth-year "
         "match, accepted only when the statsapi birth date gives exactly FanGraphs' age (on 30 June) in every "
         "season listed (that convention matched 99.5% of FanGraphs-key links). One FanGraphs spelling the register "
         "cannot match was added by hand after the Baseball-Reference coverage check (`ALIASES` in src/kbo.py).", "",
         f"Coverage check against the Baseball-Reference register: {cov['br_pairs']} KBO -> CPBL pairs there, "
         f"{cov['in_sample']} in the sample" + (f"; missed: {', '.join(m['name'] + ' ' + str(m['dest']) for m in cov['missed'])}"
                                                 if cov["missed"] else "") +
         f". KBO BF on both sources for {cov['bf_check_n']} pitcher-seasons: mean absolute difference "
         f"{_f(cov['bf_mean_abs_diff'], '.2f')} BF.", "",
         "## T1 KBO -> CPBL, pitchers", "",
         f"Forward pairs (KBO Y -> CPBL Y+1, >= 100 BF both): {t['n_pairs']} from {t['n_pitchers']} pitchers; "
         f"primary sample = each pitcher's first pair, by destination {t['by_dest']}.", "",
         md_table(pd.DataFrame(t["k1"] + t["k2_single_season"]), ".3f"), "",
         "K2 mean within-pitcher change, CPBL minus KBO (centred = minus that league-season's average; a change in a "
         "centred rate is movement relative to the two leagues' averages, not a league-strength difference):", "",
         md_table(pd.DataFrame(t["k2_change"]), "+.4f"), "",
         "K3 prediction, OLS CPBL_c = a + b x KBO Marcel_c:", ""]
    for comp, r in t["k3"].items():
        ln = r["line"]
        if "slope" in ln:
            L.append(f"- {comp}: slope {ln['slope']:.2f} (90% CI {ln['slope_lo']:.2f} to {ln['slope_hi']:.2f}), "
                     f"intercept {ln['intercept']:+.4f}, n {ln['n']}.")
        L += [_pred_line(f"{comp} leave-one-pitcher-out", r["loo"]), _pred_line(f"{comp} rolling by destination year", r["rolling"])]
    L += ["", "Registered sensitivities (reported, never choose; K1 only, its three outcomes - K2 and K3 not repeated):", "", md_table(pd.DataFrame(t["sensitivity"]), ".3f"), "",
          "CPBL -> KBO (shown separately, not pooled):", "",
          md_table(ints(pd.DataFrame(t["reverse"])), "+.3f") if t["reverse"] else "none", "",
          "Primary sample:", "",
          md_table(ints(tr["forward"][tr["forward"]["first"]][["name", "src", "dest", "kbo_bf", "cpbl_bf", "kbo_m_kbb_c",
                                                               "kbo_kbb_c", "cpbl_kbb_c"]]), "+.3f"), "",
          "## T2 AAA -> KBO (supporting; never inverted)", ""]
    for role in ("pitchers", "hitters"):
        r = out["t2"][role]
        ln = r["line"]
        L += [f"### {role.capitalize()} ({r['rows']} transitions, {r['players']} players; by KBO season {r['by_dest']})", "",
              md_table(pd.DataFrame([r["rho"]] + r.get("components", [])), ".3f"), "",
              f"Line: KBO_c = {ln['intercept']:+.4f} + {ln['slope']:.2f} x AAA Marcel_c (slope 90% CI {ln['slope_lo']:.2f} "
              f"to {ln['slope_hi']:.2f}). Mean KBO_c - AAA Marcel_c {r['mean_change']['mean']:+.4f} (90% CI "
              f"{r['mean_change']['ci_lo']:+.4f} to {r['mean_change']['ci_hi']:+.4f}).", "",
              _pred_line("Leave-one-player-out", r["loo"]),
              "'supports' above means a positive rank association, not a better prediction than the mean.", ""]
    p, h = c["pitchers"], c["hitters"]
    L += ["## 2026 KBO foreign players (reference list; not eligible for the primary pick)", "",
          f"Foreign = MLBAM-linked with a non-Korean statsapi birth country, >= 150 BF / >= 200 PA in KBO 2026. "
          f"Exposure-qualified 2026 players without a link: {out['unlinked_2026_qualified']}; foreign ones among them "
          "(identified by hand from the name, listed in data/reference/kbo_unlinked_foreign_2026.csv, excluded): "
          + (", ".join(u["name"] for u in out["unlinked_foreign_by_hand"]) or "none") + ". Deviation D2 (found in the "
          "availability news check, after the run): " + "; ".join(DOMESTIC.values()) + " - dropped.", "",
          "`watchlist` = below the median of the qualified 2026 KBO foreign players of the same role (pitchers 2026 "
          "centred K-BB%, hitters 2026 centred wOBA): a performance-based non-renewal watchlist, not availability. "
          "`availability` comes from dated KBO / club announcements and news checked by hand on 2026-10-15 "
          "(data/reference/kbo_availability_2026.csv, one source URL per entry; namu.wiki and mykbostats were not "
          "reachable, so a late transaction could be missing). 'confirmed departure or release' here is mostly a "
          "mid-season 2026 release; a released player may have signed elsewhere since (Daz Cameron went back to AAA "
          "Buffalo), so it means 'left his KBO club', not 'unsigned'; 'unknown' means no 2027 decision was found. KBO reserve "
          "lists come out around 30 November, after the competition deadline.", "",
          f"Pitchers: projected CPBL centred K-BB% = T1 line ({p['t1_line'][1]:+.4f} + {p['t1_line'][0]:.2f} x KBO "
          f"Marcel_c, forecast {cfg()['forecast_season']} from KBO 2024-26). Percentile among the {p['aaa_signable_n']} "
          f"AAA signable pitchers, whose projection is the census V3 line ({p['census_v3_line'][1]:+.4f} + "
          f"{p['census_v3_line'][0]:.2f} x AAA Marcel_c). A baseline-only percentile (no Fit, no Stuff+). "
          f"`extrapolated` = KBO Marcel outside the T1 training range ({p['t1_x_range'][0]:+.3f} to {p['t1_x_range'][1]:+.3f}).", ""]
    q = cand["pitcher"]
    cols = ["name", "birth_country", "age_2027", "bf", "kbb_c", "kbo_m", "proj_cpbl_kbb_c", "proj_lo", "proj_hi",
            "pct_among_aaa_signable", "pct_lo", "pct_hi", "extrapolated", "watchlist", "availability"]
    k3 = t["k3"]["k-bb"]["line"]
    loo1 = t["k3"]["k-bb"]["loo"]
    flat = abs(k3["slope"]) < 0.5
    L += [f"Read with care: the T1 point slope is {k3['slope']:.2f}"
          + (", so every KBO pitcher's projection sits near the average of the KBO -> CPBL arrivals and the "
             "percentiles mostly say how that average compares with the AAA line, not how one KBO pitcher compares "
             "with another" if flat else "")
          + f"; slope 90% CI {k3['slope_lo']:.2f} to {k3['slope_hi']:.2f}"
          + (" (its sign is itself uncertain)" if k3["slope_lo"] < 0 < k3["slope_hi"] else "")
          + f"; in leave-one-out (K3) the line's error minus the plain mean's is {_f(loo1.get('diff'), '+.4f')} "
          + ("(worse than the mean)." if (loo1.get("diff") or 0) > 0 else "(not worse than the mean).")
          + " The intervals are fitted-line intervals (both lines bootstrapped jointly by player, the "
          "candidates' Marcel values and the AAA pool held fixed); they leave out each player's own season-to-season "
          f"noise, so a single pitcher's outcome is far less certain. AAA pitchers outside the census line's range: "
          f"{p['aaa_extrapolated']} of {p['aaa_signable_n']}.", ""]
    L += ["Starters (KBO 2025-26 GS >= 10, IP >= 60, 2026 IP >= 20):", "", md_table(ints(q[q.starter][cols]), "+.3f"), "",
          "Relievers / short stints (reference only):", "", md_table(ints(q[~q.starter][cols]), "+.3f"), "",
          "AAA signable top 5 by the current pitcher ranking (combined_rank among all AAA candidates), on the same "
          "projected-CPBL scale:", "",
          md_table(pd.DataFrame(p["aaa_top5"]), "+.3f"), "",
          f"Hitters: no CPBL translation (one KBO -> CPBL hitter pair). Common KBO scale: AAA signable hitters' Marcel "
          f"wOBA through the T2 hitter line ({h['t2_line'][1]:+.4f} + {h['t2_line'][0]:.2f} x AAA Marcel_c) vs the KBO "
          f"candidates' KBO Marcel wOBA (centred). Percentile among the {h['aaa_signable_n']} AAA signable hitters.", ""]
    t2 = out["t2"]["hitters"]
    t2h = t2["line"]
    q = cand["hitter"]
    pc = pd.to_numeric(q.pct_among_aaa_signable, errors="coerce").dropna()
    ext = float(((pc <= 5) | (pc >= 95)).mean()) if len(pc) else float("nan")
    L += [f"Read with care: the T2 hitter point slope is {t2h['slope']:.2f} (90% CI {t2h['slope_lo']:.2f} to "
          f"{t2h['slope_hi']:.2f}); AAA Marcel vs KBO wOBA rank test: {t2['rho']['verdict']}; leave-one-out error minus "
          f"the mean's {_f(t2['loo'].get('diff'), '+.4f')}. "
          + (f"With so flat a line the AAA signable hitters project to almost the same KBO value, and {ext:.0%} of the "
             "KBO hitters' percentiles are at or beyond 5 / 95 (above or below that one value): not a validated "
             "separation of talent. " if abs(t2h["slope"]) < 0.5 else "")
          + f"AAA hitters outside the T2 training range ({h['t2_x_range'][0]:+.3f} to {h['t2_x_range'][1]:+.3f}): "
          f"{h['aaa_extrapolated']} of {h['aaa_signable_n']}.", ""]
    L += [md_table(ints(q[["name", "birth_country", "age_2027", "pa", "woba_c", "kbo_m", "pct_among_aaa_signable", "pct_lo",
                           "pct_hi", "watchlist", "availability"]]), "+.3f"), "",
          "AAA signable top 5 by the current hitter ranking, on the KBO scale:", "",
          md_table(pd.DataFrame(h["aaa_top5"]), "+.3f"), "",
          "## Not registered: AAA-pool candidates who also played in the KBO (descriptive)", "",
          "Seen while linking; reported because it bears on the picks, not used in any score.", "",
          md_table(ints(pool), "+.3f") if len(pool) else "none", ""]
    (path("reports") / "kbo.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    r = run()
    print(json.dumps({k: r[k] for k in ("links", "t1_coverage", "unlinked_2026_qualified")}, indent=1, default=str))
