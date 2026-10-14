"""Amendment 5 — AAA park factors (home park = half of a player's games).

Factors, per AAA club and per event (SO, uBB, HBP, 1B, 2B, 3B, HR), from the 2025-26 AAA
pitch-level data (plate appearances of BOTH teams) and the statsapi schedule (home club of
each game):

  raw PF(club, e)  = rate of e per PA in the club's home games / rate in its road games
  regressed PF     = exp(r_e * log raw PF), r_e = method-of-moments reliability across clubs:
                     share of the between-club variance of log PF not explained by Poisson
                     sampling (1/home count + 1/road count); r_e = 0 if sampling explains it all
  normalised       so the PA-weighted average club factor is 1 for every event

A player's park factor for a club stint is (PF(home club) + 1) / 2: half his games are at
home, the other half spread over the league's other parks (average 1). Counting stats of
that stint are divided by it before any rate, league average or z-score is computed, so
the league adjustment then works on park-neutral numbers. Game logs (bootstrap, recent
form) are adjusted game by game: home games divided by the full home factor, road games
left as they are, which is the same "half home" idea applied to the games actually played.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from .common import config, processed, raw

EVENTS = ["SO", "uBB", "HBP", "1B", "2B", "3B", "HR"]
# which counting columns each event factor adjusts, per table kind
PITCHING_COLS = {"SO": ["pitching_SO"], "uBB": ["pitching_BB"], "HBP": ["pitching_HBP"], "HR": ["pitching_HR"]}
BATTING_COLS = {"SO": ["batting_SO"], "uBB": ["batting_BB"], "HBP": ["batting_HBP"], "2B": ["batting_2B"],
                "3B": ["batting_3B"], "HR": ["batting_HR"]}


def _cfg() -> dict:
    return config().get("park", {})


def enabled() -> bool:
    return bool(_cfg().get("enabled", False))


def home_clubs(season: int, force: bool = False) -> pd.DataFrame:
    """game_pk -> home club id for AAA regular-season games (statsapi schedule)."""
    out = raw("statsapi_season", f"schedule_home_{season}.csv")
    if out.exists() and not force:
        return pd.read_csv(out)
    from .statsapi_pbp import API, _get
    d = _get(f"{API}/v1/schedule", {"sportId": 11, "gameType": "R", "startDate": f"{season}-03-01",
                                     "endDate": f"{season}-10-31"})
    rows = [{"game_pk": g["gamePk"], "season": season, "home_id": g["teams"]["home"]["team"]["id"],
             "away_id": g["teams"]["away"]["team"]["id"], "venue_id": g.get("venue", {}).get("id")}
            for day in d.get("dates", []) for g in day["games"]]
    df = pd.DataFrame(rows).drop_duplicates("game_pk")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def game_events(aaa: pd.DataFrame, seasons) -> pd.DataFrame:
    """Per game: PA and event counts (both teams), with home / away club ids."""
    pa = aaa[aaa.pa_end & aaa.season.isin(seasons)]
    g = pd.crosstab(pa.game_id.astype(str), pa.pa_event).reindex(columns=EVENTS, fill_value=0)
    g["PA"] = pa.groupby(pa.game_id.astype(str)).size()
    sched = pd.concat([home_clubs(s) for s in seasons]).assign(game_id=lambda x: x.game_pk.astype(str))
    return g.join(sched.set_index("game_id")[["home_id", "away_id"]], how="inner")


def estimate(g: pd.DataFrame) -> pd.DataFrame:
    """Regressed, normalised park factors per home club (index) and event (columns)."""
    clubs = sorted(set(g.home_id) | set(g.away_id))
    home = g.groupby("home_id")[EVENTS + ["PA"]].sum().reindex(clubs)
    road = g.groupby("away_id")[EVENTS + ["PA"]].sum().reindex(clubs)
    out = {}
    for e in EVENTS:
        lpf = np.log((home[e] / home.PA) / (road[e] / road.PA))
        samp = 1 / home[e] + 1 / road[e]                 # Poisson variance of the log ratio
        v = lpf.var(ddof=1)
        r = max(0.0, (v - samp.mean()) / v) if v > 0 else 0.0
        pf = np.exp(r * (lpf - np.average(lpf, weights=home.PA)))
        out[e] = pf / np.average(pf, weights=home.PA)      # PA-weighted league average = 1
        out[f"r_{e}"] = r
    return pd.DataFrame(out, index=pd.Index(clubs, name="club_id"))


def factors_for(seasons, aaa: pd.DataFrame) -> pd.DataFrame:
    """Factors estimated from the given seasons only (amendment-6 backtest folds); cached per season set."""
    f = processed(f"park_factors_{'_'.join(map(str, seasons))}.csv")
    if f.exists():
        return pd.read_csv(f, index_col="club_id")
    pf = estimate(game_events(aaa, seasons))
    pf.to_csv(f)
    return pf


@lru_cache(maxsize=1)
def factors() -> pd.DataFrame:
    f = processed("park_factors.csv")
    if f.exists():
        return pd.read_csv(f, index_col="club_id")
    from . import milb_loader
    pf = estimate(game_events(milb_loader.load(), _cfg()["seasons"]))
    pf.to_csv(f)
    return pf


def _divide(df: pd.DataFrame, club: pd.Series, cols: dict, half: bool, home: pd.Series | None = None,
            pf: pd.DataFrame | None = None) -> pd.DataFrame:
    pf = factors() if pf is None else pf
    df = df.copy()
    for e, targets in cols.items():
        f = club.map(pf[e]).fillna(1.0)
        if half:
            f = (f + 1) / 2
        elif home is not None:
            f = f.where(home, 1.0)
        for c in targets:
            if c in df:
                df[c] = df[c] / f.values
    return df


def _batting_totals(df: pd.DataFrame, singles_f: pd.Series) -> pd.DataFrame:
    """Re-derive H and TB from adjusted 2B/3B/HR and the adjusted singles."""
    one = (df["_1B"] / singles_f.values)
    df["batting_H"] = one + df.batting_2B + df.batting_3B + df.batting_HR
    df["batting_TB"] = one + 2 * df.batting_2B + 3 * df.batting_3B + 4 * df.batting_HR
    return df.drop(columns="_1B")


def adjust(df: pd.DataFrame, kind: str, club_col: str = "team_id", home: pd.Series | None = None,
           pf: pd.DataFrame | None = None) -> pd.DataFrame:
    """Park-adjust a per-club table (season stints: half factor; game logs: pass `home`).
    `pf` overrides the default 2025-26 factors (amendment-6 backtest folds use their own seasons)."""
    if not enabled() or df.empty:
        return df
    pf = factors() if pf is None else pf
    club = pd.to_numeric(df[club_col], errors="coerce")
    half = home is None
    if kind == "pitching":
        out = _divide(df, club, PITCHING_COLS, half, home, pf)
        # hits allowed: singles / doubles / triples are not split in pitching tables; HR is
        if "pitching_H" in out:
            f1 = club.map(pf["1B"]).fillna(1.0)
            f1 = (f1 + 1) / 2 if half else f1.where(home, 1.0)
            out["pitching_H"] = (df.pitching_H - df.pitching_HR) / f1.values + out.pitching_HR
        return out
    d = df.copy()
    d["_1B"] = d.batting_H - d.batting_2B - d.batting_3B - d.batting_HR
    d = _divide(d, club, BATTING_COLS, half, home, pf)
    f1 = club.map(pf["1B"]).fillna(1.0)
    f1 = (f1 + 1) / 2 if half else f1.where(home, 1.0)
    return _batting_totals(d, f1)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    print(factors().round(3).sort_values("HR").to_string())
