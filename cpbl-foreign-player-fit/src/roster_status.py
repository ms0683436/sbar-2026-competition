"""Recruitment-feasibility FACTS from the official MLB Stats API (statsapi.mlb.com).

For every screened candidate, as of config.recruitment.as_of:
  on_40man          member of any MLB club's 40-man roster (all 30 clubs queried by date)
  roster_club / roster_status   e.g. "San Francisco Giants" / "Injured 60-Day"
  mlb_2026_g, mlb_2026_pa, mlb_2026_ip   2026 MLB regular-season playing time (does NOT
                    affect eligibility, which only looks at 2024 and 2025)
  last_transaction  latest transaction on or before the as-of date (verbatim)

These are reported facts, never scored, and never a judgement about contract terms,
salary or willingness to play in Taiwan (none of which are public here). A 40-man
player can still be released or sold; "not on a 40-man roster" only means no MLB club
holds a major-league roster spot for him on that date.

Usage:  python -m src.roster_status     (needs processed/screening_log_*.csv)
"""
from __future__ import annotations

import pandas as pd
import requests

from .common import config, processed

API = "https://statsapi.mlb.com/api/v1"


def _get(path: str, params: dict | None = None) -> dict:
    r = requests.get(f"{API}/{path}", params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def forty_man(as_of: str) -> pd.DataFrame:
    season = as_of[:4]
    teams = _get("teams", {"sportId": 1, "season": season})["teams"]
    rows = []
    for t in teams:
        r = _get(f"teams/{t['id']}/roster", {"rosterType": "40Man", "date": as_of})
        for e in r.get("roster", []):
            rows.append({"mlbam_id": e["person"]["id"], "roster_club": t["name"],
                         "roster_status": e["status"]["description"]})
    df = pd.DataFrame(rows)
    df.attrs["clubs"] = len(teams)
    return df


def mlb_line(splits: list[dict]) -> dict | None:
    """One MLB season line from a stats group's splits.

    A traded player gets one split per club plus a combined row without a "team" key;
    use the combined row when present, otherwise add the club stints.
    """
    mlb = [sp for sp in splits if sp.get("sport", {}).get("id") in (1, None)]
    if not mlb:
        return None
    total = [sp for sp in mlb if "team" not in sp]
    if total and len(mlb) > 1:
        return total[0]["stat"]
    if len(mlb) == 1:
        return mlb[0]["stat"]
    line: dict = {"gamesPlayed": 0, "plateAppearances": 0}
    outs = 0
    for sp in mlb:
        v = sp["stat"]
        line["gamesPlayed"] += v.get("gamesPlayed") or 0
        line["plateAppearances"] += v.get("plateAppearances") or 0
        w, _, f = str(v.get("inningsPitched") or "0.0").partition(".")
        outs += int(w) * 3 + int(f or 0)
    line["inningsPitched"] = f"{outs // 3}.{outs % 3}"
    return line


def people(ids: list[int], as_of: str) -> pd.DataFrame:
    season = as_of[:4]
    hyd = f"transactions,stats(group=[hitting,pitching],type=[season],season={season},sportId=1)"
    rows = []
    for i in range(0, len(ids), 50):
        chunk = ids[i:i + 50]
        d = _get("people", {"personIds": ",".join(map(str, chunk)), "hydrate": hyd})
        for p in d.get("people", []):
            g = pa = outs = 0
            for s in p.get("stats", []):
                st = mlb_line(s.get("splits", []))
                if st is None:
                    continue
                g = max(g, st.get("gamesPlayed") or 0)       # two-way players: games, not stints
                if s["group"]["displayName"] == "hitting":
                    pa += st.get("plateAppearances") or 0
                else:
                    ip = str(st.get("inningsPitched") or "0.0")
                    w, _, f = ip.partition(".")
                    outs += int(w) * 3 + int(f or 0)
            tx = [t for t in p.get("transactions", []) if (t.get("date") or "") <= as_of]
            tx.sort(key=lambda t: (t.get("date") or "", t.get("id") or 0))
            last = tx[-1] if tx else {}
            rows.append({"mlbam_id": p["id"], "mlb_2026_g": g, "mlb_2026_pa": pa,
                         "mlb_2026_ip": f"{outs // 3}.{outs % 3}",
                         "last_transaction_date": last.get("date"), "last_transaction": last.get("description")})
    return pd.DataFrame(rows)


def tier(row) -> str:
    if row.on_40man:
        s = "on MLB 40-man roster"
        if isinstance(row.roster_status, str) and "Injured" in row.roster_status:
            s += f" ({row.roster_status})"
        return s + f" — {row.roster_club}"
    return "not on any MLB 40-man roster"


def build(ids: list[int]) -> pd.DataFrame:
    as_of = config()["recruitment"]["as_of"]
    fm = forty_man(as_of)
    pp = people(sorted(set(int(i) for i in ids)), as_of)
    df = pd.DataFrame({"mlbam_id": sorted(set(int(i) for i in ids))})
    df = df.merge(fm.drop_duplicates("mlbam_id"), on="mlbam_id", how="left").merge(pp, on="mlbam_id", how="left")
    df["on_40man"] = df.roster_club.notna()
    df["recruitment_feasibility"] = df.apply(tier, axis=1) + f" (as of {as_of}; contract/salary/willingness not assessed)"
    df["roster_as_of"] = as_of
    df.attrs["clubs"] = fm.attrs["clubs"]
    return df


def load(ids: list[int] | None = None, force: bool = False) -> pd.DataFrame:
    out = processed("recruitment_status.csv")
    if out.exists() and not force:
        return pd.read_csv(out)
    if ids is None:
        ids = []
        for role in ("pitchers", "hitters"):
            log = pd.read_csv(processed(f"screening_log_{role}.csv"))
            ids += log.loc[log.screening_result == "candidate", "mlbam_id"].tolist()
    df = build(ids)
    df.to_csv(out, index=False)
    return df


if __name__ == "__main__":
    d = load(force=True)
    print(len(d), "candidates;", int(d.on_40man.sum()), "on a 40-man roster")
    print(d.roster_status.value_counts(dropna=False))
