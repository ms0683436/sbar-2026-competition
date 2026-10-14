"""Amendment 6 — signability tiers from official MLB transactions and 40-man rosters.

A CPBL club can only sign a player his MLB organisation lets go of (or buys out). As of a date
`as_of`, every player gets one tier (config.amendment6.signability):

  T4_40man       on an MLB 40-man roster on as_of
  T1_free        not on a 40-man; his latest affiliated status event is "Declared Free Agency"
                 (DFA) or "Released" (REL) and nothing re-established control after it
  T2_expiring    still controlled, and either his latest affiliated signing was a free-agent
                 minor-league signing (SFA; such contracts normally end with the season) or his
                 first professional season is <= as_of year - expiring_years. statsapi files an
                 international amateur's first contract as SFA too; an SFA counts only when an
                 affiliated "Declared Free Agency" / "Released" precedes it (clarified after the
                 first tier table showed 2021-22 international signees as T2). The first
                 professional season is the year of the first affiliated transaction (fixed after
                 the first 2026 run counted college and showcase assignments)
  T3_controlled  every other player not on a 40-man

Only transactions whose toTeam or fromTeam is an affiliated club (MLB, AAA to Rookie) count, so
winter-league, Mexican-league or independent assignments neither free nor bind a player.
Signable = T1 + T2. These are facts and a rule of thumb, not contract terms: a "free" player
may still not want to play in Taiwan, and a T3 player can be bought out.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from . import roster_status, statsapi_pbp
from .common import config, processed, raw

CONTRACTS = {"SFA", "SGN"}     # events that create a contract; trades and waiver claims only move it


def cfg() -> dict:
    return config()["amendment6"]["signability"]


def affiliated_teams(seasons=range(2010, 2027)) -> set[int]:
    f = raw("statsapi_season", "affiliated_teams.csv")
    if f.exists():
        return set(pd.read_csv(f).team_id)
    ids = set()
    sport = ",".join(map(str, cfg()["affiliated_sport_ids"]))
    for y in seasons:
        d = statsapi_pbp._get(f"{statsapi_pbp.API}/v1/teams", {"sportIds": sport, "season": y})
        ids |= {t["id"] for t in d.get("teams", [])}
    f.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"team_id": sorted(ids)}).to_csv(f, index=False)
    return ids


def _people(chunk: list[int]) -> list[dict]:
    d = statsapi_pbp._get(f"{statsapi_pbp.API}/v1/people",
                          {"personIds": ",".join(map(str, chunk)), "hydrate": "transactions"})
    rows = []
    for p in d.get("people", []):
        base = {"mlbam_id": p["id"], "draft_year": p.get("draftYear")}
        tx = p.get("transactions", [])
        if not tx:
            rows.append(base)
        for t in tx:
            rows.append({**base, "date": t.get("date"), "tx_id": t.get("id"), "type": t.get("typeCode"),
                         "to_team": (t.get("toTeam") or {}).get("id"), "from_team": (t.get("fromTeam") or {}).get("id"),
                         "description": t.get("description")})
    return rows


def transactions(ids, force: bool = False) -> pd.DataFrame:
    """Full transaction history per player (cached; missing ids are fetched)."""
    f = raw("statsapi_season", "transactions.csv")
    ids = sorted(set(int(i) for i in ids))
    have = pd.read_csv(f) if f.exists() and not force else pd.DataFrame(columns=["mlbam_id"])
    need = sorted(set(ids) - set(have.mlbam_id.astype(int)))
    if need:
        chunks = [need[i:i + 50] for i in range(0, len(need), 50)]
        with ThreadPoolExecutor(4) as ex:
            rows = [r for part in ex.map(_people, chunks) for r in part]
        have = pd.concat([have, pd.DataFrame(rows)], ignore_index=True)
        f.parent.mkdir(parents=True, exist_ok=True)
        have.to_csv(f, index=False)
    return have[have.mlbam_id.astype(int).isin(ids)]


def classify(tx: pd.DataFrame, as_of: str, on40: set[int], aff: set[int]) -> pd.DataFrame:
    """One tier per player from his transactions dated <= as_of."""
    c = cfg()
    free, control = set(c["free_events"]), set(c["control_events"])
    year = int(as_of[:4])
    rows = []
    for pid, g in tx.groupby("mlbam_id"):
        pid = int(pid)
        g = g[g.date.notna()]
        g = g[(g.date <= as_of) & (g.to_team.isin(aff) | g.from_team.isin(aff))].sort_values(["date", "tx_id"])
        # first professional season = first affiliated transaction (college / showcase / summer-league
        # assignments are not affiliated and do not count)
        first = int(g.date.iloc[0][:4]) if len(g) else None
        st = g[g.type.isin(free | control)]
        last = st.iloc[-1] if len(st) else None
        sign = g[g.type.isin(CONTRACTS)]
        last_sign = sign.iloc[-1].type if len(sign) else None
        # statsapi also files a first (amateur, international) contract as "SFA ... to a minor league
        # contract"; only an SFA that follows an affiliated free event is a free-agent re-signing
        if last_sign == "SFA" and not (g.type.isin(free) & (g.date <= sign.iloc[-1].date)).any():
            last_sign = "SFA_first_contract"
        if pid in on40:
            tier = "T4_40man"
        elif last is not None and last.type in free:
            tier = "T1_free"
        elif last_sign == c["expiring_signing"] or (first is not None and first <= year - c["expiring_years"]):
            tier = "T2_expiring"
        else:
            tier = "T3_controlled"
        rows.append({"mlbam_id": pid, "sign_tier": tier, "first_pro_season": first, "last_signing": last_sign,
                     "last_status_date": None if last is None else last.date,
                     "last_status_event": None if last is None else last.description})
    out = pd.DataFrame(rows)
    out["signable"] = out.sign_tier.isin(c["signable"])
    out["sign_as_of"] = as_of
    return out


def tiers(ids, as_of: str | None = None) -> pd.DataFrame:
    as_of = as_of or cfg()["as_of"]
    ids = sorted(set(int(i) for i in ids))
    fm = roster_status.forty_man(as_of)
    out = classify(transactions(ids), as_of, set(fm.mlbam_id), affiliated_teams())
    return pd.DataFrame({"mlbam_id": ids}).merge(out, on="mlbam_id", how="left").fillna(
        {"sign_tier": "T3_controlled", "signable": False, "sign_as_of": as_of})


def t2_check(ids) -> dict:
    """The T2 rule once on 2025: tiers as of 2025-10-13 vs who actually declared free agency or
    was released by an affiliated club in (as_of, until] (event-based, re-signings included)."""
    k = cfg()["t2_check"]
    m = tiers(ids, k["as_of"])
    m = m[m.sign_tier.isin(["T2_expiring", "T3_controlled"])]
    tx = transactions(m.mlbam_id)
    aff = affiliated_teams()
    ev = tx[tx.type.isin(cfg()["free_events"]) & (tx.date > k["as_of"]) & (tx.date <= k["until"])
            & (tx.to_team.isin(aff) | tx.from_team.isin(aff))]
    became = m.mlbam_id.isin(ev.mlbam_id.astype(int)).values
    t2 = m.sign_tier.eq("T2_expiring").values
    return {"as_of": k["as_of"], "until": k["until"], "n_controlled_non40": int(len(m)),
            "n_t2": int(t2.sum()), "n_became_free": int(became.sum()),
            "precision_t2": float(became[t2].mean()) if t2.any() else None,
            "recall_t2": float(t2[became].mean()) if became.any() else None,
            "free_rate_t3": float(became[~t2].mean()) if (~t2).any() else None,
            "note": "population: amendment-6 2026 candidates not on a 40-man on 2025-10-13"}


if __name__ == "__main__":
    log = pd.read_csv(processed("screening_log_pitchers.csv"))
    t = tiers(log.loc[log.screening_result == "candidate", "mlbam_id"])
    print(t.sign_tier.value_counts())
