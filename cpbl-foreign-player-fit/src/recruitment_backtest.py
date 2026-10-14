"""Amendment 6 — recruitment backtest: does the list look like what CPBL clubs actually did?

For each fold (config.amendment6.backtest.folds) the pitcher list is rebuilt as of an October
date with only data available then, in two arms:

  old  one-season pool (latest season GS >= 10 & IP >= 60), 40-man players dropped
  new  two-season pool (amendment 6 rule 1) + signable tiers T1 / T2 (rule 2)

Ability is the amendment-5 pitcher baseline (K% .50, BB% .25, FIP .25, league-adjusted) on the
fold's seasons, with park factors re-estimated from those seasons only. No Fit and no
MLB-eligibility screen in either arm (the contest rule is season-specific). Positives are the
pitchers who joined CPBL the following season. Reported per arm: list size, positives in the
list, hits@k against the random expectation k*n/N, coverage and the median percentile rank
of the positives. Descriptive only: overlap says how much the list resembles the clubs'
choices, not whether its picks would pitch better. Nothing here changes the 2026 ranking.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import backtest, baseline, baseline_weights, milb_candidates, milb_loader, park, signability
from .common import ROOT, config, md_table, path, processed


def cfg() -> dict:
    return config()["amendment6"]


def _pitches(seasons) -> pd.DataFrame:
    a = milb_loader.load()
    if min(seasons) < a.season.min():
        a = pd.concat([backtest.aaa_extra(), a], ignore_index=True)
    return a[a.season.isin(seasons)]


def stints(seasons) -> pd.DataFrame:
    """Park-adjusted per-club AAA pitching stints of the fold's seasons (fold-own park factors)."""
    pf = park.factors_for(seasons, _pitches(seasons))
    df = pd.concat([baseline_weights.fetch("pitching", y, pf=pf) for y in seasons], ignore_index=True)
    df["league"] = df.team_league
    df["team_league"] = df.season.astype(str) + "_" + df.league
    return df


def ranked_pool(s: pd.DataFrame, arm: str) -> pd.DataFrame:
    """Pool for one arm with baseline scores (z within the arm's pool, before the tier filter)."""
    f, g = config()["candidate_filters"]["pitcher"], cfg()["pool"]["pitcher"]
    latest = s.season.max()
    if arm == "old":
        s = s[s.season == latest]
    df = milb_candidates._combine(s, [c for c in milb_candidates.PITCH_SUM if c in s] + ["outs"], "pitching_BF")
    df["IP"] = df.outs / 3
    df = milb_candidates._latest(s, df, ["pitching_GS", "pitching_IP"])
    if arm == "old":
        df = df[(df.pitching_GS >= f["min_gs"]) & (df.IP >= f["min_ip"])]
    else:
        df = df[(df.pitching_GS >= g["min_gs"]) & (df.IP >= g["min_ip"]) & (df.latest_pitching_IP >= g["latest_min_ip"])]
    empty = pd.DataFrame(columns=["player_id", "pitching_BF", "pitching_SO", "pitching_BB"])
    d = baseline.pitcher_metrics(df, int(latest), s, empty)
    d = baseline.score(d, "pitcher", baseline.judgment_weights("pitcher"))
    d["mlbam_id"] = d.player_id.astype(int)
    return d


def positives(kind: str) -> pd.DataFrame:
    ref = backtest.reference()
    if kind == "rebas_first_2025":
        c = pd.read_parquet(processed("cpbl_pitches.parquet"), columns=["pitcher_id", "season"])
        first = c.groupby("pitcher_id").season.min()
        new = set(first[first == 2025].index)
        p = ref[ref.rebas_id.isin(new) & ref.mlbam_id.notna()]
        return pd.DataFrame({"mlbam_id": p.mlbam_id.astype(int), "name": p.english_name})
    if kind == "news_2026":
        n = pd.read_csv(ROOT / "data/reference/cpbl_imports_2026_news.csv", comment="#", dtype={"mlbam_id": "Int64"})
        n = n[n.mlbam_id.notna() & ~n.mlbam_id.isin(ref.mlbam_id.dropna())]
        return pd.DataFrame({"mlbam_id": n.mlbam_id.astype(int), "name": n.english_name})
    raise ValueError(kind)


def evaluate_arm(d: pd.DataFrame, keep: pd.Series, pos: set[int], n_reachable: int, ks) -> tuple[dict, pd.DataFrame]:
    lst = d[keep.values].sort_values("baseline_score", ascending=False).reset_index(drop=True)
    lst["list_rank"] = np.arange(1, len(lst) + 1)
    hit = lst[lst.mlbam_id.isin(pos)]
    N, n = len(lst), len(hit)
    out = {"N": N, "positives_in_list": n, "coverage": n / n_reachable if n_reachable else None,
           "median_percentile": float(np.median(hit.list_rank / N)) if n else None}
    for k in ks:
        out[f"hits@{k}"] = int((hit.list_rank <= k).sum())
        out[f"random@{k}"] = round(k * n / N, 2) if N else None
    return out, hit[["mlbam_id", "name", "list_rank"]]


def run_fold(name: str, f: dict) -> dict:
    s = stints(f["seasons"])
    pos = positives(f["positives"])
    reach = set(pos.mlbam_id) & set(s.loc[s.pitching_BF > 0, "player_id"].astype(int))
    res = {"as_of": f["as_of"], "seasons": f["seasons"], "primary": f["primary"], "positives": len(pos),
           "positives_with_aaa": len(reach)}
    detail = []
    for arm in ("old", "new"):
        d = ranked_pool(s, arm)
        t = signability.tiers(d.mlbam_id, f["as_of"])
        d = d.merge(t[["mlbam_id", "sign_tier", "signable"]], on="mlbam_id", how="left")
        keep = d.sign_tier.ne("T4_40man") if arm == "old" else d.signable.fillna(False).astype(bool)
        res[arm], hit = evaluate_arm(d, keep, set(pos.mlbam_id), len(reach), cfg()["backtest"]["ks"])
        res[arm]["pool_before_tiers"] = len(d)
        res[arm]["tier_counts"] = d.sign_tier.value_counts().to_dict()
        miss = pos[pos.mlbam_id.isin(reach) & ~pos.mlbam_id.isin(hit.mlbam_id)].mlbam_id
        why = d.set_index("mlbam_id").reindex(miss)
        res[arm]["missed"] = {pos.set_index("mlbam_id").name[m]: ("not in pool" if pd.isna(why.sign_tier.get(m))
                                                                  else why.sign_tier[m]) for m in miss}
        detail.append(hit.assign(fold=name, arm=arm))
    res["detail"] = pd.concat(detail).to_dict(orient="records")
    return res


def run() -> dict:
    out = {k: run_fold(k, f) for k, f in cfg()["backtest"]["folds"].items()}
    with open(processed("recruitment_backtest.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o), ensure_ascii=False)
    lines = ["# Recruitment backtest (amendment 6; auto-generated by `python -m src.recruitment_backtest`)\n"]
    for k, r in out.items():
        lines.append(f"\n## Fold {k} ({'primary' if r['primary'] else 'exploratory'}): as of {r['as_of']}, "
                     f"AAA {r['seasons'][0]}-{r['seasons'][-1]}; positives {r['positives']}, "
                     f"with AAA pitching {r['positives_with_aaa']}\n")
        rows = []
        for arm in ("old", "new"):
            a = r[arm]
            rows.append({"arm": arm, "pool": a["pool_before_tiers"], "list N": a["N"], "positives in list": a["positives_in_list"],
                         "coverage": a["coverage"], "hits@10": a["hits@10"], "random@10": a["random@10"],
                         "hits@20": a["hits@20"], "random@20": a["random@20"], "median pct rank": a["median_percentile"]})
        lines.append(md_table(pd.DataFrame(rows), ".2f"))
        det = pd.DataFrame(r["detail"])
        if len(det):
            lines.append("\n" + md_table(det.pivot_table(index="name", columns="arm", values="list_rank").reset_index(), ".0f"))
        for arm in ("old", "new"):
            if r[arm]["missed"]:
                lines.append(f"\nMissed by `{arm}` (positives with AAA pitching): "
                             + ", ".join(f"{n} ({w})" for n, w in r[arm]["missed"].items()))
    (path("reports") / "recruitment_backtest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    r = run()
    print(json.dumps({k: {a: {x: y for x, y in v[a].items() if x != "missed"} for a in ("old", "new")}
                      for k, v in r.items()}, indent=1, default=str))
