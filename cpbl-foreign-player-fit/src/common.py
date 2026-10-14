"""Shared helpers: config, paths, small statistics utilities."""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def config() -> dict:
    with open(ROOT / "config" / "scoring.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def path(key: str) -> Path:
    p = (ROOT / config()["paths"][key]).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def raw(*parts: str) -> Path:
    return path("raw").joinpath(*parts)


def processed(*parts: str) -> Path:
    return path("processed").joinpath(*parts)


def snapshot() -> pd.Timestamp:
    return pd.Timestamp(config()["snapshot_date"])


# ---------------------------------------------------------------- statistics
def wilson(k, n, z: float = 1.96):
    """Wilson score interval; vectorised. Returns (lo, hi); NaN where n == 0."""
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = k / n
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    lo, hi = centre - half, centre + half
    lo = np.where(n > 0, lo, np.nan)
    hi = np.where(n > 0, hi, np.nan)
    return lo, hi


def shrink_rate(k, n, prior_rate, prior_n):
    """Beta-binomial (empirical-Bayes) shrinkage of k/n toward prior_rate."""
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    return (k + prior_rate * prior_n) / (n + prior_n)


def shrink_mean(total, n, prior_mean, prior_n):
    """Shrink a per-unit mean (e.g. wOBA = total/n) toward prior_mean."""
    total = np.asarray(total, dtype=float)
    n = np.asarray(n, dtype=float)
    return (total + prior_mean * prior_n) / (n + prior_n)


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def expit(x):
    return 1 / (1 + np.exp(-np.asarray(x, dtype=float)))


def zscore(s: pd.Series, clip: float | None = None) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std(ddof=0)
    z = (s - s.mean()) / sd if sd and not math.isnan(sd) else s * 0.0
    return z.clip(-clip, clip) if clip else z


def ip_to_outs(ip) -> int:
    """Convert baseball innings notation to outs.

    Accepts '12.1' / '12.2' strings or floats (12.1 == 12 1/3, NOT 12.1 innings)
    and true thirds (12.333.., 12.667..). Raises on impossible fractions like '12.5'.
    """
    if ip is None or (isinstance(ip, float) and math.isnan(ip)):
        raise ValueError("IP is missing")
    s = str(ip).strip()
    whole, _, frac = s.partition(".")
    whole_i = int(whole) if whole not in ("", "-") else 0
    if frac == "" or set(frac) == {"0"}:
        return whole_i * 3
    if frac in ("1", "2"):
        return whole_i * 3 + int(frac)
    # true decimals coming from (outs / 3)
    val = float(s)
    outs = val * 3
    if abs(outs - round(outs)) < 0.02:
        return int(round(outs))
    raise ValueError(f"Invalid innings value: {ip!r}")


def outs_to_ip_str(outs: int) -> str:
    return f"{outs // 3}.{outs % 3}"


def md_table(df: pd.DataFrame, floatfmt: str = ".3f", index: bool = False) -> str:
    """Tiny markdown table writer (avoids the optional `tabulate` dependency)."""
    d = df.reset_index() if index else df
    cols = list(d.columns)

    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if math.isnan(v) else format(v, floatfmt)
        return str(v)

    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, row in d.astype(object).iterrows():  # keep ints as ints in mixed rows
        lines.append("| " + " | ".join(fmt(v) for v in row.values) + " |")
    return "\n".join(lines)
