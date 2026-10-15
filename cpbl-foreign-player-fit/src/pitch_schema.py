"""Common pitch-level schema shared by the CPBL (REBAS) and AAA (MLB feed) loaders.

Every pitch row carries the same derived flags so that Whiff%, Chase%, K% and
wOBA are computed with IDENTICAL numerators/denominators in both leagues:

  swing       : swinging strike, foul, foul tip, ball in play (bunt attempts excluded)
  whiff       : swinging strike (incl. blocked); foul tips count as contact
  is_bunt     : any bunt attempt -> removed from plate-discipline denominators
  in_zone     : league-specific zone definition (see loaders); NaN if unknown
  pa_end      : last pitch of a completed plate appearance
  woba_value / woba_denom : generic weights from config (same for both leagues)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import config

COLUMNS = [
    "league", "season", "game_id", "date", "pa_id", "pitch_seq",
    "pitcher_id", "pitcher_name", "pitcher_team", "batter_id", "batter_name", "batter_team",
    "p_throws", "stand", "pitcher_role", "balls", "strikes",
    "pitch_type_raw", "pitch_type", "pitch_group", "velo_mph",
    "in_zone", "swing", "whiff", "is_bunt", "called_strike",
    "pa_end", "pa_event", "woba_value", "woba_denom", "launch_speed",
]

PA_EVENTS = ["1B", "2B", "3B", "HR", "uBB", "IBB", "HBP", "SO", "OUT", "SF", "SH", "CI", "OTHER"]


def harmonise_pitch_type(raw: pd.Series) -> tuple[pd.Series, pd.Series]:
    m = config()["pitch_map"]
    common = raw.map(m["common"])          # unknown codes -> NaN (KN, EP, PO, blanks ...)
    group = common.map(m["group"])
    return common, group


def woba_columns(event: pd.Series) -> tuple[pd.Series, pd.Series]:
    """wOBA numerator/denominator per PA-ending row (NaN elsewhere handled by caller).

    denominator = AB + uBB + SF + HBP  ->  every PA except IBB, SH, CI, OTHER.
    """
    w = config()["woba_weights"]
    value = event.map({"uBB": w["uBB"], "HBP": w["HBP"], "1B": w["1B"], "2B": w["2B"],
                       "3B": w["3B"], "HR": w["HR"]}).fillna(0.0)
    denom = (~event.isin(["IBB", "SH", "CI", "OTHER"])).astype(float)
    return value, denom


def refresh_woba(df: pd.DataFrame) -> pd.DataFrame:
    """(Re)compute woba_value / woba_denom from pa_event with the current config weights, so a
    cached parquet never carries weights from an older config."""
    v, d = woba_columns(df["pa_event"].fillna("NONE"))
    df["woba_value"] = np.where(df["pa_end"], v, np.nan)
    df["woba_denom"] = np.where(df["pa_end"], d, np.nan)
    return df


def finalize(df: pd.DataFrame) -> pd.DataFrame:
    df["pitch_type"], df["pitch_group"] = harmonise_pitch_type(df["pitch_type_raw"])
    refresh_woba(df)
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    return df[COLUMNS]
