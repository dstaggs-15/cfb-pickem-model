# scripts/lib/features.py

import pandas as pd
import numpy as np

# IMPORTANT: use absolute package import, not relative
from scripts.lib.rolling import long_stats_to_wide, build_sidewise_rollups

# Keep this list as the numeric stat names you actually expect
STAT_FEATURES = [
    "ppa",
    "success_rate",
    "explosiveness",
    "rushing_ppa",
    "passing_ppa",
    "defense_ppa",
    "points_per_play",
    "yards_per_play",
]

ID_VARS = ["season", "week", "game_id", "team", "home_away", "date"]

def _is_numeric_series(s: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(s):
        return True
    coerced = pd.to_numeric(s, errors="coerce")
    return coerced.notna().any()

def _log_small(name, df, rows=5):
    try:
        print(f"[FEATURES] {name}: shape={df.shape} cols={list(df.columns)[:20]}")
        if not df.empty:
            print(df.head(rows).to_string(index=False))
    except Exception as e:
        print(f"[FEATURES] {name}: <failed to log: {e}>")

def _melt_numeric_only(df: pd.DataFrame) -> pd.DataFrame:
    value_vars = [c for c in df.columns if c in STAT_FEATURES]
    missing_stats = [c for c in STAT_FEATURES if c not in df.columns]
    if missing_stats:
        print(f"[FEATURES] WARNING: Missing expected stat columns: {missing_stats[:20]}{' ...' if len(missing_stats)>20 else ''}")

    long_df = df.melt(
        id_vars=[c for c in ID_VARS if c in df.columns],
        value_vars=value_vars,
        var_name="stat",
        value_name="value",
    )

    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    bad = long_df["value"].isna()
    if bad.any():
        sample = long_df.loc[bad, ["game_id", "team", "home_away", "stat", "value"]].head(25)
        print("[FEATURES] Dropping non-numeric long rows (sample):")
        print(sample.to_string(index=False))
        long_df = long_df.loc[~bad].copy()

    return long_df

def create_feature_set(use_cache: bool = True, predict_only: bool = False):
    # Load your actual inputs here. Replace these placeholders with your true sources.
    try:
        team_wide = pd.read_parquet("data/derived/team_wide.parquet")
    except Exception:
        team_wide = pd.DataFrame()

    try:
        schedule = pd.read_parquet("data/derived/schedule.parquet")
    except Exception:
        schedule = pd.DataFrame()

    _log_small("team_wide (pre-melt)", team_wide)
    _log_small("schedule (pre-rollups)", schedule)

    if team_wide.empty:
        print("[FEATURES] WARNING: team_wide is empty — downstream frames will be empty.")
        team_stats_long = pd.DataFrame(columns=["season","week","game_id","team","home_away","date","stat","value"])
    else:
        for col in ["season","week","game_id","team","home_away","date"]:
            if col not in team_wide.columns:
                team_wide[col] = pd.NA

        team_stats_long = _melt_numeric_only(team_wide)

    _log_small("team_stats_long (post-melt)", team_stats_long)

    team_stats_sided = team_stats_long.pivot_table(
        index=["season","week","game_id","team","home_away","date"],
        columns="stat",
        values="value",
        aggfunc="mean",
        observed=True,
    ).reset_index()

    _log_small("team_stats_sided (pre-wide)", team_stats_sided)

    wide_stats = long_stats_to_wide(team_stats_sided)
    _log_small("wide_stats (post-wide)", wide_stats)

    LAST_N = 5
    home_roll, away_roll = build_sidewise_rollups(
        schedule=schedule,
        wide_stats=wide_stats,
        last_n=LAST_N,
        predict_df=None
    )
    _log_small("home_roll", home_roll)
    _log_small("away_roll", away_roll)

    if home_roll.empty and away_roll.empty:
        X = pd.DataFrame()
        feature_list = []
    else:
        X = home_roll.merge(away_roll, on=["game_id","team"], how="outer", suffixes=("_home","_away"))
        feature_cols = [c for c in X.columns if c not in {"game_id","team"}]
        feature_list = feature_cols

    return X, feature_list
