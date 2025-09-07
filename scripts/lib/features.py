# scripts/lib/features.py

import os
import pandas as pd
import numpy as np

from scripts.lib.rolling import long_stats_to_wide, build_sidewise_rollups

# Numeric stat names you actually expect (extend as your data allows)
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


def _log_small(name, df, rows=5):
    try:
        print(f"[FEATURES] {name}: shape={df.shape} cols={list(df.columns)[:20]}")
        if isinstance(df, pd.DataFrame) and not df.empty:
            print(df.head(rows).to_string(index=False))
    except Exception as e:
        print(f"[FEATURES] {name}: <failed to log: {e}>")


def _load_schedule_from_raw() -> pd.DataFrame:
    """
    Load schedule from cached CFBD CSV and normalize columns:
      - must have: game_id, season, week, home_team, away_team, date, home_points, away_points
    """
    path = "data/raw/cfbd/cfb_schedule.csv"
    if not os.path.exists(path):
        print(f"[FEATURES] WARNING: schedule raw file not found at {path}")
        return pd.DataFrame()

    df = pd.read_csv(path, low_memory=False)

    # Ensure required columns exist
    for c in ["game_id", "season", "week", "home_team", "away_team", "date", "home_points", "away_points"]:
        if c not in df.columns:
            df[c] = pd.NA

    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    for c in ["home_points", "away_points"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    keep = ["game_id", "season", "week", "home_team", "away_team", "date", "home_points", "away_points"]
    df = df[keep].drop_duplicates(subset=["game_id"])
    return df


def _melt_numeric_only(df: pd.DataFrame) -> pd.DataFrame:
    """
    Melt only numeric stat columns we expect. If none exist, return empty long frame.
    """
    value_vars = [c for c in df.columns if c in STAT_FEATURES]
    if not value_vars:
        return pd.DataFrame(columns=["season", "week", "game_id", "team", "home_away", "date", "stat", "value"])

    long_df = df.melt(
        id_vars=[c for c in ID_VARS if c in df.columns],
        value_vars=value_vars,
        var_name="stat",
        value_name="value",
    )
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    long_df = long_df.dropna(subset=["value"])
    return long_df


def create_feature_set(use_cache: bool = True, predict_only: bool = False):
    """
    Build the feature matrix X and the feature list.

    - loads schedule from data/raw/cfbd/cfb_schedule.csv
    - builds a shell wide_stats if no team stats are present (so we don't crash)
    - builds sidewise rollups safely
    - ALWAYS returns X with 'game_id'; includes labels if schedule present
    """
    # 1) Load schedule (has labels)
    schedule = _load_schedule_from_raw()
    _log_small("schedule (raw normalized)", schedule)

    # 2) Load team-wide stats if you produce them elsewhere (optional)
    possible_team_wide_paths = [
        "data/derived/team_wide.parquet",
        "data/derived/team_wide.feather",
        "data/derived/team_wide.csv",
    ]
    team_wide = None
    for p in possible_team_wide_paths:
        if os.path.exists(p):
            try:
                if p.endswith(".parquet"):
                    team_wide = pd.read_parquet(p)
                elif p.endswith(".feather"):
                    team_wide = pd.read_feather(p)
                else:
                    team_wide = pd.read_csv(p)
                break
            except Exception as e:
                print(f"[FEATURES] WARNING: failed to read {p}: {e}")
                team_wide = None

    if team_wide is None:
        team_wide = pd.DataFrame()

    _log_small("team_wide (pre-melt)", team_wide)

    # 3) Build long stats ONLY if team_wide exists and has numeric stats
    if team_wide.empty:
        print("[FEATURES] team_wide empty — proceeding with shell wide_stats keyed by schedule.game_id")
        if not schedule.empty and "game_id" in schedule.columns:
            wide_stats = pd.DataFrame({"game_id": schedule["game_id"].unique()})
        else:
            wide_stats = pd.DataFrame(columns=["game_id"])
    else:
        # Ensure id vars exist so melt doesn't die
        for col in ["season", "week", "game_id", "team", "home_away", "date"]:
            if col not in team_wide.columns:
                team_wide[col] = pd.NA

        team_stats_long = _melt_numeric_only(team_wide)
        _log_small("team_stats_long (post-melt)", team_stats_long)

        if team_stats_long.empty:
            print("[FEATURES] No numeric long stats — using shell wide_stats from schedule.")
            wide_stats = pd.DataFrame({"game_id": schedule["game_id"].unique()}) if "game_id" in schedule.columns else pd.DataFrame(columns=["game_id"])
        else:
            team_stats_sided = team_stats_long.pivot_table(
                index=["season", "week", "game_id", "team", "home_away", "date"],
                columns="stat",
                values="value",
                aggfunc="mean",
                observed=True,
            ).reset_index()
            _log_small("team_stats_sided (pre-wide)", team_stats_sided)
            wide_stats = long_stats_to_wide(team_stats_sided)

    _log_small("wide_stats (post-wide/shell)", wide_stats)

    # 4) Build sidewise rollups (safe if schedule is empty)
    LAST_N = 5
    home_roll, away_roll = build_sidewise_rollups(
        schedule=schedule,
        wide_stats=wide_stats,
        last_n=LAST_N,
        predict_df=None
    )
    _log_small("home_roll", home_roll)
    _log_small("away_roll", away_roll)

    # 5) Compose final X and include labels from schedule
    # ALWAYS ensure 'game_id' exists in the output, even if rollups are empty
    if home_roll.empty and away_roll.empty:
        if not schedule.empty and "game_id" in schedule.columns:
            X = schedule[["game_id", "season", "week", "home_points", "away_points"]].copy()
        else:
            X = pd.DataFrame(columns=["game_id", "season", "week", "home_points", "away_points"])
        feature_list = []  # no features yet
    else:
        X = home_roll.merge(away_roll, on=["game_id", "team"], how="outer", suffixes=("_home", "_away"))
        if not schedule.empty:
            labels = schedule[["game_id", "season", "week", "home_points", "away_points"]].copy()
            X = X.merge(labels, on="game_id", how="left")
        feature_list = [c for c in X.columns if c not in {"game_id", "team", "home_points", "away_points", "season", "week"}]

    # Guarantee 'game_id' is a column
    if "game_id" not in X.columns and getattr(X.index, "name", None) == "game_id":
        X = X.reset_index()

    return X, feature_list
