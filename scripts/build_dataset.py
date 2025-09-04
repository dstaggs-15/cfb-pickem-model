#!/usr/bin/env python3

import json
import datetime as dt
import os
import numpy as np
import pandas as pd

from .lib.io_utils import load_csv_local_or_url, save_json
from .lib.parsing import ensure_schedule_columns
from .lib.features import create_feature_set
from .lib.rolling import STAT_FEATURES
from .lib.market import fit_market_mapping

# File paths
LOCAL_DIR = "data/raw/cfbd"
LOCAL_SCHEDULE = f"{LOCAL_DIR}/cfb_schedule.csv"
LOCAL_TEAM_STATS = f"{LOCAL_DIR}/cfb_game_team_stats.csv"
LOCAL_LINES = f"{LOCAL_DIR}/cfb_lines.csv"
LOCAL_VENUES = f"{LOCAL_DIR}/cfb_venues.csv"
LOCAL_TEAMS = f"{LOCAL_DIR}/cfb_teams.csv"
LOCAL_TALENT = f"{LOCAL_DIR}/cfb_talent.csv"
RAW_BASE = "https://raw.githubusercontent.com/moneyball-ab/cfb-data/master/csv"
FALLBACK_SCHEDULE_URL = f"{RAW_BASE}/cfb_schedule.csv"
FALLBACK_TEAM_STATS_URL = f"{RAW_BASE}/cfb_game_team_stats.csv"

DERIVED = "data/derived"
TRAIN_PARQUET = f"{DERIVED}/training.parquet"
META_JSON = "docs/data/train_meta.json"
SEASON_AVG_PARQUET = f"{DERIVED}/season_averages.parquet"
MANUAL_LINES_CSV = "docs/input/lines.csv"  # for parity with create_feature_set signature

LAST_N = 5

def main():
    print("Building training dataset...")
    os.makedirs(DERIVED, exist_ok=True)
    os.makedirs(os.path.dirname(META_JSON), exist_ok=True)

    # --- Load raw data ---
    schedule = load_csv_local_or_url(LOCAL_SCHEDULE, FALLBACK_SCHEDULE_URL)
    schedule = ensure_schedule_columns(schedule)

    team_stats_long = load_csv_local_or_url(LOCAL_TEAM_STATS, FALLBACK_TEAM_STATS_URL)

    lines_df = pd.read_csv(LOCAL_LINES) if os.path.exists(LOCAL_LINES) else pd.DataFrame()
    venues_df = pd.read_csv(LOCAL_VENUES) if os.path.exists(LOCAL_VENUES) else pd.DataFrame()
    teams_df = pd.read_csv(LOCAL_TEAMS) if os.path.exists(LOCAL_TEAMS) else pd.DataFrame()
    talent_df = pd.read_csv(LOCAL_TALENT) if os.path.exists(LOCAL_TALENT) else pd.DataFrame()
    manual_lines_df = pd.read_csv(MANUAL_LINES_CSV) if os.path.exists(MANUAL_LINES_CSV) else pd.DataFrame()

    # --- Team stats pivot + engineered basics ---
    print("  Pivoting and cleaning raw team stats...")
    team_stats_long['stat_value'] = pd.to_numeric(team_stats_long['stat_value'], errors='coerce')

    team_stats = team_stats_long.pivot_table(
        index=['game_id', 'team'], columns='category', values='stat_value'
    ).reset_index()

    def camel_to_snake(name: str) -> str:
        return ''.join(['_' + c.lower() if c.isupper() else c for c in name]).lstrip('_')

    team_stats.columns = [camel_to_snake(col) for col in team_stats.columns]

    # Safe Series defaults
    idx = team_stats.index
    rushing_attempts = team_stats['rushing_attempts'] if 'rushing_attempts' in team_stats.columns else pd.Series(0.0, index=idx)
    pass_attempts    = team_stats['pass_attempts']    if 'pass_attempts'    in team_stats.columns else pd.Series(0.0, index=idx)
    total_yards      = team_stats['total_yards']      if 'total_yards'      in team_stats.columns else pd.Series(0.0, index=idx)
    first_downs      = team_stats['first_downs']      if 'first_downs'      in team_stats.columns else pd.Series(0.0, index=idx)

    total_plays = rushing_attempts.fillna(0) + pass_attempts.fillna(0)
    total_plays = total_plays.astype(float)

    team_stats['ppa'] = (total_yards.fillna(0).astype(float) / total_plays.replace(0, np.nan)).fillna(0.0)
    team_stats['success_rate'] = (first_downs.fillna(0).astype(float) / total_plays.replace(0, np.nan)).fillna(0.0)

    # --- Season averages (optional artifact) ---
    print("  Calculating and saving season average stats...")
    game_season_map = schedule[['game_id', 'season']]
    team_stats_with_season = team_stats.merge(game_season_map, on='game_id', how='left', validate='many_to_one')

    existing_stat_features = [feat for feat in STAT_FEATURES if feat in team_stats_with_season.columns]
    if existing_stat_features:
        season_avg_stats = (
            team_stats_with_season
            .groupby(['season', 'team'], as_index=False)[existing_stat_features]
            .mean()
        )
        season_avg_stats.to_parquet(SEASON_AVG_PARQUET, index=False)
    else:
        # Write an empty file for pipeline stability
        pd.DataFrame(columns=['season', 'team']).to_parquet(SEASON_AVG_PARQUET, index=False)

    # --- Build full feature set (historical training) ---
    # NOTE: pass manual_lines_df explicitly to match create_feature_set signature
    X, feature_list = create_feature_set(
        schedule=schedule,
        team_stats=team_stats,
        venues_df=venues_df,
        teams_df=teams_df,
        talent_df=talent_df,
        lines_df=lines_df,
        manual_lines_df=manual_lines_df,      # <- required by your features function
        games_to_predict_df=None              # training mode: None
    )

    # --- Labels ---
    # Ensure numeric comparison works even if points are strings
    X["home_win"] = (pd.to_numeric(X["home_points"], errors='coerce') >
                     pd.to_numeric(X["away_points"], errors='coerce')).astype(int)

    # --- Market mapping (guard for missing spreads) ---
    params = {}
    if 'spread_home' in X.columns and X['spread_home'].notna().any():
        spreads = pd.to_numeric(X["spread_home"], errors='coerce').to_numpy(dtype=float)
        labels  = X["home_win"].to_numpy(dtype=float)
        params = fit_market_mapping(spreads, labels) or {}

        if 'a' in params and 'b' in params:
            a, b = float(params["a"]), float(params["b"])
            X["market_home_prob"] = pd.to_numeric(X["spread_home"], errors='coerce').apply(
                lambda s: (1.0 / (1.0 + np.exp(-(a + b * (-(s)))))) if pd.notna(s) else np.nan
            )
        else:
            X["market_home_prob"] = np.nan
    else:
        X["market_home_prob"] = np.nan

    # Fill NA market probs within season when possible
    if 'season' in X.columns:
        X["market_home_prob"] = X.groupby("season")["market_home_prob"].transform(
            lambda s: s.fillna(s.mean())
        )

    # --- Finalize feature list & types ---
    # Include market/line features only if present
    extra = ["spread_home", "over_under", "market_home_prob"]
    final_feature_list = [f for f in (feature_list + extra) if f in X.columns]

    # Training rows: need outcomes present
    train_df = X.dropna(subset=["home_points", "away_points"]).copy()

    # Coerce features to float32 for model training stability/size
    for col in final_feature_list:
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce').fillna(0.0).astype('float32')

    train_df.to_parquet(TRAIN_PARQUET, index=False)

    meta = {
        "generated": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "features": final_feature_list,
        "market_params": params
    }
    save_json(META_JSON, meta)
    print(f"Wrote {TRAIN_PARQUET} and {META_JSON}")

if __name__ == "__main__":
    main()
