#!/usr/bin/env python3

import json, datetime as dt
import os
import sys
import numpy as np
import pandas as pd

from .lib.io_utils import save_json
from .lib.parsing import ensure_schedule_columns
from .lib.features import create_feature_set
from .lib.market import fit_market_mapping

# File paths
LOCAL_DIR = "data/raw/cfbd"
LOCAL_SCHEDULE = f"{LOCAL_DIR}/cfb_schedule.csv"
LOCAL_TEAM_STATS = f"{LOCAL_DIR}/cfb_game_team_stats.csv"
LOCAL_LINES = f"{LOCAL_DIR}/cfb_lines.csv"
LOCAL_VENUES = f"{LOCAL_DIR}/cfbd_venues.csv"
LOCAL_TEAMS = f"{LOCAL_DIR}/cfbd_teams.csv"
LOCAL_TALENT = f"{LOCAL_DIR}/cfbd_talent.csv"

DERIVED = "data/derived"
TRAIN_PARQUET = f"{DERIVED}/training.parquet"
META_JSON = "docs/data/train_meta.json"
SEASON_AVG_PARQUET = f"{DERIVED}/season_averages.parquet"
MANUAL_LINES_CSV = "docs/input/lines.csv"

def main():
    """
    This script orchestrates the creation of the final training dataset.
    It loads raw data, cleans it, engineers features, and saves the result.
    """
    print("Building training dataset...")
    os.makedirs(DERIVED, exist_ok=True)
    os.makedirs(os.path.dirname(META_JSON), exist_ok=True)

    # --- 1. Load All Raw Data ---
    if not os.path.exists(LOCAL_SCHEDULE):
        print("FATAL: cfb_schedule.csv not found. Please run the data fetch workflow.")
        sys.exit(1)
        
    schedule = pd.read_csv(LOCAL_SCHEDULE, low_memory=False)
    team_stats = pd.read_csv(LOCAL_TEAM_STATS) if os.path.exists(LOCAL_TEAM_STATS) else pd.DataFrame()
    lines_df = pd.read_csv(LOCAL_LINES) if os.path.exists(LOCAL_LINES) else pd.DataFrame()
    venues_df = pd.read_csv(LOCAL_VENUES) if os.path.exists(LOCAL_VENUES) else pd.DataFrame()
    teams_df = pd.read_csv(LOCAL_TEAMS) if os.path.exists(LOCAL_TEAMS) else pd.DataFrame()
    talent_df = pd.read_csv(LOCAL_TALENT) if os.path.exists(LOCAL_TALENT) else pd.DataFrame()
    manual_lines_df = pd.read_csv(MANUAL_LINES_CSV) if os.path.exists(MANUAL_LINES_CSV) else pd.DataFrame()

    # --- 2. Data Cleaning and Preparation ---
    print("  Cleaning and preparing raw data...")
    
    # Standardize game_id column name from 'id' to 'game_id'
    if 'id' in schedule.columns:
        schedule.rename(columns={'id': 'game_id'}, inplace=True)
    if 'gameId' in team_stats.columns:
        team_stats.rename(columns={'gameId': 'game_id'}, inplace=True)

    # Ensure game_id is a consistent string type for merging
    schedule['game_id'] = schedule['game_id'].astype(str)
    if not team_stats.empty:
        team_stats['game_id'] = team_stats['game_id'].astype(str)

    schedule = ensure_schedule_columns(schedule)
    
    # Rename raw stat columns to a cleaner format
    rename_map = {
        'offense.ppa': 'ppa', 'offense.successRate': 'success_rate',
        'offense.explosiveness': 'explosiveness', 'defense.ppa': 'defense_ppa',
    }
    team_stats.rename(columns=rename_map, inplace=True)
    
    # Force stat columns to be numeric, converting errors to missing values (NaN)
    stat_cols = ['ppa', 'success_rate', 'explosiveness', 'defense_ppa']
    for col in stat_cols:
        if col in team_stats.columns:
            team_stats[col] = pd.to_numeric(team_stats[col], errors='coerce')

    # --- 3. Create Season Averages for Carry-Forward Logic ---
    print("  Calculating and saving season average stats...")
    
    # Crucially, only use games that have actually been played to calculate averages
    completed_games_schedule = schedule[schedule['home_points'].notna()].copy()
    completed_stats = team_stats[team_stats['game_id'].isin(completed_games_schedule['game_id'])]
    
    existing_stat_features = [col for col in stat_cols if col in completed_stats.columns]
    
    season_avg_stats = completed_stats.groupby(['season', 'team'], as_index=False)[existing_stat_features].mean()
    season_avg_stats.to_parquet(SEASON_AVG_PARQUET, index=False)

    # --- 4. Build the Full Feature Set ---
    print("  Creating feature set for all historical games...")
    X, feature_list = create_feature_set(
        schedule=schedule, team_stats=team_stats, venues_df=venues_df,
        teams_df=teams_df, talent_df=talent_df, lines_df=lines_df,
        manual_lines_df=manual_lines_df, games_to_predict_df=None
    )

    # --- 5. Add Labels and Market-Derived Features ---
    X["home_win"] = (X["home_points"] > X["away_points"]).astype(int)

    params = {}
    if 'spread_home' in X.columns and X['spread_home'].notna().any():
        params = fit_market_mapping(X["spread_home"], X["home_win"]) or {}
        if 'a' in params and 'b' in params:
            a, b = float(params["a"]), float(params["b"])
            X["market_home_prob"] = 1.0 / (1.0 + np.exp(-(a + b * (-X["spread_home"]))))
    
    if "market_home_prob" not in X.columns:
        X["market_home_prob"] = np.nan
        
    X["market_home_prob"] = X.groupby("season")["market_home_prob"].transform(lambda s: s.fillna(s.mean()))

    # --- 6. Finalize and Save Training Data ---
    extra = ["spread_home", "over_under", "market_home_prob"]
    final_feature_list = [f for f in (feature_list + extra) if f in X.columns]
    
    # The final training data should only include completed games
    train_df = X[X['home_points'].notna()].copy()
    
    for col in final_feature_list:
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce').fillna(0.0).astype('float32')

    train_df.to_parquet(TRAIN_PARQUET, index=False)

    meta = {
        "generated": dt.datetime.utcnow().isoformat(),
        "features": final_feature_list,
        "market_params": params
    }
    save_json(META_JSON, meta)
    print(f"Wrote {len(train_df)} rows to {TRAIN_PARQUET} and updated {META_JSON}")

if __name__ == "__main__":
    main()

