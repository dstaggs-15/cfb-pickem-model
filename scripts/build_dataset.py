#!/usr/bin/env python3

import json, datetime as dt
import os
import numpy as np
import pandas as pd

from .lib.io_utils import load_csv_local_or_url, save_json
from .lib.parsing import ensure_schedule_columns
from .lib.features import create_feature_set
from .lib.rolling import STAT_FEATURES
from .lib.market import fit_market_mapping, median_lines

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

def main():
    print("Building training dataset...")
    os.makedirs(DERIVED, exist_ok=True)

    # Load all raw data
    schedule = load_csv_local_or_url(LOCAL_SCHEDULE, FALLBACK_SCHEDULE_URL)
    schedule = ensure_schedule_columns(schedule)
    team_stats_long = load_csv_local_or_url(LOCAL_TEAM_STATS, FALLBACK_TEAM_STATS_URL)
    lines_df = pd.read_csv(LOCAL_LINES) if os.path.exists(LOCAL_LINES) else pd.DataFrame()
    venues_df = pd.read_csv(LOCAL_VENUES) if os.path.exists(LOCAL_VENUES) else pd.DataFrame()
    teams_df = pd.read_csv(LOCAL_TEAMS) if os.path.exists(LOCAL_TEAMS) else pd.DataFrame()
    talent_df = pd.read_csv(LOCAL_TALENT) if os.path.exists(LOCAL_TALENT) else pd.DataFrame()

    # Pre-process stats (pivot and create derived features)
    team_stats_long['stat_value'] = pd.to_numeric(team_stats_long['stat_value'], errors='coerce')
    team_stats = team_stats_long.pivot_table(
        index=['game_id', 'team'], columns='category', values='stat_value'
    ).reset_index()
    def camel_to_snake(name): return ''.join(['_' + c.lower() if c.isupper() else c for c in name]).lstrip('_')
    team_stats.columns = [camel_to_snake(col) for col in team_stats.columns]
    
    rushing_attempts = team_stats.get('rushing_attempts', 0)
    pass_attempts = team_stats.get('pass_attempts', 0)
    total_yards = team_stats.get('total_yards', 0)
    first_downs = team_stats.get('first_downs', 0)
    total_plays = rushing_attempts + pass_attempts
    team_stats['ppa'] = (total_yards / total_plays.replace(0, np.nan)).fillna(0)
    team_stats['success_rate'] = (first_downs / total_plays.replace(0, np.nan)).fillna(0)

    # Calculate and save season averages for carry-forward logic
    print("  Calculating and saving season average stats...")
    game_season_map = schedule[['game_id', 'season']]
    team_stats_with_season = team_stats.merge(game_season_map, on='game_id', how='left')
    existing_stat_features = [feat for feat in STAT_FEATURES if feat in team_stats_with_season.columns]
    season_avg_stats = team_stats_with_season.groupby(['season', 'team'])[existing_stat_features].mean().reset_index()
    season_avg_stats.to_parquet(SEASON_AVG_PARQUET, index=False)
    
    # Create the main feature set using the shared function
    X, feature_list = create_feature_set(schedule, team_stats, venues_df, teams_df, talent_df, lines_df)

    # Add training-specific columns
    X["home_win"] = (pd.to_numeric(X["home_points"]) > pd.to_numeric(X["away_points"])).astype(int)
    
    # Add market features and fit the mapping
    med_lines = median_lines(lines_df)
    X = X.merge(med_lines, on="game_id", how="left")
    
    params = fit_market_mapping(X["spread_home"].to_numpy(dtype=float), X["home_win"].to_numpy(dtype=float))
    if params and 'a' in params and 'b' in params:
        a, b = params["a"], params["b"]
        X["market_home_prob"] = X["spread_home"].apply(lambda s: (1/(1+np.exp(-(a + b * (-(s))))) if pd.notna(s) else np.nan))
    X["market_home_prob"] = X.groupby("season")["market_home_prob"].transform(lambda s: s.fillna(s.mean()))
    
    # Finalize feature list
    final_feature_list = feature_list + ["spread_home", "over_under", "market_home_prob"]
    final_feature_list = [f for f in final_feature_list if f in X.columns]
    
    # Clean and save final training data
    train_df = X.dropna(subset=["home_points", "away_points"]).copy()
    for col in final_feature_list:
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce').fillna(0.0).astype('float32')

    train_df.to_parquet(TRAIN_PARQUET, index=False)

    meta = {
        "generated": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "features": final_feature_list,
        "market_params": params if params else {}
    }
    save_json(META_JSON, meta)
    print(f"Wrote {TRAIN_PARQUET} and {META_JSON}")

if __name__ == "__main__":
    main()
