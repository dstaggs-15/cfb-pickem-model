#!/usr/bin/env python3

import json, datetime as dt
import os
import numpy as np
import pandas as pd

from .lib.io_utils import load_csv_local_or_url, save_json
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
    print("Building training dataset...")
    os.makedirs(DERIVED, exist_ok=True)
    os.makedirs(os.path.dirname(META_JSON), exist_ok=True)

    # --- Load raw data ---
    schedule = pd.read_csv(LOCAL_SCHEDULE)
    schedule = ensure_schedule_columns(schedule)
    
    # This file now contains advanced stats directly
    team_stats = pd.read_csv(LOCAL_TEAM_STATS)
    
    lines_df = pd.read_csv(LOCAL_LINES) if os.path.exists(LOCAL_LINES) else pd.DataFrame()
    venues_df = pd.read_csv(LOCAL_VENUES) if os.path.exists(LOCAL_VENUES) else pd.DataFrame()
    teams_df = pd.read_csv(LOCAL_TEAMS) if os.path.exists(LOCAL_TEAMS) else pd.DataFrame()
    talent_df = pd.read_csv(LOCAL_TALENT) if os.path.exists(LOCAL_TALENT) else pd.DataFrame()
    manual_lines_df = pd.read_csv(MANUAL_LINES_CSV) if os.path.exists(MANUAL_LINES_CSV) else pd.DataFrame()

    # --- Rename columns for consistency and select key stats ---
    # The new data source has a different schema, so we adapt it here.
    rename_map = {
        'gameId': 'game_id',
        'offense.ppa': 'ppa',
        'offense.successRate': 'success_rate',
        'offense.explosiveness': 'explosiveness',
        'offense.rushingPPA': 'rushing_ppa',
        'offense.passingPPA': 'passing_ppa',
        'defense.ppa': 'defense_ppa',
    }
    team_stats.rename(columns=rename_map, inplace=True)
    
    # These are the features we will calculate rolling averages for
    STAT_FEATURES = ['ppa', 'success_rate', 'explosiveness', 'rushing_ppa', 'passing_ppa', 'defense_ppa']
    
    # --- Season averages (for carry-forward logic) ---
    print("  Calculating and saving season average stats...")
    season_avg_stats = team_stats.groupby(['season', 'team'], as_index=False)[STAT_FEATURES].mean()
    season_avg_stats.to_parquet(SEASON_AVG_PARQUET, index=False)

    # --- Build full feature set ---
    X, feature_list = create_feature_set(
        schedule=schedule,
        team_stats=team_stats,
        venues_df=venues_df,
        teams_df=teams_df,
        talent_df=talent_df,
        lines_df=lines_df,
        manual_lines_df=manual_lines_df,
        games_to_predict_df=None
    )

    # --- Labels & Market Mapping ---
    X["home_win"] = (pd.to_numeric(X["home_points"], errors='coerce') >
                     pd.to_numeric(X["away_points"], errors='coerce')).astype(int)

    params = {}
    if 'spread_home' in X.columns and X['spread_home'].notna().any():
        params = fit_market_mapping(X["spread_home"], X["home_win"]) or {}
        if 'a' in params and 'b' in params:
            a, b = float(params["a"]), float(params["b"])
            X["market_home_prob"] = 1.0 / (1.0 + np.exp(-(a + b * (-X["spread_home"]))))
        else:
            X["market_home_prob"] = np.nan
    else:
        X["market_home_prob"] = np.nan

    X["market_home_prob"] = X.groupby("season")["market_home_prob"].transform(lambda s: s.fillna(s.mean()))

    # --- Finalize and Save ---
    extra = ["spread_home", "over_under", "market_home_prob"]
    final_feature_list = [f for f in (feature_list + extra) if f in X.columns]
    
    train_df = X.dropna(subset=["home_points", "away_points"]).copy()
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
