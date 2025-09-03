#!/usr/bin/env python3

import json, datetime as dt
import os
import sys
import numpy as np
import pandas as pd

from .lib.io_utils import load_csv_local_or_url, save_json
from .lib.parsing import ensure_schedule_columns
from .lib.rolling import long_stats_to_wide, build_sidewise_rollups, STAT_FEATURES
from .lib.context import rest_and_travel
from .lib.market import median_lines, fit_market_mapping
from .lib.elo import pregame_probs

# (File paths are the same)
LOCAL_DIR = "data/raw/cfbd"
LOCAL_SCHEDULE = f"{LOCAL_DIR}/cfb_schedule.csv"
LOCAL_TEAM_STATS = f"{LOCAL_DIR}/cfb_game_team_stats.csv"
# ... (rest of file paths) ...
DERIVED = "data/derived"
TRAIN_PARQUET = f"{DERIVED}/training.parquet"
META_JSON = "docs/data/train_meta.json"
SEASON_AVG_PARQUET = f"{DERIVED}/season_averages.parquet"

LAST_N = 5
ENG_FEATURES_BASE = ["rest_diff","shortweek_diff","bye_diff","travel_diff_km","neutral_site","is_postseason"]
LINE_FEATURES = ["spread_home","over_under"]

def parse_ratio(s):
    if isinstance(s, str) and '-' in s:
        parts = s.split('-')
        if len(parts) == 2 and parts[1] is not None and parts[1] != '0':
            try:
                num = float(parts[0])
                den = float(parts[1])
                if den == 0: return 0.0
                return num / den
            except (ValueError, TypeError):
                return np.nan
    return np.nan

def parse_possession_time(s):
    if isinstance(s, str) and ':' in s:
        parts = s.split(':')
        if len(parts) == 2:
            try:
                return float(parts[0]) * 60 + float(parts[1])
            except (ValueError, TypeError):
                return np.nan
    return np.nan

def main():
    print("Building training dataset ...")
    os.makedirs(DERIVED, exist_ok=True)

    schedule = load_csv_local_or_url(LOCAL_SCHEDULE, FALLBACK_SCHEDULE_URL)
    schedule = ensure_schedule_columns(schedule)
    team_stats = load_csv_local_or_url(LOCAL_TEAM_STATS, FALLBACK_TEAM_STATS_URL)
    
    print("  Cleaning raw team stats and creating features...")
    team_stats.rename(columns={'school': 'team'}, inplace=True)
    
    # --- DIAGNOSTIC LINE ADDED HERE ---
    # This will print the exact list of column names available in your dataframe
    print("DEBUG: Columns available in team_stats:", team_stats.columns.tolist())
    # ------------------------------------
    
    # The script will still fail on the next line, which is expected.
    team_stats['third_down_eff_rate'] = team_stats['third_down_eff'].apply(parse_ratio)
    team_stats['fourth_down_eff_rate'] = team_stats['fourth_down_eff'].apply(parse_ratio)
    
    # (The rest of the script is here but won't be reached)
    # ...
    
if __name__ == "__main__":
    main()
