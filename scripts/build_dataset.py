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

# --- THIS SECTION WAS MISSING ---
LOCAL_DIR = "data/raw/cfbd"
LOCAL_SCHEDULE = f"{LOCAL_DIR}/cfb_schedule.csv"
LOCAL_TEAM_STATS = f"{LOCAL_DIR}/cfb_game_team_stats.csv"
LOCAL_LINES = f"{LOCAL_DIR}/cfb_lines.csv"
LOCAL_VENUES = f"{LOCAL_DIR}/cfbd_venues.csv"
LOCAL_TEAMS = f"{LOCAL_DIR}/cfbd_teams.csv"
LOCAL_TALENT = f"{LOCAL_DIR}/cfbd_talent.csv"

RAW_BASE = "https://raw.githubusercontent.com/moneyball-ab/cfb-data/master/csv"
FALLBACK_SCHEDULE_URL = f"{RAW_BASE}/cfb_schedule.csv"
FALLBACK_TEAM_STATS_URL = f"{RAW_BASE}/cfb_game_team_stats.csv"

DERIVED = "data/derived"
TRAIN_PARQUET = f"{DERIVED}/training.parquet"
META_JSON = "docs/data/train_meta.json"
SEASON_AVG_PARQUET = f"{DERIVED}/season_averages.parquet"
# --- END MISSING SECTION ---

LAST_N = 5
ENG_FEATURES_BASE = ["rest_diff","shortweek_diff","bye_diff","travel_diff_km","neutral_site","is_postseason"]
LINE_FEATURES = ["spread_home","over_under"]

def parse_ratio(s):
    # (Helper functions remain the same)
    pass

def parse_possession_time(s):
    # (Helper functions remain the same)
    pass

def main():
    print("Building training dataset ...")
    os.makedirs(DERIVED, exist_ok=True)

    schedule = load_csv_local_or_url(LOCAL_SCHEDULE, FALLBACK_SCHEDULE_URL)
    schedule = ensure_schedule_columns(schedule)
    team_stats = load_csv_local_or_url(LOCAL_TEAM_STATS, FALLBACK_TEAM_STATS_URL)
    
    print("  Cleaning raw team stats and creating features...")
    team_stats.rename(columns={'school': 'team'}, inplace=True)
    
    # --- DIAGNOSTIC LINE ---
    # This is the line we need the output from
    print("DEBUG: Columns available in team_stats:", team_stats.columns.tolist())
    # -----------------------
    
    # The script is still expected to fail on the next line
    team_stats['third_down_eff_rate'] = team_stats['third_down_eff'].apply(parse_ratio)
    
    # (The rest of the script is truncated as it won't be reached)

if __name__ == "__main__":
    main()
