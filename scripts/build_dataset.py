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

# File paths
LOCAL_DIR = "data/raw/cfbd"
LOCAL_SCHEDULE = f"{LOCAL_DIR}/cfb_schedule.csv"
LOCAL_TEAM_STATS = f"{LOCAL_DIR}/cfb_game_team_stats.csv"
LOCAL_LINES = f"{LOCAL_DIR}/cfb_lines.csv"
LOCAL_VENUES = f"{LOCAL_DIR}/cfbd_venues.csv"
LOCAL_TEAMS = f"{LOCAL_DIR}/cfb_teams.csv"
LOCAL_TALENT = f"{LOCAL_DIR}/cfb_talent.csv"
RAW_BASE = "https://raw.githubusercontent.com/moneyball-ab/cfb-data/master/csv"
FALLBACK_SCHEDULE_URL = f"{RAW_BASE}/cfb_schedule.csv"
FALLBACK_TEAM_STATS_URL = f"{RAW_BASE}/cfb_game_team_stats.csv"
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
                num, den = float(parts[0]), float(parts[1])
                return num / den if den != 0 else 0.0
            except (ValueError, TypeError): return np.nan
    return np.nan

def main():
    print("Building training dataset ...")
    os.makedirs(DERIVED, exist_ok=True)

    schedule = load_csv_local_or_url(LOCAL_SCHEDULE, FALLBACK_SCHEDULE_URL)
    schedule = ensure_schedule_columns(schedule)
    team_stats_long = load_csv_local_or_url(LOCAL_TEAM_STATS, FALLBACK_TEAM_STATS_URL)
    
    print("  Pivoting and cleaning raw team stats...")
    team_stats_long['stat_value'] = pd.to_numeric(team_stats_long['stat_value'], errors='coerce')
    
    team_stats = team_stats_long.pivot_table(
        index=['game_id', 'team'],
        columns='category',
        values='stat_value'
    ).reset_index()

    def camel_to_snake(name):
        return ''.join(['_' + c.lower() if c.isupper() else c for c in name]).lstrip('_')
    team_stats.columns = [camel_to_snake(col) for col in team_stats.columns]

    # Create derived features
    total_plays = team_stats.get('rushing_attempts', 0) + team_stats.get('pass_attempts', 0)
    team_stats['ppa'] = (team_stats.get('total_yards', 0) / total_plays.replace(0, np.nan)).fillna(0)
    team_stats['success_rate'] = (team_stats.get('first_downs', 0) / total_plays.replace(0, np.nan)).fillna(0)
    team_stats['explosiveness'] = (team_stats.get('yards_per_pass', 0).fillna(0) * 0.5 + team_stats.get('yards_per_rush_attempt', 0).fillna(0) * 0.5)
    team_stats['possession_seconds'] = team_stats.get('possession_time', '0:0').astype(str).apply(lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]))
    team_stats.drop(columns=['possession_time'], inplace=True, errors='ignore')

    team_stats = team_stats.drop_duplicates(subset=['game_id', 'team'])
    
    print("  Calculating and saving season average stats...")
    game_season_map = schedule[['game_id', 'season']]
    team_stats_with_season = team_stats.merge(game_season_map, on='game_id', how='left')
    
    existing_stat_features = [feat for feat in STAT_FEATURES if feat in team_stats_with_season.columns]
    season_avg_stats = team_stats_with_season.groupby(['season', 'team'])[existing_stat_features].mean().reset_index()
    season_avg_stats.to_parquet(SEASON_AVG_PARQUET, index=False)
    print(f"  Wrote season averages to {SEASON_AVG_PARQUET}")

    home_team_map = schedule[['game_id', 'home_team']]
    team_stats = team_stats.merge(home_team_map, on='game_id', how='left')
    team_stats['home_away'] = np.where(team_stats['team'] == team_stats['home_team'], 'home', 'away')
    team_stats = team_stats.drop(columns=['home_team'])
    
    wide = long_stats_to_wide(team_stats)

    lines_df = pd.read_csv(LOCAL_LINES) if os.path.exists(LOCAL_LINES) else pd.DataFrame()
    venues_df = pd.read_csv(LOCAL_VENUES) if os.path.exists(LOCAL_VENUES) else pd.DataFrame()
    teams_df = pd.read_csv(LOCAL_TEAMS) if os.path.exists(LOCAL_TEAMS) else pd.DataFrame()
    talent_df = pd.read_csv(LOCAL_TALENT) if os.path.exists(LOCAL_TALENT) else pd.DataFrame()

    home_roll, away_roll = build_sidewise_rollups(schedule, wide, LAST_N)
    
    base_cols = ["game_id","season","week","date","home_team","away_team","home_points","away_points","season_type","venue_id"]
    base = schedule[base_cols].copy()

    X = base.merge(home_roll, left_on=["game_id","home_team"], right_on=["game_id","team"], how="left").drop(columns=["team"])
    X = X.merge(away_roll, left_on=["game_id","away_team"], right_on=["game_id","team"], how="left").drop(columns=["team"])

    diff_cols = []
    for c in STAT_FEATURES:
        hc, ac = f"home_R{LAST_N}_{c}", f"away_R{LAST_N}_{c}"
        dc = f"diff_R{LAST_N}_{c}"
        
        if hc in X.columns and ac in X.columns:
            X[dc] = X[hc] - X[ac]
            diff_cols.append(dc)

    if f"home_R{LAST_N}_count" not in X.columns: X[f"home_R{LAST_N}_count"]=0.0
    if f"away_R{LAST_N}_count" not in X.columns: X[f"away_R{LAST_N}_count"]=0.0

    eng = rest_and_travel(schedule, teams_df, venues_df)
    X = X.merge(eng, on="game_id", how="left")
    med = median_lines(lines_df)
    X = X.merge(med, on="game_id", how="left")
    elo_df = pregame_probs(schedule, talent_df)
    X = X.merge(elo_df, on="game_id", how="left")
    X["home_win"] = (pd.to_numeric(X["home_points"], errors="coerce") > pd.to_numeric(X["away_points"], errors="coerce")).astype(int)

    feature_cols = diff_cols + [f"home_R{LAST_N}_count", f"away_R{LAST_N}_count"] + ENG_FEATURES_BASE + LINE_FEATURES + ["elo_home_prob","market_home_prob"]
    feature_cols = [col for col in feature_cols if col in X.columns]
    
    train_df = X.dropna(subset=["home_points","away_points"]).copy()

    for col in feature_cols:
        if col in train_df.columns:
            train_df[col] = pd.to_numeric(train_df[col], errors='coerce').fillna(0.0).astype('float32')

    train_df.to_parquet(TRAIN_PARQUET, index=False)

    meta = {
        "generated": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "last_n": LAST_N,
        "features": feature_cols,
        "market_params": fit_market_mapping(X["spread_home"].to_numpy(dtype=float), X["home_win"].to_numpy(dtype=float)),
    }
    save_json(META_JSON, meta)
    print(f"Wrote {TRAIN_PARQUET} and {META_JSON}")

if __name__ == "__main__":
    main()
