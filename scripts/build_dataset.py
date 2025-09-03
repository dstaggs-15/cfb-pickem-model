#!/usr/bin/env python3

import json, datetime as dt
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss

from .lib.io_utils import load_csv_local_or_url, save_json
from .lib.parsing import ensure_schedule_columns
from .lib.rolling import long_stats_to_wide, build_sidewise_rollups, STAT_FEATURES
from .lib.context import rest_and_travel
from .lib.market import median_lines, fit_market_mapping
from .lib.elo import pregame_probs

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

LAST_N = 5
ENG_FEATURES_BASE = ["rest_diff","shortweek_diff","bye_diff","travel_diff_km","neutral_site","is_postseason"]
LINE_FEATURES = ["spread_home","over_under"]

def main():
    print("Building training dataset ...")
    os.makedirs(DERIVED, exist_ok=True)

    schedule = load_csv_local_or_url(LOCAL_SCHEDULE, FALLBACK_SCHEDULE_URL)
    schedule = ensure_schedule_columns(schedule)
    team_stats = load_csv_local_or_url(LOCAL_TEAM_STATS, FALLBACK_TEAM_STATS_URL)

    team_stats = team_stats.drop_duplicates(subset=['game_id', 'team'])
    
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
    for bc in base_cols:
        if bc not in schedule.columns:
            schedule[bc] = np.nan
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

    feat_cols = diff_cols + [f"home_R{LAST_N}_count", f"away_R{LAST_N}_count"] + ENG_FEATURES_BASE + LINE_FEATURES + ["elo_home_prob"]
    X["_season"] = pd.to_numeric(X["season"], errors="coerce")
    for c in feat_cols:
        if c in X.columns:
            if c in ["neutral_site","is_postseason"]:
                X[c] = X[c].fillna(0.0); continue
            X[c] = pd.to_numeric(X[c], errors="coerce")
            m = X.groupby("_season")[c].transform("mean")
            X[c] = X[c].fillna(m)
    X = X.drop(columns=["_season"])

    params = fit_market_mapping(X["spread_home"].to_numpy(dtype=float), X["home_win"].to_numpy(dtype=float))
    a, b = params["a"], params["b"]
    X["market_home_prob"] = X["spread_home"].apply(lambda s: (1/(1+np.exp(-(a + b * (-(s))))) if pd.notna(s) else np.nan))
    X["market_home_prob"] = X.groupby("season")["market_home_prob"].transform(lambda s: s.fillna(s.mean()))

    feature_cols = diff_cols + [f"home_R{LAST_N}_count", f"away_R{LAST_N}_count"] + ENG_FEATURES_BASE + LINE_FEATURES + ["elo_home_prob","market_home_prob"]
    feature_cols = [col for col in feature_cols if col in X.columns]

    train_df = X.dropna(subset=["home_points","away_points"]).copy()

    # --- NEW: Final, robust data cleaning and type conversion ---
    # This loop guarantees every feature column is a clean numeric type.
    for col in feature_cols:
        if col in train_df.columns:
            train_df[col] = pd.to_numeric(train_df[col], errors='coerce').astype('float32')

    # As a final safeguard, drop any rows that still have nulls in feature columns
    train_df.dropna(subset=feature_cols, inplace=True)
    # --- END NEW SECTION ---

    train_df.to_parquet(TRAIN_PARQUET, index=False)

    meta = {
        "generated": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "last_n": LAST_N,
        "features": feature_cols,
        "market_params": params,
    }
    save_json(META_JSON, meta)
    print(f"Wrote {TRAIN_PARQUET} and {META_JSON}")

if __name__ == "__main__":
    main()
