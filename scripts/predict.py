# scripts/predict.py

import pandas as pd
import numpy as np
import json
import joblib
import os

from .lib.io_utils import load_csv_local_or_url, save_json
from .lib.parsing import load_aliases, parse_games_txt
from .lib.rolling import long_stats_to_wide, build_sidewise_rollups
from .lib.context import rest_and_travel
from .lib.market import median_lines
from .lib.elo import pregame_probs

# Define file paths for data and models
DERIVED = "data/derived"
MODEL_JOBLIB = f"{DERIVED}/model.joblib"
META_JSON = "docs/data/train_meta.json"
PREDICTIONS_JSON = "docs/data/predictions.json"

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

# User inputs
GAMES_TXT = "docs/input/games.txt"
ALIASES_JSON = "docs/input/aliases.json"
MANUAL_LINES_CSV = "docs/input/lines.csv"

def main():
    print("Generating predictions ...")
    
    # Load model, metadata, and historical data
    model = joblib.load(MODEL_JOBLIB)
    with open(META_JSON, 'r') as f:
        meta = json.load(f)
    
    feats = meta["features"]
    last_n = meta["last_n"]
    market_params = meta["market_params"]

    schedule = load_csv_local_or_url(LOCAL_SCHEDULE, FALLBACK_SCHEDULE_URL)
    team_stats = load_csv_local_or_url(LOCAL_TEAM_STATS, FALLBACK_TEAM_STATS_URL)
    venues_df = pd.read_csv(LOCAL_VENUES) if os.path.exists(LOCAL_VENUES) else pd.DataFrame()
    teams_df = pd.read_csv(LOCAL_TEAMS) if os.path.exists(LOCAL_TEAMS) else pd.DataFrame()
    talent_df = pd.read_csv(LOCAL_TALENT) if os.path.exists(LOCAL_TALENT) else pd.DataFrame()
    lines_df = pd.read_csv(LOCAL_LINES) if os.path.exists(LOCAL_LINES) else pd.DataFrame()
    manual_lines_df = pd.read_csv(MANUAL_LINES_CSV) if os.path.exists(MANUAL_LINES_CSV) else pd.DataFrame()

    # --- FIX STARTS HERE ---
    # Perform the exact same data cleaning on team_stats as in build_dataset.py
    # 1. Remove duplicate stat lines for the same team in the same game.
    team_stats = team_stats.drop_duplicates(subset=['game_id', 'team'])
    
    # 2. Create the 'home_away' column by cross-referencing the schedule.
    home_team_map = schedule[['game_id', 'home_team']]
    team_stats = team_stats.merge(home_team_map, on='game_id', how='left')
    team_stats['home_away'] = np.where(team_stats['team'] == team_stats['home_team'], 'home', 'away')
    team_stats = team_stats.drop(columns=['home_team'])
    # --- FIX ENDS HERE ---

    # Load and parse user input games
    aliases = load_aliases(ALIASES_JSON)
    games_to_predict = parse_games_txt(GAMES_TXT, aliases)
    
    if not games_to_predict:
        print("No games found in games.txt. Exiting.")
        save_json(PREDICTIONS_JSON, [])
        return

    predict_df = pd.DataFrame(games_to_predict)
    predict_df['game_id'] = [f"predict_{i}" for i in range(len(predict_df))]
    predict_df['season'] = schedule['season'].max()

    # --- Feature Engineering (ensuring train/predict parity) ---
    
    wide_stats = long_stats_to_wide(team_stats)
    home_roll, away_roll = build_sidewise_rollups(schedule, wide_stats, last_n, predict_df)
    
    X = predict_df.merge(home_roll, left_on=["game_id", "home_team"], right_on=["game_id", "team"], how="left").drop(columns=["team"])
    X = X.merge(away_roll, left_on=["game_id", "away_team"], right_on=["game_id", "team"], how="left").drop(columns=["team"])

    diff_cols = []
    for c in [f for f in feats if f.startswith('diff_')]:
        stat_name = c.replace(f'diff_R{last_n}_', '')
        hc, ac = f"home_R{last_n}_{stat_name}", f"away_R{last_n}_{stat_name}"
        if hc in X.columns and ac in X.columns:
            X[c] = X[hc] - X[ac]
            diff_cols.append(c)

    eng = rest_and_travel(schedule, teams_df, venues_df, predict_df)
    X = X.merge(eng, on="game_id", how="left")

    elo_df = pregame_probs(schedule, talent_df, predict_df)
    X = X.merge(elo_df, on="game_id", how="left")

    if not manual_lines_df.empty:
        # Create a mapping for easier lookup
        manual_lines_map = {}
        for _, row in manual_lines_df.iterrows():
            key = (row['home'], row['away'])
            manual_lines_map[key] = row['spread']

        predict_df['spread_home'] = predict_df.apply(
            lambda row: manual_lines_map.get((row['home_team'], row['away_team'])),
            axis=1
        )
        X = X.merge(predict_df[['game_id', 'spread_home']], on='game_id', how='left')
    else:
        med = median_lines(lines_df)
        X = X.merge(med, on=['home_team', 'away_team'], how='left', suffixes=('', '_median'))

    a, b = market_params["a"], market_params["b"]
    X["market_home_prob"] = X["spread_home"].apply(lambda s: (1/(1+np.exp(-(a + b * (-(s))))) if pd.notna(s) else np.nan))

    for c in feats:
        if c not in X.columns:
            X[c] = np.nan
        X[c] = pd.to_numeric(X[c], errors='coerce')
        # A more robust imputation would use saved means from the training set
        # For simplicity, we use the column's own mean if any values are present
        if X[c].isnull().any():
             X[c] = X[c].fillna(X[c].mean())

    X_predict = X[feats].fillna(0) # Final fillna(0) for any remaining NaNs

    probs = model.predict_proba(X_predict)[:, 1]
    
    output = []
    for i, row in X.iterrows():
        prob = probs[i]
        pick = row['home_team'] if prob > 0.5 else row['away_team']
        output.append({
            'home_team': row['home_team'],
            'away_team': row['away_team'],
            'neutral_site': bool(row['neutral_site']),
            'model_prob_home': prob,
            'pick': pick,
            'spread_home': row.get('spread_home')
        })

    save_json(PREDICTIONS_JSON, output)
    print(f"Successfully wrote {len(output)} predictions to {PREDICTIONS_JSON}")

if __name__ == "__main__":
    main()
