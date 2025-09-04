import pandas as pd
import numpy as np
import json
import joblib
import os
import shap

from .lib.io_utils import load_csv_local_or_url, save_json
from .lib.parsing import parse_games_txt, load_aliases, ensure_schedule_columns
from .lib.rolling import long_stats_to_wide, build_sidewise_rollups, STAT_FEATURES
from .lib.context import rest_and_travel
from .lib.market import median_lines
from .lib.elo import pregame_probs

# File paths
DERIVED = "data/derived"
LOCAL_DIR = "data/raw/cfbd"
TRAIN_PARQUET = f"{DERIVED}/training.parquet"
MODEL_JOBLIB = f"{DERIVED}/model.joblib"
META_JSON = "docs/data/train_meta.json"
PREDICTIONS_JSON = "docs/data/predictions.json"
GAMES_TXT = "docs/input/games.txt"
ALIASES_JSON = "docs/input/aliases.json"
MANUAL_LINES_CSV = "docs/input/lines.csv"

# Re-use the full file paths from build_dataset for consistency
LOCAL_SCHEDULE = f"{LOCAL_DIR}/cfb_schedule.csv"
LOCAL_TEAM_STATS = f"{LOCAL_DIR}/cfb_game_team_stats.csv"
LOCAL_LINES = f"{LOCAL_DIR}/cfb_lines.csv"
LOCAL_VENUES = f"{LOCAL_DIR}/cfb_venues.csv"
LOCAL_TEAMS = f"{LOCAL_DIR}/cfb_teams.csv"
LOCAL_TALENT = f"{LOCAL_DIR}/cfb_talent.csv"
RAW_BASE = "https://raw.githubusercontent.com/moneyball-ab/cfb-data/master/csv"
FALLBACK_SCHEDULE_URL = f"{RAW_BASE}/cfb_schedule.csv"
FALLBACK_TEAM_STATS_URL = f"{RAW_BASE}/cfb_game_team_stats.csv"

def parse_possession_time(s):
    if not isinstance(s, str) or ':' not in s:
        return 0.0
    try:
        minutes, seconds = s.split(':')
        return int(minutes) * 60 + int(seconds)
    except (ValueError, TypeError):
        return 0.0

def main():
    print("Generating predictions with two-model system...")
    
    # --- Load Models and Metadata ---
    model_payload = joblib.load(MODEL_JOBLIB)
    fundamentals_model = model_payload['fundamentals_model']
    stats_model = model_payload['stats_model']
    fundamentals_features = model_payload['fundamentals_features']
    stats_features = model_payload['stats_features']

    with open(META_JSON, 'r') as f:
        meta = json.load(f)
    market_params = meta["market_params"]
    last_n = meta["last_n"]

    # --- Load all necessary data for feature creation ---
    schedule = load_csv_local_or_url(LOCAL_SCHEDULE, FALLBACK_SCHEDULE_URL)
    schedule = ensure_schedule_columns(schedule)
    team_stats_long = load_csv_local_or_url(LOCAL_TEAM_STATS, FALLBACK_TEAM_STATS_URL)
    venues_df = pd.read_csv(LOCAL_VENUES) if os.path.exists(LOCAL_VENUES) else pd.DataFrame()
    teams_df = pd.read_csv(LOCAL_TEAMS) if os.path.exists(LOCAL_TEAMS) else pd.DataFrame()
    talent_df = pd.read_csv(LOCAL_TALENT) if os.path.exists(LOCAL_TALENT) else pd.DataFrame()
    lines_df = pd.read_csv(LOCAL_LINES) if os.path.exists(LOCAL_LINES) else pd.DataFrame()
    manual_lines_df = pd.read_csv(MANUAL_LINES_CSV) if os.path.exists(MANUAL_LINES_CSV) else pd.DataFrame()

    aliases = load_aliases(ALIASES_JSON)
    games_to_predict = parse_games_txt(GAMES_TXT, aliases)

    if not games_to_predict:
        save_json(PREDICTIONS_JSON, [])
        print("No games in input/games.txt. Wrote empty predictions.json.")
        return

    predict_df = pd.DataFrame(games_to_predict)
    predict_df['game_id'] = [f"predict_{i}" for i in range(len(predict_df))]
    predict_df['season'] = schedule['season'].max()

    # --- FULL FEATURE ENGINEERING PIPELINE (Mirrors build_dataset.py) ---
    print("  Pivoting and cleaning raw team stats for prediction...")
    team_stats_long['stat_value'] = pd.to_numeric(team_stats_long['stat_value'], errors='coerce')
    team_stats = team_stats_long.pivot_table(
        index=['game_id', 'team'], columns='category', values='stat_value'
    ).reset_index()

    def camel_to_snake(name):
        return ''.join(['_' + c.lower() if c.isupper() else c for c in name]).lstrip('_')
    team_stats.columns = [camel_to_snake(col) for col in team_stats.columns]

    rushing_attempts = team_stats['rushing_attempts'] if 'rushing_attempts' in team_stats.columns else pd.Series(0, index=team_stats.index)
    pass_attempts = team_stats['pass_attempts'] if 'pass_attempts' in team_stats.columns else pd.Series(0, index=team_stats.index)
    total_yards = team_stats['total_yards'] if 'total_yards' in team_stats.columns else pd.Series(0, index=team_stats.index)
    first_downs = team_stats['first_downs'] if 'first_downs' in team_stats.columns else pd.Series(0, index=team_stats.index)
    yards_per_pass = team_stats['yards_per_pass'] if 'yards_per_pass' in team_stats.columns else pd.Series(0, index=team_stats.index)
    yards_per_rush_attempt = team_stats['yards_per_rush_attempt'] if 'yards_per_rush_attempt' in team_stats.columns else pd.Series(0, index=team_stats.index)
    
    total_plays = rushing_attempts.fillna(0) + pass_attempts.fillna(0)
    
    team_stats['ppa'] = (total_yards.fillna(0) / total_plays.replace(0, np.nan)).fillna(0)
    team_stats['success_rate'] = (first_downs.fillna(0) / total_plays.replace(0, np.nan)).fillna(0)
    team_stats['explosiveness'] = (yards_per_pass.fillna(0) * 0.5 + yards_per_rush_attempt.fillna(0) * 0.5)
    
    if 'possession_time' in team_stats.columns:
        team_stats['possession_seconds'] = team_stats['possession_time'].apply(parse_possession_time)
        team_stats.drop(columns=['possession_time'], inplace=True)
    else:
        team_stats['possession_seconds'] = 0.0

    home_team_map = schedule[['game_id', 'home_team']]
    team_stats = team_stats.merge(home_team_map, on='game_id', how='left')
    team_stats['home_away'] = np.where(team_stats['team'] == team_stats['home_team'], 'home', 'away')
    team_stats = team_stats.drop(columns=['home_team'])
    
    wide_stats = long_stats_to_wide(team_stats)
    home_roll, away_roll = build_sidewise_rollups(schedule, wide_stats, last_n, predict_df)

    X = predict_df.merge(home_roll, on=['game_id', 'home_team'], how='left')
    X = X.merge(away_roll, on=['game_id', 'away_team'], how='left')
    
    for c in STAT_FEATURES:
        hc, ac = f"home_R{last_n}_{c}", f"away_R{last_n}_{c}"
        dc = f"diff_R{last_n}_{c}"
        if hc in X.columns and ac in X.columns:
            X[dc] = X[hc] - X[ac]

    eng = rest_and_travel(schedule, teams_df, venues_df, predict_df)
    X = X.merge(eng, on="game_id", how="left")
    elo_df = pregame_probs(schedule, talent_df, predict_df)
    X = X.merge(elo_df, on="game_id", how="left")

    if not manual_lines_df.empty:
        X = X.merge(manual_lines_df, on=['home_team', 'away_team'], how='left')
    else:
        med = median_lines(lines_df)
        X = X.merge(med, on=['home_team', 'away_team'], how='left')

    a, b = market_params["a"], market_params["b"]
    X["market_home_prob"] = X["spread_home"].apply(lambda s: (1/(1+np.exp(-(a + b * (-(s))))) if pd.notna(s) else np.nan))

    all_features = fundamentals_features + stats_features
    for col in all_features:
        if col not in X.columns:
            X[col] = 0.0
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0.0)
    
    # --- Two-Model Prediction Logic ---
    print("  Generating predictions from both models...")
    fundamentals_probs = fundamentals_model.predict_proba(X[fundamentals_features])[:, 1]
    stats_probs = stats_model.predict_proba(X[stats_features])[:, 1]
    blended_probs = (fundamentals_probs * 0.6) + (stats_probs * 0.4)

    # --- SHAP Explanation Logic ---
    print("  Generating SHAP explanations...")
    train_df = pd.read_parquet(TRAIN_PARQUET)
    explainer = shap.TreeExplainer(fundamentals_model.base_estimator, train_df[fundamentals_features])
    shap_values = explainer.shap_values(X[fundamentals_features])
    
    output = []
    for i, row in X.iterrows():
        prob = blended_probs[i]
        pick = row['home_team'] if prob > 0.5 else row['away_team']
        
        shap_row = shap_values[i]
        feature_names = X[fundamentals_features].columns
        
        explanation = sorted(
            [{'feature': name, 'value': val} for name, val in zip(feature_names, shap_row)],
            key=lambda x: abs(x['value']),
            reverse=True
        )

        output.append({
            'home_team': row['home_team'],
            'away_team': row['away_team'],
            'neutral_site': bool(row['neutral_site']),
            'model_prob_home': prob,
            'pick': pick,
            'explanation': explanation[:5]
        })

    save_json(PREDICTIONS_JSON, output)
    print(f"Successfully wrote {len(output)} predictions to {PREDICTIONS_JSON}")

if __name__ == "__main__":
    main()
