import pandas as pd
import numpy as np
import json
import joblib
import os
import shap
import sys

from .lib.io_utils import load_csv_local_or_url, save_json
from .lib.parsing import parse_games_txt, load_aliases
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

LOCAL_SCHEDULE = f"{LOCAL_DIR}/cfb_schedule.csv"
LOCAL_TEAM_STATS = f"{LOCAL_DIR}/cfb_game_team_stats.csv"
LOCAL_LINES = f"{LOCAL_DIR}/cfb_lines.csv"
LOCAL_VENUES = f"{LOCAL_DIR}/cfbd_venues.csv"
LOCAL_TEAMS = f"{LOCAL_DIR}/cfb_teams.csv"
LOCAL_TALENT = f"{LOCAL_DIR}/cfb_talent.csv"
RAW_BASE = "https://raw.githubusercontent.com/moneyball-ab/cfb-data/master/csv"
FALLBACK_SCHEDULE_URL = f"{RAW_BASE}/cfb_schedule.csv"
FALLBACK_TEAM_STATS_URL = f"{RAW_BASE}/cfb_game_team_stats.csv"

def main():
    print("Generating predictions with two-model system...")
    
    model_payload = joblib.load(MODEL_JOBLIB)
    fundamentals_model = model_payload['fundamentals_model']
    stats_model = model_payload['stats_model']
    fundamentals_features = model_payload['fundamentals_features']
    stats_features = model_payload['stats_features']
    base_estimator = model_payload.get('base_estimator') # Safely get the base estimator

    with open(META_JSON, 'r') as f:
        meta = json.load(f)
    market_params = meta.get("market_params", {})
    last_n = meta["last_n"]

    schedule = load_csv_local_or_url(LOCAL_SCHEDULE, FALLBACK_SCHEDULE_URL)
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

    # (This large feature engineering section remains the same)
    # ...
    
    print("  Generating predictions from both models...")
    fundamentals_probs = fundamentals_model.predict_proba(X[fundamentals_features])[:, 1]
    stats_probs = stats_model.predict_proba(X[stats_features])[:, 1]
    blended_probs = (fundamentals_probs * 0.6) + (stats_probs * 0.4)

    # --- SHAP EXPLANATION LOGIC ---
    shap_values = None
    if base_estimator:
        print("  Generating SHAP explanations...")
        train_df = pd.read_parquet(TRAIN_PARQUET)
        explainer = shap.TreeExplainer(base_estimator, train_df[fundamentals_features])
        shap_values = explainer.shap_values(X[fundamentals_features])
    else:
        print("  Skipping SHAP explanations, base_estimator not found.")
    
    output = []
    for i, row in X.iterrows():
        prob = blended_probs[i]
        pick = row['home_team'] if prob > 0.5 else row['away_team']
        
        explanation = []
        if shap_values is not None:
            shap_row = shap_values[i]
            feature_names = X[fundamentals_features].columns
            
            explanation = sorted(
                [{'feature': name, 'value': val} for name, val in zip(feature_names, shap_row)],
                key=lambda x: abs(x['value']),
                reverse=True
            )[:5]

        output.append({
            'home_team': row['home_team'],
            'away_team': row['away_team'],
            'neutral_site': bool(row['neutral_site']),
            'model_prob_home': prob,
            'pick': pick,
            'explanation': explanation
        })

    save_json(PREDICTIONS_JSON, output)
    print(f"Successfully wrote {len(output)} predictions to {PREDICTIONS_JSON}")

# (Need to add the full feature engineering section for completeness)
# ...
if __name__ == "__main__":
    main()
