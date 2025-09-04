import pandas as pd
import numpy as np
import json
import joblib
import os
import shap

from .lib.io_utils import load_csv_local_or_url, save_json
from .lib.parsing import parse_games_txt, load_aliases, ensure_schedule_columns
from .lib.features import create_feature_set

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
LOCAL_VENUES = f"{LOCAL_DIR}/cfb_venues.csv"
LOCAL_TEAMS = f"{LOCAL_DIR}/cfb_teams.csv"
LOCAL_TALENT = f"{LOCAL_DIR}/cfb_talent.csv"
RAW_BASE = "https://raw.githubusercontent.com/moneyball-ab/cfb-data/master/csv"
FALLBACK_SCHEDULE_URL = f"{RAW_BASE}/cfb_schedule.csv"
FALLBACK_TEAM_STATS_URL = f"{RAW_BASE}/cfb_game_team_stats.csv"

def main():
    print("Generating predictions...")
    
    # Load model and metadata
    model_payload = joblib.load(MODEL_JOBLIB)
    model = model_payload['model']
    base_estimator = model_payload.get('base_estimator')

    with open(META_JSON, 'r') as f:
        meta = json.load(f)
    features = meta["features"]
    market_params = meta.get("market_params", {})

    # Load raw data
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
        return

    predict_df = pd.DataFrame(games_to_predict)
    predict_df['game_id'] = [f"predict_{i}" for i in range(len(predict_df))]
    predict_df['season'] = schedule['season'].max()

    # Create features for the games to predict using the shared function
    X, _ = create_feature_set(schedule, team_stats_long, venues_df, teams_df, talent_df, lines_df, games_to_predict_df=predict_df)

    # Add market features using saved parameters
    if market_params and 'a' in market_params and 'b' in market_params:
        a, b = market_params["a"], market_params["b"]
        X["market_home_prob"] = X["spread_home"].apply(lambda s: (1/(1+np.exp(-(a + b * (-(s))))) if pd.notna(s) else np.nan))
    else:
        X["market_home_prob"] = 0.5

    # Final cleaning
    for col in features:
        if col not in X.columns:
            X[col] = 0.0
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0.0)
    
    # Predict
    probs = model.predict_proba(X[features])[:, 1]

    # SHAP Explanations
    shap_values = None
    if base_estimator:
        print("  Generating SHAP explanations...")
        train_df = pd.read_parquet(TRAIN_PARQUET)
        # Ensure training data has same columns in same order for explainer
        train_df_features = train_df[features]
        explainer = shap.TreeExplainer(base_estimator, train_df_features)
        shap_values = explainer.shap_values(X[features])
    
    output = []
    for i, row in X.iterrows():
        prob = probs[i]
        pick = row['home_team'] if prob > 0.5 else row['away_team']
        
        explanation = []
        if shap_values is not None:
            shap_row = shap_values[i]
            feature_names = X[features].columns
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

if __name__ == "__main__":
    main()
