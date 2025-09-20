import pandas as pd
import numpy as np
import json
import joblib
import os
import sys
import shap

from .lib.io_utils import save_json
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
LOCAL_VENUES = f"{LOCAL_DIR}/cfbd_venues.csv"
LOCAL_TEAMS = f"{LOCAL_DIR}/cfbd_teams.csv"
LOCAL_TALENT = f"{LOCAL_DIR}/cfbd_talent.csv"

def main():
    print("Generating predictions...")
    
    # --- 1. Load Model and Metadata ---
    if not os.path.exists(MODEL_JOBLIB) or not os.path.exists(META_JSON):
        print("FATAL: Model files not found. Please run the training workflow first.")
        sys.exit(1)
        
    model_payload = joblib.load(MODEL_JOBLIB)
    model = model_payload['model']
    base_estimator = model_payload.get('base_estimator')

    with open(META_JSON, 'r') as f:
        meta = json.load(f)
    features = meta["features"]
    market_params = meta.get("market_params", {})

    # --- 2. Load Raw Data for Prediction Context ---
    schedule = pd.read_csv(LOCAL_SCHEDULE, low_memory=False)
    team_stats = pd.read_csv(LOCAL_TEAM_STATS) if os.path.exists(LOCAL_TEAM_STATS) else pd.DataFrame()
    lines_df = pd.read_csv(LOCAL_LINES) if os.path.exists(LOCAL_LINES) else pd.DataFrame()
    venues_df = pd.read_csv(LOCAL_VENUES) if os.path.exists(LOCAL_VENUES) else pd.DataFrame()
    teams_df = pd.read_csv(LOCAL_TEAMS) if os.path.exists(LOCAL_TEAMS) else pd.DataFrame()
    talent_df = pd.read_csv(LOCAL_TALENT) if os.path.exists(LOCAL_TALENT) else pd.DataFrame()
    manual_lines_df = pd.read_csv(MANUAL_LINES_CSV) if os.path.exists(MANUAL_LINES_CSV) else pd.DataFrame()

    # --- 3. Prepare Games to Predict ---
    aliases = load_aliases(ALIASES_JSON)
    games_to_predict = parse_games_txt(GAMES_TXT, aliases)

    if not games_to_predict:
        save_json(PREDICTIONS_JSON, [])
        return

    predict_df = pd.DataFrame(games_to_predict)
    predict_df['game_id'] = [f"predict_{i}" for i in range(len(predict_df))]
    predict_df['season'] = schedule['season'].max()
    
    # --- 4. Create Features for Prediction Games ---
    # This now uses the exact same logic as the training script, ensuring consistency.
    X, _ = create_feature_set(
        schedule=schedule, team_stats=team_stats, venues_df=venues_df,
        teams_df=teams_df, talent_df=talent_df, lines_df=lines_df,
        manual_lines_df=manual_lines_df, games_to_predict_df=predict_df
    )

    # Add market features using parameters learned from training
    if market_params and 'a' in market_params and 'b' in market_params:
        a, b = market_params["a"], market_params["b"]
        X["market_home_prob"] = 1.0 / (1.0 + np.exp(-(a + b * (-X["spread_home"]))))
    else:
        X["market_home_prob"] = 0.5

    # Fill any missing feature columns with 0, which is safe.
    for col in features:
        if col not in X.columns:
            X[col] = 0.0
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0.0)
    
    # --- 5. Generate Predictions and Explanations ---
    probs = model.predict_proba(X[features])[:, 1]

    shap_values = None
    if base_estimator:
        print("  Generating SHAP explanations...")
        train_df = pd.read_parquet(TRAIN_PARQUET)
        # Ensure training columns match prediction columns exactly
        train_df_features = train_df[features]
        explainer = shap.TreeExplainer(base_estimator, train_df_features)
        shap_values = explainer.shap_values(X[features])
    
    output = []
    for i in range(len(X)):
        prob = probs[i]
        home_team = X['home_team'].iloc[i]
        away_team = X['away_team'].iloc[i]
        neutral_site = bool(X['neutral_site'].iloc[i])
        pick = home_team if prob > 0.5 else away_team
        
        explanation = []
        if shap_values is not None:
            shap_row = shap_values[i]
            feature_names = X[features].columns
            explanation = sorted(
                [{'feature': name, 'value': val} for name, val in zip(feature_names, shap_row)],
                key=lambda x: abs(x['value']),
                reverse=True
            )

        output.append({
            'home_team': home_team, 'away_team': away_team, 'neutral_site': neutral_site,
            'model_prob_home': prob, 'pick': pick, 'explanation': explanation
        })

    save_json(PREDICTIONS_JSON, output)
    print(f"Successfully wrote {len(output)} predictions to {PREDICTIONS_JSON}")

if __name__ == "__main__":
    main()

