import pandas as pd
import numpy as np
import json
import joblib
import os
import shap

# (Imports and file paths are the same as the last working version)
from .lib.io_utils import load_csv_local_or_url, save_json
from .lib.parsing import parse_games_txt # We will update parsing.py separately
from .lib.rolling import long_stats_to_wide, build_sidewise_rollups, STAT_FEATURES
from .lib.context import rest_and_travel
from .lib.market import median_lines
from .lib.elo import pregame_probs
# ... all file paths ...
DERIVED = "data/derived"
#... etc ...
PREDICTIONS_JSON = "docs/data/predictions.json"
MODEL_JOBLIB = f"{DERIVED}/model.joblib"
#... etc ...

def main():
    print("Generating predictions with two-model system...")
    # ... (all data loading is the same as the last working version) ...
    
    # --- THIS SECTION IS THE SAME AS THE LAST WORKING BUILD_DATASET.PY ---
    # It must be duplicated here to ensure features are created identically.
    schedule = load_csv_local_or_url(...)
    team_stats_long = load_csv_local_or_url(...)
    # ... all other data files ...
    print("  Pivoting and cleaning raw team stats...")
    # ... pivot logic ...
    # ... data cleaning logic ...
    # ... feature creation logic for PPA, success_rate, etc. ...
    # ... build home_roll, away_roll, X dataframe ...
    # --- END DUPLICATED SECTION ---

    # --- NEW TWO-MODEL PREDICTION LOGIC ---
    print("  Loading two-model payload...")
    model_payload = joblib.load(MODEL_JOBLIB)
    fundamentals_model = model_payload['fundamentals_model']
    stats_model = model_payload['stats_model']
    fundamentals_features = model_payload['fundamentals_features']
    stats_features = model_payload['stats_features']

    # Ensure all feature columns are present and numeric
    for col in fundamentals_features + stats_features:
        if col not in X.columns:
            X[col] = 0.0
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0.0)

    print("  Generating predictions from both models...")
    fundamentals_probs = fundamentals_model.predict_proba(X[fundamentals_features])[:, 1]
    stats_probs = stats_model.predict_proba(X[stats_features])[:, 1]

    # Blend the predictions (e.g., 60% Fundamentals, 40% Stats)
    blended_probs = (fundamentals_probs * 0.6) + (stats_probs * 0.4)

    # --- SHAP EXPLANATIONS (from the more intuitive fundamentals model) ---
    print("  Generating SHAP explanations...")
    train_df = pd.read_parquet("data/derived/training.parquet") # background data
    explainer = shap.TreeExplainer(fundamentals_model.base_estimator, train_df[fundamentals_features])
    shap_values = explainer.shap_values(X[fundamentals_features])
    
    # ... (Final loop to build JSON output is the same, but uses `blended_probs`) ...
    output = []
    for i, row in X.iterrows():
        prob = blended_probs[i]
        # ... rest of the output generation ...
        output.append({
            # ... all fields ...
            'explanation': # ... formatted shap values from fundamentals_model
        })
    save_json(PREDICTIONS_JSON, output)
    print(f"Successfully wrote {len(output)} predictions to {PREDICTIONS_JSON}")

if __name__ == "__main__":
    main()
