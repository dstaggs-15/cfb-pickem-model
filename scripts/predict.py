#!/usr/bin/env python3
# scripts/predict.py
#
# Generate matchup predictions from the trained model.
# SHAP explanations have been removed to reduce dependencies.

import os
import json
import joblib
import pandas as pd
import numpy as np

from .lib.io_utils import load_csv_local_or_url, save_json  # re-use any helpers you have
from .lib.parsing import parse_games_txt, load_aliases
from .lib.features import create_feature_set

# Paths
DERIVED = "data/derived"
TRAIN_PARQUET = os.path.join(DERIVED, "training.parquet")
FEATURES_JSON = os.path.join(DERIVED, "feature_list.json")
MODEL_FILE = "models/model.pkl"
PREDICTIONS_JSON = "docs/data/predictions.json"
GAMES_TXT = "docs/input/games.txt"
ALIASES_JSON = "docs/input/aliases.json"

def main():
    print("Generating predictions...")

    # Load model
    model = joblib.load(MODEL_FILE)

    # Load feature list if present
    features = None
    if os.path.exists(FEATURES_JSON):
        try:
            with open(FEATURES_JSON, "r") as f:
                features = json.load(f)
        except Exception as e:
            print(f"[PREDICT] WARNING: failed to read {FEATURES_JSON}: {e}")

    # Load alias map and games list
    aliases = load_aliases(ALIASES_JSON)
    games_to_predict = parse_games_txt(GAMES_TXT, aliases)
    if not games_to_predict:
        save_json(PREDICTIONS_JSON, [])
        print("No games to predict; wrote empty predictions.")
        return

    predict_df = pd.DataFrame(games_to_predict)

    # Build feature matrix
    X, _ = create_feature_set()

    # Filter to requested matchups
    if {"home_team", "away_team"}.issubset(predict_df.columns):
        want = predict_df[["home_team","away_team"]].drop_duplicates()
        before = len(X)
        X = X.merge(want, on=["home_team","away_team"], how="inner")
        print(f"Filtered feature matrix to requested games: {before} -> {len(X)} rows")

    if X.empty:
        save_json(PREDICTIONS_JSON, [])
        print("No matching games after filtering; wrote empty predictions.")
        return

    # Market-implied probability (constant 0.5 if lines missing)
    if "home_closing_spread" in X.columns:
        # If you have a calibration, use it; otherwise default to 0.5
        X["market_home_prob"] = 0.5
    else:
        X["market_home_prob"] = 0.5

    # Resolve feature list if still None
    if features is None:
        if hasattr(model, "feature_names_in_"):
            features = list(model.feature_names_in_)
            print(f"[PREDICT] Using model.feature_names_in_ (n={len(features)})")
        else:
            # fallback: all numeric columns except identifiers
            exclude = {"game_id","season","week","home_team","away_team",
                       "home_points","away_points","neutral_site"}
            features = [c for c in X.columns if c not in exclude and pd.api.types.is_numeric_dtype(X[c])]
            print(f"[PREDICT] Using fallback features (n={len(features)})")

    # Ensure numeric and fill NaNs
    for col in features:
        if col not in X.columns:
            X[col] = 0.0
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)

    # Predict probabilities
    probs = model.predict_proba(X[features])[:, 1]

    # Build results
    out = []
    for i in range(len(X)):
        prob = float(probs[i])
        home_team = X["home_team"].iloc[i]
        away_team = X["away_team"].iloc(i) if "away_team" in X.columns else None
        away_team = X["away_team"].iloc[i] if "away_team" in X.columns else None
        neutral = bool(X["neutral_site"].iloc[i]) if "neutral_site" in X.columns else False
        pick = home_team if prob > 0.5 else away_team
        out.append({
            "home_team": home_team,
            "away_team": away_team,
            "neutral_site": neutral,
            "model_prob_home": prob,
            "pick": pick
        })

    # Save predictions
    save_json(PREDICTIONS_JSON, out)
    print(f"Successfully wrote {len(out)} predictions to {PREDICTIONS_JSON}")

if __name__ == "__main__":
    main()
