# scripts/predict.py

import pandas as pd
import numpy as np
import json
import joblib
import os
import shap

from .lib.io_utils import load_csv_local_or_url, save_json
from .lib.parsing import parse_games_txt, load_aliases
from .lib.features import create_feature_set

# Paths
DERIVED = "data/derived"
TRAIN_PARQUET = f"{DERIVED}/training.parquet"
MODEL_JOBLIB = f"{DERIVED}/model.joblib"
META_JSON = "docs/data/train_meta.json"
PREDICTIONS_JSON = "docs/data/predictions.json"
GAMES_TXT = "docs/input/games.txt"
ALIASES_JSON = "docs/input/aliases.json"

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

    # Load game aliases and the list of games to predict
    aliases = load_aliases(ALIASES_JSON)
    games_to_predict = parse_games_txt(GAMES_TXT, aliases)
    if not games_to_predict:
        save_json(PREDICTIONS_JSON, [])
        print("No games to predict; wrote empty predictions.")
        return

    predict_df = pd.DataFrame(games_to_predict)

    # Build the full feature matrix from raw schedule + lines (no extra args)
    X, _ = create_feature_set()

    # Filter to only the matchups we care about
    if {"home_team", "away_team"}.issubset(predict_df.columns):
        want = predict_df[["home_team", "away_team"]].drop_duplicates()
        before = len(X)
        X = X.merge(want, on=["home_team", "away_team"], how="inner")
        print(f"Filtered feature matrix to requested games: {before} -> {len(X)} rows")

    if X.empty:
        save_json(PREDICTIONS_JSON, [])
        print("No matching games after filtering; wrote empty predictions.")
        return

    # Market-implied home win probability (use home_closing_spread)
    if (market_params
        and "a" in market_params and "b" in market_params
        and "home_closing_spread" in X.columns):
        a, b = market_params["a"], market_params["b"]
        X["market_home_prob"] = X["home_closing_spread"].apply(
            lambda s: 1.0 / (1.0 + np.exp(-(a + b * (-s)))) if pd.notna(s) else np.nan
        )
    else:
        X["market_home_prob"] = 0.5

    # Ensure all model features exist and are numeric
    for col in features:
        if col not in X.columns:
            X[col] = 0.0
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0.0)

    # Predict probabilities
    probs = model.predict_proba(X[features])[:, 1]

    # Optionally compute SHAP explanations
    shap_values = None
    if base_estimator and os.path.exists(TRAIN_PARQUET):
        print("  Generating SHAP explanations...")
        train_df = pd.read_parquet(TRAIN_PARQUET)
        train_df_features = train_df[features]
        explainer = shap.TreeExplainer(base_estimator, train_df_features)
        shap_values = explainer.shap_values(X[features])

    # Build the output predictions
    output = []
    for i in range(len(X)):
        prob = float(probs[i])
        home_team = X["home_team"].iloc[i]
        away_team = X["away_team"].iloc[i]
        neutral_site = bool(X["neutral_site"].iloc[i]) if "neutral_site" in X.columns else False
        pick = home_team if prob > 0.5 else away_team

        explanation = []
        if shap_values is not None:
            shap_row = shap_values[i]
            feature_names = X[features].columns
            explanation = sorted(
                [
                    {"feature": name, "value": val}
                    for name, val in zip(feature_names, shap_row)
                ],
                key=lambda x: abs(x["value"]),
                reverse=True,
            )

        output.append({
            "home_team": home_team,
            "away_team": away_team,
            "neutral_site": neutral_site,
            "model_prob_home": prob,
            "pick": pick,
            "explanation": explanation
        })

    # Save predictions
    save_json(PREDICTIONS_JSON, output)
    print(f"Successfully wrote {len(output)} predictions to {PREDICTIONS_JSON}")

if __name__ == "__main__":
    main()
