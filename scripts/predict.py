#!/usr/bin/env python3
# scripts/predict.py
#
# Generate matchup predictions from the trained model.
# No SHAP / explanations. Robust to missing optional features.

import os
import json
import joblib
import pandas as pd
import numpy as np

from .lib.io_utils import save_json
from .lib.parsing import parse_games_txt, load_aliases
from .lib.features import create_feature_set

# Paths
DERIVED = "data/derived"
FEATURES_JSON = os.path.join(DERIVED, "feature_list.json")
MODEL_FILE = "models/model.pkl"
PREDICTIONS_JSON = "docs/data/predictions.json"
GAMES_TXT = "docs/input/games.txt"
ALIASES_JSON = "docs/input/aliases.json"

ID_EXCLUDE = {
    "game_id", "season", "week",
    "home_team", "away_team", "neutral_site",
    "home_points", "away_points"
}

def _load_feature_list(X: pd.DataFrame, model) -> list[str]:
    # Prefer saved list
    if os.path.exists(FEATURES_JSON):
        try:
            with open(FEATURES_JSON, "r") as f:
                feats = json.load(f)
            if isinstance(feats, list) and feats:
                print(f"[PREDICT] Using {FEATURES_JSON} (n={len(feats)})")
                return feats
        except Exception:
            pass
    # Next, model's remembered names
    if hasattr(model, "feature_names_in_"):
        feats = list(model.feature_names_in_)
        print(f"[PREDICT] Using model.feature_names_in_ (n={len(feats)})")
        return feats
    # Fallback: all numeric non-ID cols
    feats = [c for c in X.columns if c not in ID_EXCLUDE and pd.api.types.is_numeric_dtype(X[c])]
    print(f"[PREDICT] Using fallback features (n={len(feats)})")
    return feats

def main():
    print("Generating predictions...")

    # Load model
    clf = joblib.load(MODEL_FILE)

    # Parse games to predict
    aliases = load_aliases(ALIASES_JSON)
    games_to_predict = parse_games_txt(GAMES_TXT, aliases)
    if not games_to_predict:
        save_json(PREDICTIONS_JSON, [])
        print("No games to predict; wrote empty predictions.")
        return
    want = pd.DataFrame(games_to_predict)[["home_team", "away_team"]].drop_duplicates()

    # Build feature matrix from raw data
    X_all, _ = create_feature_set()

    # Filter to requested matchups
    before = len(X_all)
    X = X_all.merge(want, on=["home_team", "away_team"], how="inner")
    print(f"Filtered feature matrix to requested games: {before} -> {len(X)} rows")

    if X.empty:
        save_json(PREDICTIONS_JSON, [])
        print("No matching games after filtering; wrote empty predictions.")
        return

    # Optional market feature (default 0.5 if unavailable)
    if "home_closing_spread" in X.columns and "market_home_prob" not in X.columns:
        X["market_home_prob"] = 0.5
    elif "market_home_prob" not in X.columns:
        X["market_home_prob"] = 0.5

    # Decide which features to use
    features = _load_feature_list(X, clf)

    # Ensure required columns exist & are numeric
    for col in features:
        if col not in X.columns:
            X[col] = 0.0
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)

    # Predict probabilities
    probs = clf.predict_proba(X[features])[:, 1]

    # Build output
    out = []
    for prob, row in zip(probs, X.itertuples(index=False)):
        home_team = getattr(row, "home_team")
        away_team = getattr(row, "away_team")
        neutral = bool(getattr(row, "neutral_site", 0))
        pick = home_team if float(prob) > 0.5 else away_team
        out.append({
            "home_team": home_team,
            "away_team": away_team,
            "neutral_site": neutral,
            "model_prob_home": float(prob),
            "pick": pick,
        })

    # Save predictions
    os.makedirs(os.path.dirname(PREDICTIONS_JSON), exist_ok=True)
    save_json(PREDICTIONS_JSON, out)
    print(f"Successfully wrote {len(out)} predictions to {PREDICTIONS_JSON}")

if __name__ == "__main__":
    main()
