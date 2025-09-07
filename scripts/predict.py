#!/usr/bin/env python3
# scripts/predict.py

import os
import json
import joblib
import pandas as pd
import numpy as np
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
LOCAL_LINES = f"{LOCAL_DIR}/cfb_lines.csv"

RAW_BASE = "https://raw.githubusercontent.com/moneyball-ab/cfb-data/master/csv"
FALLBACK_SCHEDULE_URL = f"{RAW_BASE}/cfb_schedule.csv"


def main():
    print("Generating predictions...")

    # --- Load model and metadata ---
    model_payload = joblib.load(MODEL_JOBLIB)
    model = model_payload["model"]
    base_estimator = model_payload.get("base_estimator")

    with open(META_JSON, "r") as f:
        meta = json.load(f)
    features = meta["features"]
    market_params = meta.get("market_params", {})

    # --- Load schedule and requested games ---
    schedule = load_csv_local_or_url(LOCAL_SCHEDULE, FALLBACK_SCHEDULE_URL)
    schedule = ensure_schedule_columns(schedule)

    aliases = load_aliases(ALIASES_JSON)
    games_to_predict = parse_games_txt(GAMES_TXT, aliases)

    if not games_to_predict:
        save_json(PREDICTIONS_JSON, [])
        print("No games listed in docs/input/games.txt. Wrote empty predictions.")
        return

    # Make a DF for requested games if team names are available; used later to filter X.
    predict_df = pd.DataFrame(games_to_predict)
    # We'll try to filter by home/away if those columns exist after we build X.

    # --- Build features with the simplified API ---
    # current create_feature_set() builds from cached raw CSVs; no kwargs supported.
    X, _ = create_feature_set()

    # --- Ensure identifiers present for output (merge from schedule if needed) ---
    id_cols = ["game_id", "season", "week", "home_team", "away_team"]
    if "neutral_site" in schedule.columns:
        id_cols.append("neutral_site")
    sched_small = schedule[id_cols].drop_duplicates("game_id")
    # Merge on keys we know exist in X: "game_id" (and season/week we preserved)
    # If season/week aren't in X for some reason, the left merge on game_id will still work.
    merge_keys = [k for k in ["game_id", "season", "week"] if k in X.columns and k in sched_small.columns]
    if not merge_keys:
        merge_keys = ["game_id"]
    X = X.merge(sched_small, on=merge_keys, how="left")

    # Default neutral_site if missing after merge
    if "neutral_site" not in X.columns:
        X["neutral_site"] = 0

    # Optional: filter to requested games if predict_df has team names
    if {"home_team", "away_team"}.issubset(predict_df.columns):
        want = predict_df[["home_team", "away_team"]].drop_duplicates()
        before = len(X)
        X = X.merge(want, on=["home_team", "away_team"], how="inner")
        print(f"Filtered feature matrix to requested games: {before} -> {len(X)} rows")

    if X.empty:
        save_json(PREDICTIONS_JSON, [])
        print("No matching games after filtering; wrote empty predictions.")
        return

    # --- Market-implied home win prob from spread (if configured) ---
    # Features use "home_closing_spread" (not "spread_home")
    if market_params and "a" in market_params and "b" in market_params and "home_closing_spread" in X.columns:
        a, b = market_params["a"], market_params["b"]
        X["market_home_prob"] = X["home_closing_spread"].apply(
            lambda s: (1.0 / (1.0 + np.exp(-(a + b * (-(s))))) if pd.notna(s) else np.nan)
        )
    else:
        X["market_home_prob"] = 0.5

    # --- Ensure model features exist and are numeric ---
    for col in features:
        if col not in X.columns:
            X[col] = 0.0
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)

    # --- Predict ---
    probs = model.predict_proba(X[features])[:, 1]

    # --- Optional SHAP explanations if base_estimator available ---
    shap_values = None
    if base_estimator is not None and os.path.exists(TRAIN_PARQUET):
        print("  Generating SHAP explanations...")
        train_df = pd.read_parquet(TRAIN_PARQUET)
        train_df_features = train_df[features]
        explainer = shap.TreeExplainer(base_estimator, train_df_features)
        shap_values = explainer.shap_values(X[features])

    # --- Build output JSON ---
    output = []
    n = len(X)
    for i in range(n):
        prob = float(probs[i])

        # These columns now exist because we merged from schedule
        home_team = X.get("home_team").iloc[i] if "home_team" in X.columns else None
        away_team = X.get("away_team").iloc[i] if "away_team" in X.columns else None
        neutral_site = bool(X.get("neutral_site").iloc[i]) if "neutral_site" in X.columns else False

        pick = home_team if prob > 0.5 else away_team

        explanation = []
        if shap_values is not None:
            shap_row = shap_values[i]
            feature_names = X[features].columns
            explanation = sorted(
                [{"feature": name, "value": float(val)} for name, val in zip(feature_names, shap_row)],
                key=lambda x: abs(x["value"]),
                reverse=True,
            )

        output.append(
            {
                "home_team": home_team,
                "away_team": away_team,
                "neutral_site": neutral_site,
                "model_prob_home": prob,
                "pick": pick,
                "explanation": explanation,
            }
        )

    save_json(PREDICTIONS_JSON, output)
    print(f"Successfully wrote {len(output)} predictions to {PREDICTIONS_JSON}")


if __name__ == "__main__":
    main()
