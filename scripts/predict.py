#!/usr/bin/env python3
# scripts/predict.py

import os
import sys
import json
import types
import pandas as pd
import numpy as np
import joblib

# ---- NumPy 1.x -> 2.x pickle compatibility shim ---------------------------
try:
    import numpy.random as _npr
    if hasattr(_npr, "PCG64") and "numpy.random._pcg64" not in sys.modules:
        _shim = types.ModuleType("numpy.random._pcg64")
        _shim.PCG64 = _npr.PCG64
        sys.modules["numpy.random._pcg64"] = _shim
except Exception:
    pass
# ---------------------------------------------------------------------------

from .lib.io_utils import load_csv_local_or_url, save_json
from .lib.parsing import parse_games_txt, load_aliases, ensure_schedule_columns
from .lib.features import create_feature_set

# Paths
DERIVED = "data/derived"
LOCAL_DIR = "data/raw/cfbd"
MODEL_JOBLIB = f"{DERIVED}/model.joblib"
META_JSON = "docs/data/train_meta.json"
PREDICTIONS_JSON = "docs/data/predictions.json"
GAMES_TXT = "docs/input/games.txt"
ALIASES_JSON = "docs/input/aliases.json"

LOCAL_SCHEDULE = f"{LOCAL_DIR}/cfb_schedule.csv"
RAW_BASE = "https://raw.githubusercontent.com/moneyball-ab/cfb-data/master/csv"
FALLBACK_SCHEDULE_URL = f"{RAW_BASE}/cfb_schedule.csv"


def _load_model_payload(path: str):
    try:
        payload = joblib.load(path)
        return payload
    except Exception as e:
        msg = (
            f"[PREDICT] Failed to load {path}: {e}\n"
            "Likely version mismatch (NumPy/Sklearn). Pin deps and/or retrain."
        )
        raise RuntimeError(msg) from e


def _resolve_model_and_base(payload):
    """
    Accept either:
      - dict-like payload: {'model': estimator, 'base_estimator': optional}
      - bare estimator object: CalibratedClassifierCV / HistGradientBoostingClassifier, etc.
    Return (model, base_estimator_or_None)
    """
    if isinstance(payload, dict):
        model = payload["model"]
        base = payload.get("base_estimator")
        return model, base

    # Bare estimator; try to detect a calibrated wrapper
    model = payload
    base = getattr(model, "base_estimator_", None) or getattr(model, "base_estimator", None)
    return model, base


def _load_features_list():
    # Prefer meta file; otherwise try model.feature_names_in_
    if os.path.exists(META_JSON):
        with open(META_JSON, "r") as f:
            meta = json.load(f)
        feats = meta.get("features")
        market_params = meta.get("market_params", {})
        if feats:
            return feats, market_params
    # Fallback: we’ll fill features later from model.feature_names_in_
    return None, {}


def main():
    print("Generating predictions...")

    # Load model payload (dict or estimator)
    payload = _load_model_payload(MODEL_JOBLIB)
    model, base_estimator = _resolve_model_and_base(payload)

    # Load features metadata (or fallback)
    features, market_params = _load_features_list()

    # Load schedule and requested games
    schedule = load_csv_local_or_url(LOCAL_SCHEDULE, FALLBACK_SCHEDULE_URL)
    schedule = ensure_schedule_columns(schedule)

    aliases = load_aliases(ALIASES_JSON)
    games_to_predict = parse_games_txt(GAMES_TXT, aliases)

    if not games_to_predict:
        save_json(PREDICTIONS_JSON, [])
        print("No games listed in docs/input/games.txt. Wrote empty predictions.")
        return

    predict_df = pd.DataFrame(games_to_predict)

    # Build features with the simplified API (no kwargs supported)
    X, _ = create_feature_set()

    # Ensure identifiers present for output (merge from schedule if needed)
    id_cols = ["game_id", "season", "week", "home_team", "away_team"]
    if "neutral_site" in schedule.columns:
        id_cols.append("neutral_site")
    sched_small = schedule[id_cols].drop_duplicates("game_id")

    merge_keys = [k for k in ["game_id", "season", "week"] if k in X.columns and k in sched_small.columns]
    if not merge_keys:
        merge_keys = ["game_id"]

    X = X.merge(sched_small, on=merge_keys, how="left")
    if "neutral_site" not in X.columns:
        X["neutral_site"] = 0

    # Optional: filter to requested games if predict list contains team names
    if {"home_team", "away_team"}.issubset(predict_df.columns):
        want = predict_df[["home_team", "away_team"]].drop_duplicates()
        before = len(X)
        X = X.merge(want, on=["home_team", "away_team"], how="inner")
        print(f"Filtered feature matrix to requested games: {before} -> {len(X)} rows")

    if X.empty:
        save_json(PREDICTIONS_JSON, [])
        print("No matching games after filtering; wrote empty predictions.")
        return

    # Market-implied home win prob from spread (if configured)
    if market_params and "a" in market_params and "b" in market_params and "home_closing_spread" in X.columns:
        a, b = market_params["a"], market_params["b"]
        X["market_home_prob"] = X["home_closing_spread"].apply(
            lambda s: (1.0 / (1.0 + np.exp(-(a + b * (-(s))))) if pd.notna(s) else np.nan)
        )
    else:
        X["market_home_prob"] = 0.5

    # Resolve feature list for the model
    if features is None:
        # Try to infer from the model object if available
        if hasattr(model, "feature_names_in_"):
            features = list(model.feature_names_in_)
            print(f"[PREDICT] Using model.feature_names_in_ (n={len(features)})")
        else:
            # Brutal fallback: use all non-identifier columns except obvious labels
            exclude = {"game_id", "season", "week", "home_points", "away_points",
                       "home_team", "away_team", "neutral_site"}
            features = [c for c in X.columns if c not in exclude]
            print(f"[PREDICT] Using all columns minus identifiers as features (n={len(features)})")

    # Ensure model features exist and are numeric
    for col in features:
        if col not in X.columns:
            X[col] = 0.0
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)

    # Predict
    probs = model.predict_proba(X[features])[:, 1]

    # Build output JSON (no SHAP/explanations)
    output = []
    for i in range(len(X)):
        prob = float(probs[i])
        home_team = X.get("home_team").iloc[i] if "home_team" in X.columns else None
        away_team = X.get("away_team").iloc[i] if "away_team" in X.columns else None
        neutral_site = bool(X.get("neutral_site").iloc[i]) if "neutral_site" in X.columns else False
        pick = home_team if prob > 0.5 else away_team

        output.append(
            {
                "home_team": home_team,
                "away_team": away_team,
                "neutral_site": neutral_site,
                "model_prob_home": prob,
                "pick": pick,
            }
        )

    save_json(PREDICTIONS_JSON, output)
    print(f"Successfully wrote {len(output)} predictions to {PREDICTIONS_JSON}")


if __name__ == "__main__":
    main()
