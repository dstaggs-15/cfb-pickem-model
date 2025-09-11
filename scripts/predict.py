#!/usr/bin/env python3
# scripts/predict.py
# Generate matchup predictions. Applies optional Platt calibrator if present.

import os, json
import joblib
import numpy as np
import pandas as pd

from .lib.io_utils import save_json
from .lib.parsing import parse_games_txt, load_aliases
from .lib.features import create_feature_set

DERIVED = "data/derived"
FEATURES_JSON = os.path.join(DERIVED, "feature_list.json")
MODEL_FILE = "models/model.pkl"
PREDICTIONS_JSON = "docs/data/predictions.json"
GAMES_TXT = "docs/input/games.txt"
ALIASES_JSON = "docs/input/aliases.json"
CALIB_PATH = os.path.join(DERIVED, "calibrator.json")

ID_EXCLUDE = {
    "game_id","season","week","home_team","away_team",
    "neutral_site","home_points","away_points"
}
EPS = 1e-6

def _load_feature_list(X: pd.DataFrame, model) -> list[str]:
    # prefer saved list
    if os.path.exists(FEATURES_JSON):
        try:
            with open(FEATURES_JSON, "r") as f:
                feats = json.load(f)
            if isinstance(feats, list) and feats:
                print(f"[PREDICT] Using {FEATURES_JSON} (n={len(feats)})")
                return feats
        except Exception:
            pass
    # else model's names
    if hasattr(model, "feature_names_in_"):
        feats = list(model.feature_names_in_)
        print(f"[PREDICT] Using model.feature_names_in_ (n={len(feats)})")
        return feats
    # fallback: all numeric non-ID
    feats = [c for c in X.columns if c not in ID_EXCLUDE and pd.api.types.is_numeric_dtype(X[c])]
    print(f"[PREDICT] Using fallback features (n={len(feats)})")
    return feats

def _dedup_latest(X: pd.DataFrame) -> pd.DataFrame:
    keep_cols = list(X.columns)
    sort_cols = [c for c in ["season","week","game_id"] if c in X.columns]
    if not sort_cols:
        return X.drop_duplicates(["home_team","away_team"], keep="last")
    Xs = X.sort_values(sort_cols).drop_duplicates(["home_team","away_team"], keep="last")
    return Xs[keep_cols]

def _load_calibrator(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            j = json.load(f)
        if j.get("type") != "platt":  # ignore unknown types
            return None
        w = float(j.get("w", 1.0))
        b = float(j.get("b", 0.0))
    except Exception:
        return None
    def apply(p):
        p = np.clip(np.asarray(p, dtype=float), EPS, 1-EPS)
        z = np.log(p/(1-p))
        q = 1/(1+np.exp(-(w*z + b)))
        return q
    print(f"[PREDICT] Calibrator loaded: w={w:.3f}, b={b:.3f}")
    return apply

def main():
    print("Generating predictions...")

    # Load model
    clf = joblib.load(MODEL_FILE)

    # Parse target matchups
    aliases = load_aliases(ALIASES_JSON)
    games_to_predict = parse_games_txt(GAMES_TXT, aliases)
    if not games_to_predict:
        save_json(PREDICTIONS_JSON, [])
        print("No games to predict; wrote empty predictions.")
        return
    want = pd.DataFrame(games_to_predict)[["home_team","away_team"]].drop_duplicates()

    # Build feature matrix & filter to desired games
    X_all, _ = create_feature_set()
    before = len(X_all)
    X = X_all.merge(want, on=["home_team","away_team"], how="inner")
    print(f"Filtered feature matrix to requested games: {before} -> {len(X)} rows")
    X = _dedup_latest(X)
    print(f"[PREDICT] After dedup: {len(X)} rows")

    if X.empty:
        save_json(PREDICTIONS_JSON, [])
        print("No matching games after filtering/dedup; wrote empty predictions.")
        return

    # Feature selection & sanitation
    feats = _load_feature_list(X, clf)
    for col in feats:
        if col not in X.columns:
            X[col] = 0.0
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)

    # Raw model probabilities
    p_raw = clf.predict_proba(X[feats])[:, 1].astype(float)

    # Optional calibration
    calib = _load_calibrator(CALIB_PATH)
    p_final = calib(p_raw) if calib is not None else p_raw

    # Build output
    out = []
    for prob_raw, prob_final, row in zip(p_raw, p_final, X.itertuples(index=False)):
        home_team = getattr(row, "home_team")
        away_team = getattr(row, "away_team")
        neutral = bool(getattr(row, "neutral_site", 0))
        pick = home_team if float(prob_final) >= 0.5 else away_team
        out.append({
            "home_team": home_team,
            "away_team": away_team,
            "neutral_site": neutral,
            "model_prob_home_raw": float(prob_raw),
            "model_prob_home": float(prob_final),
            "pick": pick,
        })

    os.makedirs(os.path.dirname(PREDICTIONS_JSON), exist_ok=True)
    save_json(PREDICTIONS_JSON, out)
    print(f"Successfully wrote {len(out)} predictions to {PREDICTIONS_JSON}")

if __name__ == "__main__":
    main()
