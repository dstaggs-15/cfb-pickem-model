#!/usr/bin/env python3
# scripts/train_model.py
#
# Trains a simple classifier. Reads CSV first; Parquet optional.

from __future__ import annotations
import os, json
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

DERIVED_DIR = "data/derived"
TRAIN_PARQUET = os.path.join(DERIVED_DIR, "training.parquet")
TRAIN_CSV = os.path.join(DERIVED_DIR, "training.csv")
FEATURES_JSON = os.path.join(DERIVED_DIR, "feature_list.json")
MODEL_PKL = "models/model.pkl"

ID_COLS = {
    "game_id","season","week","home_team","away_team",
    "home_points","away_points","neutral_site"
}

def _load_training() -> pd.DataFrame:
    if os.path.exists(TRAIN_CSV):
        df = pd.read_csv(TRAIN_CSV)
        print(f"[TRAIN] loaded CSV {TRAIN_CSV} shape={df.shape}")
        return df
    if os.path.exists(TRAIN_PARQUET):
        df = pd.read_parquet(TRAIN_PARQUET)
        print(f"[TRAIN] loaded Parquet {TRAIN_PARQUET} shape={df.shape}")
        return df
    raise FileNotFoundError("No training file found (training.csv or training.parquet). Run build_dataset first.")

def _load_features(df: pd.DataFrame) -> list[str]:
    if os.path.exists(FEATURES_JSON):
        with open(FEATURES_JSON, "r") as f:
            feats = json.load(f)
        print(f"[TRAIN] using feature_list.json (n={len(feats)})")
        return feats
    # fallback: all numeric non-ID columns
    numeric = [c for c in df.columns if c not in ID_COLS and pd.api.types.is_numeric_dtype(df[c])]
    print(f"[TRAIN] using fallback features (n={len(numeric)})")
    return numeric

def main() -> int:
    os.makedirs(os.path.dirname(MODEL_PKL), exist_ok=True)
    df = _load_training()

    # target: home win
    df["home_points"] = pd.to_numeric(df.get("home_points"), errors="coerce")
    df["away_points"] = pd.to_numeric(df.get("away_points"), errors="coerce")
    y = (df["home_points"] > df["away_points"]).astype(int)

    features = _load_features(df)
    X = df[features].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    base = HistGradientBoostingClassifier(
        max_depth=None,
        learning_rate=0.08,
        max_iter=350,
        l2_regularization=0.0,
        random_state=42
    )
    clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
    clf.fit(X_train, y_train)

    try:
        p = clf.predict_proba(X_valid)[:, 1]
        auc = roc_auc_score(y_valid, p)
        print(f"[TRAIN] validation AUC: {auc:.4f}")
    except Exception as e:
        print(f"[TRAIN] WARN could not score validation: {e}")

    joblib.dump(clf, MODEL_PKL)
    print(f"[TRAIN] wrote {MODEL_PKL}")

    # Save features for predict.py
    with open(os.path.join(DERIVED_DIR, "feature_list.json"), "w") as f:
        json.dump(features, f, indent=2)
    print(f"[TRAIN] wrote {FEATURES_JSON}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
