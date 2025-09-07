# scripts/train_model.py

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

DERIVED = "data/derived"
RAW_SCHED = "data/raw/cfbd/cfb_schedule.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
FEATURES_PATH = os.path.join(DERIVED, "feature_list.json")
TRAIN_PATH = os.path.join(DERIVED, "training.parquet")


def _load_training():
    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(f"Training matrix not found at {TRAIN_PATH}. Did build_dataset run successfully?")
    df = pd.read_parquet(TRAIN_PATH)

    # If labels missing, try to merge from raw schedule cache
    need_labels = ("home_points" not in df.columns) or ("away_points" not in df.columns)
    if need_labels:
        if os.path.exists(RAW_SCHED):
            sched = pd.read_csv(RAW_SCHED)
            if "id" in sched.columns and "game_id" not in sched.columns:
                sched = sched.rename(columns={"id": "game_id"})
            for c in ["home_points", "away_points"]:
                if c not in sched.columns:
                    sched[c] = pd.NA
                sched[c] = pd.to_numeric(sched[c], errors="coerce")
            df = df.merge(sched[["game_id", "home_points", "away_points"]], on="game_id", how="left")
        else:
            print("[TRAIN] WARNING: raw schedule not found; cannot attach labels. Training will be skipped if labels remain NaN.")

    return df


def _target_columns_ok(df: pd.DataFrame) -> bool:
    return ("home_points" in df.columns) and ("away_points" in df.columns)


def main():
    print("Training single model system...")

    os.makedirs(MODEL_DIR, exist_ok=True)

    df = _load_training()
    print(f"[TRAIN] training.parquet shape={df.shape} cols={list(df.columns)[:25]}")

    if not _target_columns_ok(df):
        raise KeyError("Expected 'home_points' and 'away_points' in training data. Fix features/build to include labels.")

    # Keep only rows with completed games (labels present)
    if "home_points" not in df.columns or "away_points" not in df.columns:
        raise KeyError("Labels still missing after attempted merge from schedule.")

    train_df = df[df["home_points"].notna() & df["away_points"].notna()].copy()

    if train_df.empty:
        raise ValueError("No completed games with labels found. Ensure your raw schedule cache includes historical results.")

    # Example binary target: did the home team win?
    train_df["home_win"] = (train_df["home_points"] > train_df["away_points"]).astype(int)

    # Load feature list if produced; otherwise infer from df (exclude ids/labels)
    features = None
    if os.path.exists(FEATURES_PATH):
        try:
            with open(FEATURES_PATH, "r") as f:
                features = json.load(f)
        except Exception as e:
            print(f"[TRAIN] WARNING: failed to read {FEATURES_PATH}: {e}")

    if not features:
        blacklist = {"game_id", "team", "season", "week", "home_points", "away_points", "home_win"}
        features = [c for c in train_df.columns if c not in blacklist and pd.api.types.is_numeric_dtype(train_df[c])]

    if not features:
        raise ValueError("No numeric features available to train on after exclusions.")

    X = train_df[features].fillna(0.0).values
    y = train_df["home_win"].values

    # Simple baseline: logistic regression
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = LogisticRegression(max_iter=200, n_jobs=None, solver="lbfgs")
    clf.fit(X_train, y_train)

    # Eval
    prob = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, prob)
    print(f"[TRAIN] Validation AUC: {auc:.3f}")

    # Save model + feature list
    joblib.dump({"model": clf, "features": features}, MODEL_PATH)
    print(f"[TRAIN] Saved model to {MODEL_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
