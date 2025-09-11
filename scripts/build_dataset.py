#!/usr/bin/env python3
# scripts/build_dataset.py
#
# Builds training dataset (CSV first; Parquet optional if pyarrow/fastparquet present).

from __future__ import annotations
import os, json
import pandas as pd
import numpy as np

from .lib.features import create_feature_set

RAW_DIR = "data/raw/cfbd"
DERIVED_DIR = "data/derived"

SCHED_CSV = os.path.join(RAW_DIR, "cfb_schedule.csv")
TRAIN_PARQUET = os.path.join(DERIVED_DIR, "training.parquet")
TRAIN_CSV = os.path.join(DERIVED_DIR, "training.csv")
FEATURES_JSON = os.path.join(DERIVED_DIR, "feature_list.json")
SEASON_AVG_CSV = os.path.join(DERIVED_DIR, "season_team_avgs.csv")

def _ensure_dirs():
    os.makedirs(DERIVED_DIR, exist_ok=True)

def _calc_and_save_season_avgs() -> None:
    if not os.path.exists(SCHED_CSV):
        print(f"[BUILD_DATASET] WARN: schedule not found: {SCHED_CSV}")
        return
    df = pd.read_csv(SCHED_CSV)
    # build long
    home = df.rename(columns={
        "home_team": "team",
        "away_team": "opponent",
        "home_points": "points_for",
        "away_points": "points_against",
    })[["season","team","opponent","points_for","points_against"]].copy()
    away = df.rename(columns={
        "away_team": "team",
        "home_team": "opponent",
        "away_points": "points_for",
        "home_points": "points_against",
    })[["season","team","opponent","points_for","points_against"]].copy()
    long = pd.concat([home, away], ignore_index=True)
    for c in ["points_for","points_against"]:
        long[c] = pd.to_numeric(long[c], errors="coerce")
    long["margin"] = long["points_for"] - long["points_against"]
    grp = long.groupby(["season","team"], as_index=False).agg(
        pf_mean=("points_for","mean"),
        pa_mean=("points_against","mean"),
        margin_mean=("margin","mean"),
        games=("points_for","count"),
    )
    os.makedirs(DERIVED_DIR, exist_ok=True)
    grp.to_csv(SEASON_AVG_CSV, index=False)
    print(f"[BUILD_DATASET] wrote {SEASON_AVG_CSV} (rows={len(grp)})")

def _try_save_parquet(df: pd.DataFrame, path: str) -> bool:
    try:
        df.to_parquet(path, index=False)
        print(f"[BUILD_DATASET] wrote {path} (parquet)")
        return True
    except Exception as e:
        print(f"[BUILD_DATASET] parquet unavailable ({e}); skipping parquet.")
        return False

def main() -> int:
    print("Building training dataset...")
    _ensure_dirs()

    print("  Calculating and saving season average stats...")
    try:
        _calc_and_save_season_avgs()
    except Exception as e:
        print(f"[BUILD_DATASET] WARN season avgs failed: {e}")

    print("  Creating feature set...")
    try:
        X, feature_list = create_feature_set()
        print(f"[BUILD_DATASET] X (features): shape={X.shape}, columns(sample)={list(X.columns)[:15]}")
        print("[BUILD_DATASET] X (features): head(5):")
        try:
            print(X.head(5).to_string(index=False))
        except Exception:
            print(X.head(5))

        print(f"[BUILD_DATASET] feature_list (len={len(feature_list)}): {feature_list[:18]}")

        # Save CSV (primary)
        X.to_csv(TRAIN_CSV, index=False)
        print(f"[BUILD_DATASET] wrote {TRAIN_CSV} (csv)")

        # Save Parquet (optional)
        _try_save_parquet(X, TRAIN_PARQUET)

        # Save feature list
        with open(FEATURES_JSON, "w") as f:
            json.dump(feature_list, f, indent=2)
        print(f"[BUILD_DATASET] wrote {FEATURES_JSON}")

    except Exception as e:
        print("  Creating feature set... (FAILED)")
        print(f"[BUILD_DATASET] ERROR: {e}")
        return 1

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
