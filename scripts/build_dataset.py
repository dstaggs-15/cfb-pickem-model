#!/usr/bin/env python3

import json, datetime as dt
import os
import sys
import numpy as np
import pandas as pd

from .lib.io_utils import save_json
from .lib.features import create_feature_set
from .lib.market import fit_market_mapping

# File paths
DERIVED = "data/derived"
TRAIN_PARQUET = f"{DERIVED}/training.parquet"
META_JSON = "docs/data/train_meta.json"
# Note: The season_averages.parquet is now likely handled inside create_feature_set

def main():
    """
    This script orchestrates the creation of the final training dataset.
    It now calls a self-contained feature engineering function and then
    processes and validates the results to create the final training files.
    """
    print("Building training dataset...")
    os.makedirs(DERIVED, exist_ok=True)
    os.makedirs(os.path.dirname(META_JSON), exist_ok=True)

    # --- 1. Build the Full Feature Set ---
    print("  Creating feature set for all historical games...")
    X, feature_list = create_feature_set()

    # --- 2. Add Labels and Market-Derived Features ---
    home_points = pd.to_numeric(X["home_points"], errors='coerce')
    away_points = pd.to_numeric(X["away_points"], errors='coerce')
    X["home_win"] = (home_points > away_points).astype(int)

    params = {}
    if 'spread_home' in X.columns and X['spread_home'].notna().any():
        clean_market_data = X[['spread_home', 'home_win']].dropna()
        spreads = clean_market_data['spread_home']
        labels = clean_market_data['home_win']
        
        if not spreads.empty:
            params = fit_market_mapping(spreads, labels) or {}
            if 'a' in params and 'b' in params:
                a, b = float(params["a"]), float(params["b"])
                X["market_home_prob"] = 1.0 / (1.0 + np.exp(-(a + b * (-X["spread_home"]))))
    
    if "market_home_prob" not in X.columns:
        X["market_home_prob"] = np.nan
        
    X["market_home_prob"] = X.groupby("season")["market_home_prob"].transform(lambda s: s.fillna(s.mean()))

    # --- 3. Finalize and Clean Data ---
    extra = ["spread_home", "over_under", "market_home_prob"]
    final_feature_list = [f for f in (feature_list + extra) if f in X.columns]
    
    train_df = X[X['home_points'].notna()].copy()
    
    for col in final_feature_list:
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce').fillna(0.0).astype('float32')

    # --- 4. CRITICAL VALIDATION STEP ---
    # This safeguard prevents a "dumb" model from ever being trained.
    # It checks if the engineered features have any predictive power.
    print("  Validating feature matrix...")
    feature_variance = train_df[final_feature_list].var()
    if feature_variance.sum() < 1e-6:
        print("\nCRITICAL ERROR: Feature matrix has no variance.")
        print("This means all feature values are zero, which will result in a useless model.")
        print("This is likely due to an error in the feature engineering or data fetching process.")
        sys.exit(1) # Fail the workflow with a non-zero exit code
    else:
        print("  Validation passed: Features contain meaningful data.")

    # --- 5. Save Final Training Files ---
    train_df.to_parquet(TRAIN_PARQUET, index=False)

    meta = {
        "generated": dt.datetime.utcnow().isoformat(),
        "features": final_feature_list,
        "market_params": params
    }
    save_json(META_JSON, meta)
    print(f"Wrote {len(train_df)} rows to {TRAIN_PARQUET} and updated {META_JSON}")

if __name__ == "__main__":
    main()

