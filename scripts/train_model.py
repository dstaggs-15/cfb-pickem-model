import pandas as pd
import numpy as np
import json
import joblib
import os
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score

# (File paths are the same)
DERIVED = "data/derived"
TRAIN_PARQUET = f"{DERIVED}/training.parquet"
MODEL_JOBLIB = f"{DERIVED}/model.joblib"
META_JSON = "docs/data/train_meta.json"
METRICS_JSON = "docs/data/train_metrics.json"

def train_and_calibrate_model(df, features):
    """Helper function to train a single calibrated model on a given feature set."""
    # Ensure there are features to train on
    if not features:
        print("Warning: No features provided to train_and_calibrate_model. Skipping.")
        return None
        
    X_full, y_full = df[features], df["home_win"]
    
    # Return if the dataframe is empty
    if X_full.empty:
        print("Warning: DataFrame is empty. Skipping model training.")
        return None

    base_model = HistGradientBoostingClassifier(random_state=42, l2_regularization=2.0)
    base_model.fit(X_full, y_full)
    
    calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv='prefit')
    calibrated_model.fit(X_full, y_full)
    
    return calibrated_model

def main():
    print("Training two-model system ...")
    os.makedirs(DERIVED, exist_ok=True)

    df = pd.read_parquet(TRAIN_PARQUET)
    with open(META_JSON, 'r') as f:
        meta = json.load(f)
    
    train_df = df[df["home_points"].notna()].copy()

    # --- NEW: Load feature lists directly from meta file ---
    fundamentals_features = meta["fundamentals_features"]
    stats_features = meta["stats_features"]

    print("  Training Fundamentals Model...")
    fundamentals_model = train_and_calibrate_model(train_df, fundamentals_features)

    print("  Training Stats Model...")
    stats_model = train_and_calibrate_model(train_df, stats_features)

    model_payload = {
        'fundamentals_model': fundamentals_model,
        'stats_model': stats_model,
        'fundamentals_features': fundamentals_features,
        'stats_features': stats_features
    }
    
    joblib.dump(model_payload, MODEL_JOBLIB)
    print(f"Wrote two-model payload to {MODEL_JOBLIB}")

if __name__ == "__main__":
    main()
