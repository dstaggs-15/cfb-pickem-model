import pandas as pd
import numpy as np
import json
import joblib
import os
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score

DERIVED = "data/derived"
TRAIN_PARQUET = f"{DERIVED}/training.parquet"
MODEL_JOBLIB = f"{DERIVED}/model.joblib"
META_JSON = "docs/data/train_meta.json"
METRICS_JSON = "docs/data/train_metrics.json"

def train_and_calibrate_model(df, features):
    """Helper function to train a single calibrated model on a given feature set."""
    if not features:
        print("Warning: No features provided to train_and_calibrate_model. Skipping.")
        return None
        
    X_full, y_full = df[features], df["home_win"]
    
    if X_full.empty:
        print("Warning: DataFrame is empty. Skipping model training.")
        return None

    base_model = HistGradientBoostingClassifier(random_state=42, l2_regularization=10.0)
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

    fundamentals_features = meta["fundamentals_features"]
    stats_features = meta["stats_features"]

    print("  Training Fundamentals Model...")
    fundamentals_model = train_and_calibrate_model(train_df, fundamentals_features)

    print("  Training Stats Model...")
    stats_model = train_and_calibrate_model(train_df, stats_features)

    # --- THIS SECTION IS NOW CORRECT ---
    # It correctly retrieves the base_estimator from the fundamentals_model to save.
    base_estimator_for_shap = None
    if fundamentals_model and hasattr(fundamentals_model, 'estimator'):
        base_estimator_for_shap = fundamentals_model.estimator

    model_payload = {
        'fundamentals_model': fundamentals_model,
        'stats_model': stats_model,
        'fundamentals_features': fundamentals_features,
        'stats_features': stats_features,
        'base_estimator': base_estimator_for_shap # Correctly save the base estimator
    }
    # --- END CORRECTION ---
    
    joblib.dump(model_payload, MODEL_JOBLIB)
    print(f"Wrote two-model payload to {MODEL_JOBLIB}")

if __name__ == "__main__":
    main()
