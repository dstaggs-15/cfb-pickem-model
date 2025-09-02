import pandas as pd
import numpy as np
import json
import joblib
import os
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score

# Define file paths
DERIVED = "data/derived"
TRAIN_PARQUET = f"{DERIVED}/training.parquet"
MODEL_JOBLIB = f"{DERIVED}/model.joblib"
META_JSON = "docs/data/train_meta.json"
METRICS_JSON = "docs/data/train_metrics.json"

def season_ahead(df, feats):
    """
    Trains and calibrates a model using season-ahead cross-validation.
    For each season, it trains on all prior seasons, calibrates on the current,
    and tests on the next, providing a robust out-of-sample performance metric.
    """
    seasons = sorted(pd.to_numeric(df["season"], errors="coerce").dropna().unique().tolist())
    metrics = []
    
    # We can't evaluate the last season, so we iterate up to the second to last
    for i, season in enumerate(seasons[:-1]):
        print(f"  Processing season {season} ...")
        
        # Train on all data up to the current season
        train_df = df[df["season"] < season]
        
        # Calibrate on the current season
        calibrate_df = df[df["season"] == season]
        
        # Test on the next season
        test_df = df[df["season"] == season + 1]

        if len(train_df) == 0 or len(calibrate_df) == 0 or len(test_df) == 0:
            continue

        X_train, y_train = train_df[feats], train_df["home_win"]
        X_cal, y_cal = calibrate_df[feats], calibrate_df["home_win"]
        X_test, y_test = test_df[feats], test_df["home_win"]

        # Base model (uncalibrated)
        base_model = HistGradientBoostingClassifier(random_state=42)
        base_model.fit(X_train, y_train)

        # Calibrated model
        calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv='prefit')
        calibrated_model.fit(X_cal, y_cal)

        # Evaluate on the test set
        preds = calibrated_model.predict_proba(X_test)[:, 1]
        
        metrics.append({
            'season': season + 1,
            'auc': roc_auc_score(y_test, preds),
            'brier': brier_score_loss(y_test, preds),
            'accuracy': accuracy_score(y_test, preds > 0.5)
        })

    # Final model: train on all available historical data
    print("  Training final model on all data ...")
    X_full, y_full = df[feats], df["home_win"]
    final_base_model = HistGradientBoostingClassifier(random_state=42)
    final_base_model.fit(X_full, y_full)

    # Use the last complete season for calibration data
    last_season_df = df[df["season"] == seasons[-1]]
    X_final_cal, y_final_cal = last_season_df[feats], last_season_df["home_win"]
    
    final_calibrated_model = CalibratedClassifierCV(final_base_model, method='isotonic', cv='prefit')
    final_calibrated_model.fit(X_final_cal, y_final_cal)
    
    return final_calibrated_model, metrics

def main():
    print("Training model ...")
    os.makedirs(DERIVED, exist_ok=True)

    df = pd.read_parquet(TRAIN_PARQUET)
    with open(META_JSON, 'r') as f:
        meta = json.load(f)
    
    feats = meta["features"]
    
    # Filter for games with results to be used in training
    train_df = df[df["home_points"].notna()].copy()
    
    model, metrics = season_ahead(train_df, feats)
    
    # Save the final trained model
    joblib.dump(model, MODEL_JOBLIB)
    print(f"Wrote model to {MODEL_JOBLIB}")

    # Save the metrics
    if metrics:
        metrics_df = pd.DataFrame(metrics)
        avg_metrics = metrics_df.mean().to_dict()
        print("Season-ahead metrics (avg):")
        for k, v in avg_metrics.items():
            print(f"  {k}: {v:.4f}")
        
        with open(METRICS_JSON, 'w') as f:
            json.dump({'season_ahead_avg': avg_metrics}, f, indent=2)
        print(f"Wrote metrics to {METRICS_JSON}")

if __name__ == "__main__":
    main()
