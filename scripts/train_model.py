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
    seasons = sorted(pd.to_numeric(df["season"], errors="coerce").dropna().unique().tolist())
    metrics = []
    
    for i, season in enumerate(seasons[:-1]):
        print(f"  Processing season {season} ...")
        
        train_df = df[df["season"] < season]
        calibrate_df = df[df["season"] == season]
        test_df = df[df["season"] == season + 1]

        if len(train_df) == 0 or len(calibrate_df) == 0 or len(test_df) == 0:
            continue

        X_train, y_train = train_df[feats], train_df["home_win"]
        X_cal, y_cal = calibrate_df[feats], calibrate_df["home_win"]
        X_test, y_test = test_df[feats], test_df["home_win"]

        # --- MODIFIED: Increased regularization penalty ---
        base_model = HistGradientBoostingClassifier(random_state=42, l2_regularization=10.0)
        base_model.fit(X_train, y_train)

        calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv='prefit')
        calibrated_model.fit(X_cal, y_cal)

        preds = calibrated_model.predict_proba(X_test)[:, 1]
        
        metrics.append({
            'season': season + 1,
            'auc': roc_auc_score(y_test, preds),
            'brier': brier_score_loss(y_test, preds),
            'accuracy': accuracy_score(y_test, preds > 0.5)
        })

    print("  Training final model on all data ...")
    X_full, y_full = df[feats], df["home_win"]
    
    # --- MODIFIED: Increased regularization penalty ---
    final_base_model = HistGradientBoostingClassifier(random_state=42, l2_regularization=10.0)
    final_base_model.fit(X_full, y_full)

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
    
    train_df = df[df["home_points"].notna()].copy()
    
    calibrated_model, metrics = season_ahead(train_df, feats)
    
    model_payload = {
        'calibrated_model': calibrated_model,
        'base_model': calibrated_model.estimator 
    }
    joblib.dump(model_payload, MODEL_JOBLIB)
    print(f"Wrote model payload to {MODEL_JOBLIB}")
    
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
