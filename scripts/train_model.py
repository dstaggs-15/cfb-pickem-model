import pandas as pd
import joblib
import json
import os
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

DERIVED = "data/derived"
TRAIN_PARQUET = f"{DERIVED}/training.parquet"
MODEL_JOBLIB = f"{DERIVED}/model.joblib"
META_JSON = "docs/data/train_meta.json"

def main():
    print("Training single model system...")
    os.makedirs(DERIVED, exist_ok=True)

    df = pd.read_parquet(TRAIN_PARQUET)
    with open(META_JSON, 'r') as f:
        meta = json.load(f)
    
    features = meta["features"]
    train_df = df[df["home_points"].notna()].copy()
    
    X_full, y_full = train_df[features], train_df["home_win"]

    print("  Training final model on all data...")
    # Use a very high penalty to force the model to be balanced
    base_model = HistGradientBoostingClassifier(random_state=42, l2_regularization=20.0)
    base_model.fit(X_full, y_full)
    
    calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv='prefit')
    calibrated_model.fit(X_full, y_full)
    
    model_payload = {
        'model': calibrated_model,
        'base_estimator': base_model
    }
    
    joblib.dump(model_payload, MODEL_JOBLIB)
    print(f"Wrote model payload to {MODEL_JOBLIB}")

if __name__ == "__main__":
    main()
