import pandas as pd
import json
import joblib
import os
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

# File paths
DERIVED = "data/derived"
TRAIN_PARQUET = f"{DERIVED}/training.parquet"
META_JSON = "docs/data/train_meta.json"
MODEL_JOBLIB = f"{DERIVED}/model.joblib"

def main():
    print("Training model system...")
    
    # Load the prepared training data and metadata
    train_df = pd.read_parquet(TRAIN_PARQUET)
    with open(META_JSON, 'r') as f:
        meta = json.load(f)
    
    features = meta['features']
    target = 'home_win'

    X_train = train_df[features]
    y_train = train_df[target]

    # --- Model Definition ---
    # A heavily regularized gradient boosting model to prevent overfitting
    # and encourage a balanced use of all features.
    base_estimator = HistGradientBoostingClassifier(
        l2_regularization=20.0,
        max_iter=200,
        learning_rate=0.05,
        random_state=42
    )

    # --- Calibration ---
    # This step ensures the model's probabilities are reliable.
    # A 70% prediction should correspond to a ~70% win rate historically.
    model = CalibratedClassifierCV(
        base_estimator,
        method='isotonic',
        cv='prefit' # Use the already-fitted base estimator
    )
    
    print("  Training final model on all data...")
    base_estimator.fit(X_train, y_train)
    model.fit(X_train, y_train)
    
    # Save the entire payload: the final calibrated model and the base estimator for SHAP
    model_payload = {
        'model': model,
        'base_estimator': base_estimator
    }
    
    joblib.dump(model_payload, MODEL_JOBLIB)
    print(f"Wrote model payload to {MODEL_JOBLIB}")

if __name__ == "__main__":
    main()
