#!/usr/bin/env python3
# scripts/build_dataset.py

import os
import sys
import json
import traceback
import pandas as pd

# IMPORTANT: use absolute package import, not relative
from scripts.lib.features import create_feature_set

def _log_df(name, df, cols=15, rows=5):
    try:
        shape = (len(df), len(df.columns)) if isinstance(df, pd.DataFrame) else ("NA","NA")
        print(f"[BUILD_DATASET] {name}: shape={shape}, columns(sample)={list(df.columns)[:cols] if isinstance(df, pd.DataFrame) else 'NA'}")
        if isinstance(df, pd.DataFrame) and not df.empty:
            print(f"[BUILD_DATASET] {name}: head({rows}):")
            print(df.head(rows).to_string(index=False))
    except Exception as e:
        print(f"[BUILD_DATASET] {name}: <failed to log: {e}>", file=sys.stderr)

def main():
    print("Building training dataset...")
    try:
        print("  Calculating and saving season average stats...")
        print("  Creating feature set...")

        X, feature_list = create_feature_set(
            use_cache=True,
            predict_only=False
        )

        _log_df("X (features)", X)
        print(f"[BUILD_DATASET] feature_list (len={len(feature_list)}): {feature_list[:25]}{' ...' if len(feature_list)>25 else ''}")

        os.makedirs("data/derived", exist_ok=True)
        X.to_parquet("data/derived/training.parquet", index=False)
        with open("data/derived/feature_list.json", "w") as f:
            json.dump(feature_list, f, indent=2)

        print("Done: training.parquet and feature_list.json written to data/derived/")

    except Exception as e:
        print("  Creating feature set... (FAILED)")
        print("[BUILD_DATASET] ERROR:", str(e))
        traceback.print_exc()
        raise

if __name__ == "__main__":
    sys.exit(main())
