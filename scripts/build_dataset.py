#!/usr/bin/env python3
# scripts/build_dataset.py

import os
import sys
import json
import traceback
import pandas as pd

# IMPORTANT: use absolute package import, not relative
from scripts.lib.features import create_feature_set


def _log_df(name, df: pd.DataFrame, cols: int = 15, rows: int = 5) -> None:
    """Safe logging for small previews of DataFrames."""
    try:
        if isinstance(df, pd.DataFrame):
            shape = (len(df), len(df.columns))
            print(f"[BUILD_DATASET] {name}: shape={shape}, columns(sample)={list(df.columns)[:cols]}")
            if not df.empty:
                print(f"[BUILD_DATASET] {name}: head({rows}):")
                print(df.head(rows).to_string(index=False))
        else:
            print(f"[BUILD_DATASET] {name}: not a DataFrame")
    except Exception as e:
        print(f"[BUILD_DATASET] {name}: <failed to log: {e}>", file=sys.stderr)


def main() -> int:
    print("Building training dataset...")
    try:
        print("  Calculating and saving season average stats...")
        print("  Creating feature set...")

        # NOTE: create_feature_set no longer accepts use_cache/predict_only.
        # It builds directly from data/raw/cfbd/*.csv and returns (X, feature_list).
        X, feature_list = create_feature_set()

        _log_df("X (features)", X)
        print(f"[BUILD_DATASET] feature_list (len={len(feature_list)}): {feature_list[:25]}{' ...' if len(feature_list) > 25 else ''}")

        os.makedirs("data/derived", exist_ok=True)
        X.to_parquet("data/derived/training.parquet", index=False)
        with open("data/derived/feature_list.json", "w") as f:
            json.dump(feature_list, f, indent=2)

        print("Done: training.parquet and feature_list.json written to data/derived/")
        return 0

    except Exception as e:
        print("  Creating feature set... (FAILED)")
        print("[BUILD_DATASET] ERROR:", str(e))
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
