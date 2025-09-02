import os
import json
from io import StringIO
from typing import Dict
import pandas as pd
import requests

def load_csv_local_or_url(local_path: str, fallback_url: str) -> pd.DataFrame:
    """
    Try reading a CSV from disk; if missing, fetch from a URL.
    Raises on HTTP errors; returns an empty DataFrame only if both fail.
    """
    if os.path.exists(local_path):
        try:
            return pd.read_csv(local_path)
        except Exception as e:
            print(f"[WARN] Failed to read {local_path}: {e}. Falling back to URL...")
    if not fallback_url:
        print(f"[WARN] No local file and no fallback URL for {local_path}")
        return pd.DataFrame()
    r = requests.get(fallback_url, timeout=60)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text))

def save_json(path: str, obj: dict):
    """
    Write pretty JSON, creating parent directories as needed.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def load_json(path: str) -> dict:
    """
    Read JSON if present; otherwise return {}.
    """
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)
