# scripts/lib/io_utils.py
# makes scripts.lib a package
# makes scripts a package (useful for IDEs and -m runs)

import os, json
from io import StringIO
from typing import Dict
import pandas as pd
import requests

def load_csv_local_or_url(local_path: str, fallback_url: str) -> pd.DataFrame:
    if os.path.exists(local_path):
        return pd.read_csv(local_path)
    r = requests.get(fallback_url, timeout=60)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text))

def save_json(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)
