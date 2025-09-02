# scripts/lib/parsing.py
import re, pandas as pd, numpy as np, datetime as dt, json, os
from typing import Dict, List

BUILTIN_ALIASES = {
    "hawaii": "Hawai'i Rainbow Warriors",
    "hawai'i": "Hawai'i Rainbow Warriors",
    "miami": "Miami (FL) Hurricanes",
    "miami (fl)": "Miami (FL) Hurricanes",
    "ole miss": "Ole Miss Rebels",
    "utep": "UTEP Miners",
    "utsa": "UTSA Roadrunners",
}

GAMES_PATTERNS = [
    re.compile(r"^\s*(?P<away>.+?)\s*@\s*(?P<home>.+?)\s*$", re.I),                   # Away @ Home
    re.compile(r"^\s*(?P<home>.+?)\s*vs\.?\s*(?P<away>.+?)(?:\s*\(N\))?\s*$", re.I),  # Home vs Away (N)
    re.compile(r"^\s*(?P<home>.+?)\s*,\s*(?P<away>.+?)\s*$", re.I),                   # Home, Away
]

def to_int(x, default=0):
    try: return int(float(x))
    except Exception: return default

def to_dt(x):
    try: return pd.to_datetime(x, utc=True)
    except Exception: return pd.NaT

def ensure_schedule_columns(schedule: pd.DataFrame) -> pd.DataFrame:
    df = schedule.rename(columns=lambda c: c.strip())
    for c in ["season","week"]:
        df[c] = df[c].apply(to_int) if c in df.columns else 0
    date_col = None
    for cand in ["date","startDate","start_date","game_date","startTime","start_time"]:
        if cand in df.columns: date_col = cand; break
    df["date"] = df[date_col].apply(to_dt) if date_col else pd.NaT
    if "season_type" not in df.columns: df["season_type"] = "regular"
    if "neutral_site" not in df.columns: df["neutral_site"] = False
    if "venue_id" not in df.columns: df["venue_id"] = np.nan
    if "home_points" not in df.columns: df["home_points"] = np.nan
    if "away_points" not in df.columns: df["away_points"] = np.nan
    if "game_id" not in df.columns:
        df["game_id"] = pd.util.hash_pandas_object(
            df[["season","week","home_team","away_team"]].fillna(""),
            index=False
        ).astype(np.int64)
    return df

def load_alias_map(path: str) -> Dict[str,str]:
    alias = dict(BUILTIN_ALIASES)
    if os.path.exists(path):
        try:
            extra = json.load(open(path, "r"))
            for k, v in extra.items():
                alias[str(k).strip().lower()] = str(v).strip()
        except Exception:
            pass
    return alias

def normalize_name(name: str, alias_map: Dict[str,str]) -> str:
    if not name: return ""
    key = name.strip().lower()
    return alias_map.get(key, name.strip())

def parse_games_txt(path: str, alias_map: Dict[str,str]) -> List[dict]:
    out = []
    if not os.path.exists(path): return out
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"): continue
            neutral = "(N)" in line or " (n)" in line
            m = None
            for pat in GAMES_PATTERNS:
                mm = pat.match(line)
                if mm: m = mm.groupdict(); break
            if not m: continue
            home = normalize_name(" ".join(m["home"].split()), alias_map)
            away = normalize_name(" ".join(m["away"].split()), alias_map)
            out.append({"home": home, "away": away, "neutral": neutral})
    return out

def parse_ratio_val(val):
    import math
    if not isinstance(val, str):
        return float(val) if pd.notna(val) else np.nan
    s = val.strip().lower().replace("–","-").replace("—","-")
    m = re.match(r"^\s*(\d+)\s*(?:[-/]|of|for)\s*(\d+)\s*$", s)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        return a / b if b else np.nan
    try:
        return float(val)
    except Exception:
        return np.nan
