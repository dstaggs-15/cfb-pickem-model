#!/usr/bin/env python3
import os, json
import pandas as pd
from io import StringIO
import requests

# Paths
LOCAL_DIR = "data/raw/cfbd"
LOCAL_SCHEDULE = f"{LOCAL_DIR}/cfb_schedule.csv"
LOCAL_LINES = f"{LOCAL_DIR}/cfb_lines.csv"

RAW_BASE = "https://raw.githubusercontent.com/moneyball-ab/cfb-data/master/csv"
FALLBACK_SCHEDULE_URL = f"{RAW_BASE}/cfb_schedule.csv"
FALLBACK_LINES_URL = f"{RAW_BASE}/cfb_lines.csv"

PREDICTIONS_JSON = "docs/data/predictions.json"
MANUAL_LINES_CSV = "docs/input/lines.csv"
ALIASES_JSON = "docs/input/aliases.json"
SPREADS_OUT = "docs/data/spreads.json"

def load_csv_local_or_url(local_path: str, fallback_url: str) -> pd.DataFrame:
    if os.path.exists(local_path):
        return pd.read_csv(local_path)
    r = requests.get(fallback_url, timeout=60)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text))

def load_aliases():
    base = {
        "ohio state": "Ohio State Buckeyes", "texas": "Texas Longhorns",
        "northwestern": "Northwestern Wildcats", "tulane": "Tulane Green Wave",
        "lsu": "LSU Tigers", "clemson": "Clemson Tigers",
        "utep": "UTEP Miners", "utah state": "Utah State Aggies",
        "fresno state": "Fresno State Bulldogs", "georgia southern": "Georgia Southern Eagles",
        "arizona": "Arizona Wildcats", "hawaii": "Hawai'i Rainbow Warriors",
        "hawai'i": "Hawai'i Rainbow Warriors", "utah": "Utah Utes",
        "ucla": "UCLA Bruins", "south carolina": "South Carolina Gamecocks",
        "virginia": "Virginia Cavaliers", "oregon": "Oregon Ducks",
        "california": "California Golden Bears", "notre dame": "Notre Dame Fighting Irish",
        "miami": "Miami (FL) Hurricanes", "miami (fl)": "Miami (FL) Hurricanes",
    }
    if os.path.exists(ALIASES_JSON):
        try:
            with open(ALIASES_JSON, "r") as f:
                extra = json.load(f)
            for k, v in extra.items():
                base[k.strip().lower()] = v.strip()
        except Exception as e:
            print(f"[WARN] Could not read aliases.json: {e}")
    return base

def norm(name: str, alias_map) -> str:
    if not isinstance(name, str): return ""
    key = name.strip().lower()
    return alias_map.get(key, name.strip())

def main():
    if not os.path.exists(PREDICTIONS_JSON):
        raise SystemExit(f"{PREDICTIONS_JSON} not found; run train first.")

    with open(PREDICTIONS_JSON, "r") as f:
        preds = json.load(f)

    alias_map = load_aliases()

    # Load schedule + lines (prefer local, fallback to public snapshot)
    schedule = load_csv_local_or_url(LOCAL_SCHEDULE, FALLBACK_SCHEDULE_URL)
    lines_all = None
    if os.path.exists(MANUAL_LINES_CSV):
        try:
            man = pd.read_csv(MANUAL_LINES_CSV)
            man.columns = [c.strip().lower() for c in man.columns]
            for c in ["spread","over_under"]:
                if c in man.columns:
                    man[c] = pd.to_numeric(man[c], errors="coerce")
            # Normalize names
            if "home" in man.columns: man["home"] = man["home"].apply(lambda x: norm(x, alias_map))
            if "away" in man.columns: man["away"] = man["away"].apply(lambda x: norm(x, alias_map))
            lines_all = man
        except Exception as e:
            print(f"[WARN] Could not read manual lines: {e}")

    if lines_all is None:
        lines_all = load_csv_local_or_url(LOCAL_LINES, FALLBACK_LINES_URL)

    # Ensure we have convenient schedule cols
    for c in ["season","week","home_team","away_team","game_id","date"]:
        if c not in schedule.columns:
            schedule[c] = None
    # Cast
    schedule["season"] = pd.to_numeric(schedule["season"], errors="coerce")
    schedule["week"] = pd.to_numeric(schedule["week"], errors="coerce")
    # Use latest season present
    latest_season = int(schedule["season"].dropna().max())

    # Build quick lookup: (home,away)->game_id for latest season
    sched_now = schedule[schedule["season"]==latest_season].copy()
    # If multiple rows (neutral-site changes, etc.), take the latest by 'date' or highest week
    if "date" in sched_now.columns:
        sched_now["date"] = pd.to_datetime(sched_now["date"], errors="coerce", utc=True)
    sched_now = sched_now.sort_values(["week","date"], na_position="last")
    key_rows = sched_now.dropna(subset=["home_team","away_team"])[["home_team","away_team","game_id"]].drop_duplicates(keep="last")
    key_rows["home_team"] = key_rows["home_team"].astype(str).str.strip()
    key_rows["away_team"] = key_rows["away_team"].astype(str).str.strip()

    # Lines prep
    df_lines = pd.DataFrame()
    used_manual = False
    if {"home","away","spread","over_under"}.issubset(set(lines_all.columns)):
        # Manual format
        df_lines = lines_all.copy()
        used_manual = True
    else:
        # CFBD lines format → median per game_id
        tmp = lines_all.copy()
        for old, new in [("spread","spread"), ("overUnder","over_under"), ("overunder","over_under")]:
            if old in tmp.columns and new not in tmp.columns:
                tmp[new] = tmp[old]
        tmp["spread"] = pd.to_numeric(tmp.get("spread"), errors="coerce")
        tmp["over_under"] = pd.to_numeric(tmp.get("over_under"), errors="coerce")
        if "game_id" in tmp.columns:
            df_lines = tmp.groupby("game_id")[["spread","over_under"]].median().reset_index()

    # Build output aligned to predictions.json order
    out_rows = []
    for g in preds.get("games", []):
        home = norm(g["home"], alias_map)
        away = norm(g["away"], alias_map)
        spread_val = None
        ou_val = None

        if used_manual:
            row = df_lines[(df_lines["home"]==home) & (df_lines["away"]==away)]
            if not row.empty:
                spread_val = float(pd.to_numeric(row.iloc[0]["spread"], errors="coerce"))
                ou_val = float(pd.to_numeric(row.iloc[0]["over_under"], errors="coerce"))
        else:
            gid = key_rows[(key_rows["home_team"]==home) & (key_rows["away_team"]==away)]["game_id"]
            if not gid.empty:
                sgid = gid.iloc[0]
                row = df_lines[df_lines["game_id"]==sgid]
                if not row.empty:
                    spread_val = float(pd.to_numeric(row.iloc[0]["spread"], errors="coerce"))
                    ou_val = float(pd.to_numeric(row.iloc[0]["over_under"], errors="coerce"))

        favorite = None
        market_line = None
        if spread_val is not None and not pd.isna(spread_val):
            if spread_val < 0:
                favorite = home
                market_line = f"{home} {abs(spread_val):g}"
            elif spread_val > 0:
                favorite = away
                market_line = f"{away} {abs(spread_val):g}"
            else:
                favorite = None
                market_line = "Pick’em"

        out_rows.append({
            "home": home,
            "away": away,
            "spread_home": spread_val if spread_val is not None else None,
            "over_under": ou_val if ou_val is not None else None,
            "favorite": favorite,
            "market_line": market_line
        })

    os.makedirs(os.path.dirname(SPREADS_OUT), exist_ok=True)
    with open(SPREADS_OUT, "w") as f:
        json.dump({"season": latest_season, "games": out_rows}, f, indent=2)
    print(f"Wrote {SPREADS_OUT} with {len(out_rows)} rows")

if __name__ == "__main__":
    main()
