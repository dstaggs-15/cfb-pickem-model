#!/usr/bin/env python3
"""
Build docs/data/spreads.json for the website.

Order of precedence for lines:
1) docs/input/lines.csv (home,away,spread,over_under)  ← manual overrides
2) data/raw/cfbd/cfb_lines.csv (median by game_id)     ← if fetched by your CFBD step
3) docs/data/predictions.json (spread_home/over_under) ← fallback
4) else: nulls (UI hides market row)

Notes:
- 'spread' is treated as HOME-relative (negative = home favored), consistent with train_and_predict.py.
- Outputs one row per game found in predictions.json, preserving the same order.
"""

import os
import json
from io import StringIO
from typing import Optional

import pandas as pd
import requests

# Paths
LOCAL_DIR = "data/raw/cfbd"
LOCAL_SCHEDULE = f"{LOCAL_DIR}/cfb_schedule.csv"
LOCAL_LINES = f"{LOCAL_DIR}/cfb_lines.csv"

RAW_BASE = "https://raw.githubusercontent.com/moneyball-ab/cfb-data/master/csv"
FALLBACK_SCHEDULE_URL = f"{RAW_BASE}/cfb_schedule.csv"
FALLBACK_LINES_URL = f"{RAW_BASE}/cfb_lines.csv"  # may 404; we handle gracefully

PREDICTIONS_JSON = "docs/data/predictions.json"
MANUAL_LINES_CSV = "docs/input/lines.csv"
ALIASES_JSON = "docs/input/aliases.json"
SPREADS_OUT = "docs/data/spreads.json"


def load_csv_local_or_url(local_path: str, fallback_url: Optional[str]) -> pd.DataFrame:
    """Load CSV from local if exists, else try URL; return EMPTY df on failure."""
    if os.path.exists(local_path):
        try:
            return pd.read_csv(local_path)
        except Exception as e:
            print(f"[WARN] Failed reading {local_path}: {e}")
            return pd.DataFrame()
    if fallback_url:
        try:
            r = requests.get(fallback_url, timeout=60)
            r.raise_for_status()
            return pd.read_csv(StringIO(r.text))
        except Exception as e:
            print(f"[WARN] Fallback fetch failed {fallback_url}: {e}")
            return pd.DataFrame()
    return pd.DataFrame()


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def save_json(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"Wrote {path}")


def load_aliases() -> dict:
    base = {}
    if os.path.exists(ALIASES_JSON):
        try:
            base = json.load(open(ALIASES_JSON, "r"))
            base = {str(k).strip().lower(): str(v).strip() for k, v in base.items()}
        except Exception as e:
            print(f"[WARN] Could not read aliases.json: {e}")
    return base


def norm(name: str, alias_map: dict) -> str:
    if not isinstance(name, str):
        return ""
    key = name.strip().lower()
    return alias_map.get(key, name.strip())


def median_by_game(lines_df: pd.DataFrame) -> pd.DataFrame:
    """CFBD lines format → median per game_id (columns: game_id, spread, over_under)."""
    if lines_df.empty:
        return pd.DataFrame(columns=["game_id", "spread", "over_under"])
    df = lines_df.copy()
    for old, new in [("spread", "spread"), ("overUnder", "over_under"), ("overunder", "over_under")]:
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
    df["spread"] = pd.to_numeric(df.get("spread"), errors="coerce")
    df["over_under"] = pd.to_numeric(df.get("over_under"), errors="coerce")
    if "game_id" not in df.columns:
        return pd.DataFrame(columns=["game_id", "spread", "over_under"])
    return df.groupby("game_id")[["spread", "over_under"]].median().reset_index()


def main():
    if not os.path.exists(PREDICTIONS_JSON):
        raise SystemExit(f"{PREDICTIONS_JSON} not found; run train first.")

    preds = load_json(PREDICTIONS_JSON)
    alias_map = load_aliases()

    # Load schedule (local or snapshot) to map (home,away) -> game_id for latest season (only needed for CFBD lines branch)
    schedule = load_csv_local_or_url(LOCAL_SCHEDULE, FALLBACK_SCHEDULE_URL)
    sched_key = pd.DataFrame(columns=["home_team", "away_team", "game_id"])
    if not schedule.empty:
        for c in ["season", "week"]:
            if c in schedule.columns:
                schedule[c] = pd.to_numeric(schedule[c], errors="coerce")
        latest_season = int(schedule["season"].dropna().max()) if "season" in schedule.columns else None
        sch = schedule.copy()
        if latest_season is not None:
            sch = sch[sch["season"] == latest_season]
        if "date" in sch.columns:
            sch["date"] = pd.to_datetime(sch["date"], errors="coerce", utc=True)
            sch = sch.sort_values(["week", "date"], na_position="last")
        cols = [c for c in ["home_team", "away_team", "game_id"] if c in sch.columns]
        if set(cols) == {"home_team", "away_team", "game_id"}:
            sched_key = (
                sch.dropna(subset=["home_team", "away_team", "game_id"])[cols]
                .astype({"home_team": str, "away_team": str})
                .drop_duplicates(keep="last")
            )

    # 1) Manual lines?
    used_manual = False
    man = pd.DataFrame()
    if os.path.exists(MANUAL_LINES_CSV):
        try:
            man = pd.read_csv(MANUAL_LINES_CSV)
            man.columns = [c.strip().lower() for c in man.columns]
            if {"home", "away", "spread", "over_under"}.issubset(set(man.columns)):
                used_manual = True
                man["spread"] = pd.to_numeric(man["spread"], errors="coerce")
                man["over_under"] = pd.to_numeric(man["over_under"], errors="coerce")
                man["home"] = man["home"].apply(lambda x: norm(x, alias_map))
                man["away"] = man["away"].apply(lambda x: norm(x, alias_map))
            else:
                print("[WARN] docs/input/lines.csv missing required columns; ignoring.")
                man = pd.DataFrame()
        except Exception as e:
            print(f"[WARN] Could not read manual lines: {e}")
            man = pd.DataFrame()

    # 2) CFBD lines (local or snapshot). If snapshot 404s, this stays empty.
    cfbd_lines = pd.DataFrame()
    if not used_manual:
        raw_lines = load_csv_local_or_url(LOCAL_LINES, FALLBACK_LINES_URL)
        cfbd_lines = median_by_game(raw_lines)

    # Build schedule key map for lookup
    key_map = {}
    if not sched_key.empty:
        for _, r in sched_key.iterrows():
            key_map[(str(r["home_team"]).strip(), str(r["away_team"]).strip())] = r["game_id"]

    # Iterate predictions in order and produce output
    out_rows = []
    for g in preds.get("games", []):
        home = norm(g["home"], alias_map)
        away = norm(g["away"], alias_map)
        spread_val = None
        ou_val = None

        if used_manual:
            row = man[(man["home"] == home) & (man["away"] == away)]
            if not row.empty:
                spread_val = float(pd.to_numeric(row.iloc[0]["spread"], errors="coerce"))
                ou_val = float(pd.to_numeric(row.iloc[0]["over_under"], errors="coerce"))
        elif not cfbd_lines.empty and key_map:
            gid = key_map.get((home, away))
            if gid is not None:
                row = cfbd_lines[cfbd_lines["game_id"] == gid]
                if not row.empty:
                    spread_val = float(pd.to_numeric(row.iloc[0]["spread"], errors="coerce"))
                    ou_val = float(pd.to_numeric(row.iloc[0]["over_under"], errors="coerce"))

        # 3) Fallback to predictions.json fields if still missing
        if (spread_val is None or pd.isna(spread_val)) and ("spread_home" in g):
            try:
                spread_val = float(pd.to_numeric(g.get("spread_home"), errors="coerce"))
            except Exception:
                spread_val = None
        if (ou_val is None or pd.isna(ou_val)) and ("over_under" in g):
            try:
                ou_val = float(pd.to_numeric(g.get("over_under"), errors="coerce"))
            except Exception:
                ou_val = None

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

    # Try to infer season from schedule; otherwise copy model meta if present
    season = None
    if "season" in schedule.columns and not schedule["season"].dropna().empty:
        try:
            season = int(schedule["season"].dropna().max())
        except Exception:
            season = None

    save_json(SPREADS_OUT, {"season": season, "games": out_rows})


if __name__ == "__main__":
    main()
