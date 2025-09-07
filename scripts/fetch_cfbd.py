#!/usr/bin/env python3
# scripts/fetch_cfbd_api.py
#
# Fetch required raw data from the CollegeFootballData API and write CSVs under data/raw/cfbd/.
# Reads seasons from env or CLI flags and stitches all seasons into single CSVs.
#
# ENV:
#   CFBD_API_KEY          (required)
#   START_SEASON, END_SEASON (ints, default 2014..current)
#   INCLUDE_CURRENT       ("true"/"false", default true)
#
# Usage:
#   CFBD_API_KEY=... python -m scripts.fetch_cfbd_api
#   START_SEASON=2014 END_SEASON=2025 INCLUDE_CURRENT=true python -m scripts.fetch_cfbd_api
#   python -m scripts.fetch_cfbd_api --start 2014 --end 2025 --include-current

import os
import sys
import argparse
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import time
import json
import requests
import pandas as pd

DEST_DIR = "data/raw/cfbd"
API_BASE = "https://api.collegefootballdata.com"

# -------- helpers --------

def _bool_env(val: Optional[str], default: bool) -> bool:
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "y", "yes"}

def _headers() -> Dict[str, str]:
    key = os.environ.get("CFBD_API_KEY")
    if not key:
        print("[fetch_api] ERROR: CFBD_API_KEY env var is not set.", file=sys.stderr)
        sys.exit(1)
    return {"Authorization": f"Bearer {key}"}

def _get(path: str, params: Dict[str, Any]) -> Any:
    url = f"{API_BASE}{path}"
    for attempt in range(1, 5):
        resp = requests.get(url, headers=_headers(), params=params, timeout=30)
        if resp.status_code == 200:
            try:
                return resp.json()
            except Exception:
                # sometimes API returns empty string
                return []
        # CFBD has rate limits; a short backoff helps
        time.sleep(1.0 * attempt)
    raise RuntimeError(f"GET {url} failed status={resp.status_code} text={resp.text[:200]}")

def _season_range(start: int, end: int, include_current: bool) -> List[int]:
    this_year = datetime.now(timezone.utc).year
    last = max(end, this_year) if include_current else end
    return list(range(start, last + 1))

def _save_df(df: pd.DataFrame, name: str) -> None:
    os.makedirs(DEST_DIR, exist_ok=True)
    path = os.path.join(DEST_DIR, name)
    df.to_csv(path, index=False)
    print(f"[fetch_api] wrote {path} (rows={len(df)}, cols={len(df.columns)})")

# -------- per-entity fetchers --------

def fetch_schedule(seasons: List[int]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for y in seasons:
        data = _get("/games", {"year": y, "division": "fbs"})
        rows.extend(data or [])
        # Postseason sometimes requires seasonType
        data_post = _get("/games", {"year": y, "seasonType": "postseason", "division": "fbs"})
        rows.extend(data_post or [])
    df = pd.json_normalize(rows)
    # Normalize/canonicalize columns used downstream
    rename = {
        "id": "game_id",
        "home_points": "home_points",
        "away_points": "away_points",
        "home_team": "home_team",
        "away_team": "away_team",
        "neutral_site": "neutral_site",
        "season": "season",
        "week": "week",
    }
    for k, v in rename.items():
        if v not in df.columns and k in df.columns:
            df[v] = df[k]
    for c in ["game_id", "season", "week", "home_points", "away_points"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "neutral_site" not in df.columns:
        df["neutral_site"] = 0
    return df

def fetch_lines(seasons: List[int]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    # CFBD: /lines?year=YYYY (may include multiple providers)
    for y in seasons:
        data = _get("/lines", {"year": y})
        rows.extend(data or [])
    # Flatten provider lines if nested
    flat: List[Dict[str, Any]] = []
    for r in rows:
        # shape: {id, season, week, homeTeam, awayTeam, lines: [{provider,...,spread,overUnder}]}
        base = {
            "game_id": r.get("id"),
            "season": r.get("season"),
            "week": r.get("week"),
            "home_team": r.get("homeTeam"),
            "away_team": r.get("awayTeam"),
        }
        for ln in (r.get("lines") or []):
            rec = base.copy()
            rec["provider"] = ln.get("provider")
            rec["spread_home"] = ln.get("spread")  # CFBD spread is from home perspective (+ = home favored)
            rec["over_under"] = ln.get("overUnder")
            rec["formattedSpread"] = ln.get("formattedSpread")
            rec["openingSpread"] = ln.get("openingSpread")
            flat.append(rec)
    df = pd.DataFrame(flat)
    return df

def fetch_team_game_stats(seasons: List[int]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    # CFBD: /games/teams?year=YYYY
    for y in seasons:
        data = _get("/games/teams", {"year": y})
        rows.extend(data or [])
    # Flatten; each game has 'teams': [{school, points, stats:[{category,stat}]}, ...]
    flat: List[Dict[str, Any]] = []
    for g in rows:
        game_id = g.get("id")
        season = g.get("season")
        week = g.get("week")
        teams = g.get("teams") or []
        for t in teams:
            school = t.get("school")
            pts = t.get("points")
            stats = t.get("stats") or []
            rec = {"game_id": game_id, "season": season, "week": week, "team": school, "points": pts}
            for s in stats:
                cat = (s.get("category") or "").strip()
                val = s.get("stat")
                if cat:
                    rec[cat] = val
            flat.append(rec)
    df = pd.DataFrame(flat)
    return df

def fetch_venues() -> pd.DataFrame:
    data = _get("/venues", {})
    return pd.json_normalize(data or [])

def fetch_teams(seasons: List[int]) -> pd.DataFrame:
    # Team list can vary by year; stitch
    parts = []
    for y in seasons:
        data = _get("/teams/fbs", {"year": y})
        parts.append(pd.json_normalize(data or []).assign(season=y))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

def fetch_talent(seasons: List[int]) -> pd.DataFrame:
    parts = []
    for y in seasons:
        data = _get("/talent", {"year": y})
        parts.append(pd.json_normalize(data or []).assign(season=y))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

# -------- main --------

def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch raw CFBD data via API and write CSVs.")
    parser.add_argument("--start", type=int, default=int(os.environ.get("START_SEASON", "2014")))
    parser.add_argument("--end", type=int, default=int(os.environ.get("END_SEASON", str(datetime.now().year))))
    parser.add_argument("--include-current", action="store_true",
                        default=_bool_env(os.environ.get("INCLUDE_CURRENT"), True))
    args = parser.parse_args()

    seasons = _season_range(args.start, args.end, args.include_current)
    print(f"[fetch_api] seasons={seasons}")

    # Fetch
    print("[fetch_api] schedule ...")
    schedule = fetch_schedule(seasons)

    print("[fetch_api] lines ...")
    lines = fetch_lines(seasons)

    print("[fetch_api] team game stats ...")
    team_stats = fetch_team_game_stats(seasons)

    print("[fetch_api] venues ...")
    venues = fetch_venues()

    print("[fetch_api] teams ...")
    teams = fetch_teams(seasons)

    print("[fetch_api] talent ...")
    talent = fetch_talent(seasons)

    # Save
    _save_df(schedule, "cfb_schedule.csv")
    _save_df(lines, "cfb_lines.csv")
    _save_df(team_stats, "cfb_game_team_stats.csv")
    _save_df(venues, "cfb_venues.csv")
    _save_df(teams, "cfb_teams.csv")
    _save_df(talent, "cfb_talent.csv")

    # Minimal snapshot so you can inspect what you got
    snapshot = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seasons": {"min": min(seasons), "max": max(seasons)},
        "rows": {
            "cfb_schedule.csv": int(len(schedule)),
            "cfb_lines.csv": int(len(lines)),
            "cfb_game_team_stats.csv": int(len(team_stats)),
            "cfb_venues.csv": int(len(venues)),
            "cfb_teams.csv": int(len(teams)),
            "cfb_talent.csv": int(len(talent)),
        },
    }
    os.makedirs("docs/data", exist_ok=True)
    with open("docs/data/fetch_snapshot.json", "w") as f:
        json.dump(snapshot, f, indent=2)
    print("[fetch_api] wrote docs/data/fetch_snapshot.json")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
