#!/usr/bin/env python3
# scripts/fetch_cfbd.py
#
# Fetches ALL required raw data from the CollegeFootballData API.
# Data saved under data/raw/cfbd/*.csv
#
# Usage:
#   python -m scripts.fetch_cfbd --start 2014 --end 2025 --include-current

import os
import sys
import argparse
from datetime import datetime, timezone
from typing import Any, Dict, List
import time
import requests
import pandas as pd
import json

# ---------------------- YOUR CFBD API KEY ----------------------
API_KEY = "B0R+lKakGg0vl2SFDvbmhYpY0M0YHz0OXKVhQnYQ2cwPdOuFLsvU5T4oDUmz2YY/"
# ----------------------------------------------------------------

DEST_DIR = "data/raw/cfbd"
API_BASE = "https://api.collegefootballdata.com"

def _headers() -> Dict[str, str]:
    return {"Authorization": f"Bearer {API_KEY}"}

def _get(path: str, params: Dict[str, Any]) -> Any:
    url = f"{API_BASE}{path}"
    for attempt in range(1, 5):
        resp = requests.get(url, headers=_headers(), params=params, timeout=30)
        if resp.status_code == 200:
            try:
                return resp.json()
            except Exception:
                return []
        time.sleep(1.0 * attempt)
    raise RuntimeError(f"GET {url} failed {resp.status_code}: {resp.text[:200]}")

def _season_range(start: int, end: int, include_current: bool) -> List[int]:
    this_year = datetime.now(timezone.utc).year
    last = max(end, this_year) if include_current else end
    return list(range(start, last + 1))

def _save(df: pd.DataFrame, name: str):
    os.makedirs(DEST_DIR, exist_ok=True)
    path = os.path.join(DEST_DIR, name)
    df.to_csv(path, index=False)
    print(f"[fetch] wrote {path} (rows={len(df)})")

# --- Fetchers ---

def fetch_schedule(seasons: List[int]) -> pd.DataFrame:
    rows = []
    for y in seasons:
        rows.extend(_get("/games", {"year": y, "division": "fbs"}) or [])
        rows.extend(_get("/games", {"year": y, "seasonType": "postseason", "division": "fbs"}) or [])
    df = pd.json_normalize(rows)
    if "id" in df.columns: df.rename(columns={"id":"game_id"}, inplace=True)
    if "neutral_site" not in df.columns: df["neutral_site"] = 0
    return df

def fetch_lines(seasons: List[int]) -> pd.DataFrame:
    flat = []
    for y in seasons:
        data = _get("/lines", {"year": y}) or []
        for r in data:
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
                rec["spread_home"] = ln.get("spread")
                rec["over_under"] = ln.get("overUnder")
                flat.append(rec)
    return pd.DataFrame(flat)

def fetch_team_game_stats(seasons: List[int]) -> pd.DataFrame:
    flat = []
    for y in seasons:
        data = _get("/games/teams", {"year": y}) or []
        for g in data:
            for t in (g.get("teams") or []):
                rec = {
                    "game_id": g.get("id"),
                    "season": g.get("season"),
                    "week": g.get("week"),
                    "team": t.get("school"),
                    "points": t.get("points"),
                }
                for s in (t.get("stats") or []):
                    rec[s.get("category")] = s.get("stat")
                flat.append(rec)
    return pd.DataFrame(flat)

def fetch_venues() -> pd.DataFrame:
    return pd.json_normalize(_get("/venues", {}) or [])

def fetch_teams(seasons: List[int]) -> pd.DataFrame:
    parts = []
    for y in seasons:
        parts.append(pd.json_normalize(_get("/teams/fbs", {"year": y}) or []).assign(season=y))
    return pd.concat(parts, ignore_index=True)

def fetch_talent(seasons: List[int]) -> pd.DataFrame:
    parts = []
    for y in seasons:
        parts.append(pd.json_normalize(_get("/talent", {"year": y}) or []).assign(season=y))
    return pd.concat(parts, ignore_index=True)

# --- Main ---

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, default=2014)
    p.add_argument("--end", type=int, default=datetime.now().year)
    p.add_argument("--include-current", action="store_true")
    args = p.parse_args()

    seasons = _season_range(args.start, args.end, args.include_current)
    print(f"[fetch] seasons={seasons}")

    sched = fetch_schedule(seasons);      _save(sched, "cfb_schedule.csv")
    lines = fetch_lines(seasons);         _save(lines, "cfb_lines.csv")
    stats = fetch_team_game_stats(seasons); _save(stats, "cfb_game_team_stats.csv")
    venues = fetch_venues();              _save(venues, "cfb_venues.csv")
    teams = fetch_teams(seasons);         _save(teams, "cfb_teams.csv")
    talent = fetch_talent(seasons);       _save(talent, "cfb_talent.csv")

    # snapshot
    snap = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rows": {
            "schedule": len(sched),
            "lines": len(lines),
            "team_stats": len(stats),
            "venues": len(venues),
            "teams": len(teams),
            "talent": len(talent),
        }
    }
    os.makedirs("docs/data", exist_ok=True)
    with open("docs/data/fetch_snapshot.json", "w") as f:
        json.dump(snap, f, indent=2)
    print("[fetch] snapshot saved docs/data/fetch_snapshot.json")

if __name__ == "__main__":
    sys.exit(main())
