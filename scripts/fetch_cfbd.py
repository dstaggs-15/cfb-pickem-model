#!/usr/bin/env python3
# scripts/fetch_cfbd.py
#
# Fetch all required raw CollegeFootballData into data/raw/cfbd.
# Uses the CFBD REST API. You must provide CFBD_API_KEY in the environment.
#
# Example:
#   CFBD_API_KEY=your_key START_SEASON=2014 END_SEASON=2025 INCLUDE_CURRENT=true python -m scripts.fetch_cfbd

import os
import sys
import argparse
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple

import requests
import pandas as pd
import json

API_BASE = "https://api.collegefootballdata.com"
DEST_DIR = "data/raw/cfbd"

# Helper for boolean environment variables
def _bool_env(val: str | None, default: bool) -> bool:
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "y"}

def _headers() -> Dict[str, str]:
    key = os.environ.get("CFBD_API_KEY")
    if not key:
        raise RuntimeError("CFBD_API_KEY environment variable is not set.")
    return {"Authorization": f"Bearer {key}"}

def _season_range(start: int, end: int, include_current: bool) -> List[int]:
    this_year = datetime.now(timezone.utc).year
    last = max(end, this_year) if include_current else end
    return list(range(start, last + 1))

def _get(endpoint: str, params: Dict[str, Any], retries: int = 4, backoff: float = 1.0) -> Any:
    url = f"{API_BASE}{endpoint}"
    last_status, last_text = None, None
    for attempt in range(1, retries + 1):
        resp = requests.get(url, headers=_headers(), params=params, timeout=30)
        if resp.status_code == 200:
            try:
                return resp.json()
            except Exception:
                return []
        last_status, last_text = resp.status_code, resp.text[:200]
        time.sleep(backoff * attempt)
    raise RuntimeError(f"GET {url} failed {last_status}: {last_text}")

def _save_csv(df: pd.DataFrame, name: str) -> None:
    os.makedirs(DEST_DIR, exist_ok=True)
    path = os.path.join(DEST_DIR, name)
    df.to_csv(path, index=False)
    print(f"[fetch] wrote {path} (rows={len(df)}, cols={len(df.columns)})")

# --- Fetchers ---

def fetch_schedule(seasons: List[int]) -> pd.DataFrame:
    """Fetch regular and post-season schedules."""
    rows = []
    for y in seasons:
        rows.extend(_get("/games", {"year": y, "division": "fbs"}) or [])
        rows.extend(_get("/games", {"year": y, "seasonType": "postseason", "division": "fbs"}) or [])
    df = pd.json_normalize(rows)
    # normalize key columns
    if "id" in df.columns:
        df.rename(columns={"id": "game_id"}, inplace=True)
    # unify seasonType
    if "season_type" in df.columns and "seasonType" not in df.columns:
        df.rename(columns={"season_type": "seasonType"}, inplace=True)
    for col in ["game_id", "season", "week", "home_points", "away_points"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "neutral_site" not in df.columns:
        df["neutral_site"] = 0
    return df

def fetch_lines(seasons: List[int]) -> pd.DataFrame:
    """Fetch lines per year, flattening provider info."""
    flat = []
    for y in seasons:
        lines = _get("/lines", {"year": y}) or []
        for g in lines:
            base = {
                "game_id": g.get("id"),
                "season": g.get("season"),
                "week": g.get("week"),
                "home_team": g.get("homeTeam"),
                "away_team": g.get("awayTeam"),
            }
            for ln in (g.get("lines") or []):
                rec = base.copy()
                rec["provider"] = ln.get("provider")
                rec["spread_home"] = ln.get("spread")
                rec["over_under"] = ln.get("overUnder")
                rec["openingSpread"] = ln.get("openingSpread")
                rec["formattedSpread"] = ln.get("formattedSpread")
                flat.append(rec)
    return pd.DataFrame(flat)

def _derive_weeks_from_schedule(schedule: pd.DataFrame, year: int) -> List[Tuple[str, int]]:
    """Derive (seasonType, week) pairs from schedule for a given year."""
    if schedule.empty:
        return []
    if "season" not in schedule.columns or "week" not in schedule.columns:
        return []
    st_col = "seasonType"
    if st_col not in schedule.columns and "season_type" in schedule.columns:
        st_col = "season_type"
    if st_col not in schedule.columns:
        return []
    df = schedule[schedule["season"] == year].copy()
    if df.empty:
        return []
    df[st_col] = df[st_col].astype(str).str.lower().str.strip()
    df["week"] = pd.to_numeric(df["week"], errors="coerce").astype("Int64")
    df = df[df["week"].notna()]
    pairs = sorted(
        {
            (st, int(wk))
            for st, wk in df[[st_col, "week"]].itertuples(index=False, name=None)
            if st in {"regular", "postseason"}
        },
        key=lambda x: (x[0], x[1])
    )
    return pairs

def fetch_team_stats(seasons: List[int], schedule: pd.DataFrame) -> pd.DataFrame:
    """Fetch /games/teams by looping derived weeks from the schedule."""
    all_parts = []
    for y in seasons:
        week_pairs = _derive_weeks_from_schedule(schedule, y)
        if not week_pairs:
            print(f"[fetch]  • no weeks found for {y}, skipping team stats")
            continue
        for season_type, wk in week_pairs:
            data = _get("/games/teams", {"year": y, "week": wk, "seasonType": season_type, "division": "fbs"}) or []
            for g in data:
                gid = g.get("id")
                season = g.get("season")
                week = g.get("week")
                teams = g.get("teams") or []
                for t in teams:
                    rec = {
                        "game_id": gid,
                        "season": season,
                        "week": week,
                        "team": t.get("school"),
                        "points": t.get("points"),
                    }
                    for stat in (t.get("stats") or []):
                        cat = stat.get("category")
                        if cat:
                            rec[cat] = stat.get("stat")
                    all_parts.append(rec)
            time.sleep(0.12)
    return pd.DataFrame(all_parts)

def fetch_venues() -> pd.DataFrame:
    return pd.json_normalize(_get("/venues", {}) or [])

def fetch_teams(seasons: List[int]) -> pd.DataFrame:
    parts = []
    for y in seasons:
        parts.append(pd.json_normalize(_get("/teams/fbs", {"year": y}) or []).assign(season=y))
        time.sleep(0.05)
    return pd.concat(parts, ignore_index=True)

def fetch_talent(seasons: List[int]) -> pd.DataFrame:
    parts = []
    for y in seasons:
        parts.append(pd.json_normalize(_get("/talent", {"year": y}) or []).assign(season=y))
        time.sleep(0.05)
    return pd.concat(parts, ignore_index=True)

# --- Main entrypoint ---

def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch raw CFBD data via API.")
    parser.add_argument("--start", type=int, default=int(os.environ.get("START_SEASON", "2014")))
    parser.add_argument("--end", type=int, default=int(os.environ.get("END_SEASON", str(datetime.now().year))))
    parser.add_argument("--include-current", action="store_true",
                        default=_bool_env(os.environ.get("INCLUDE_CURRENT"), True))
    args = parser.parse_args()

    seasons = _season_range(args.start, args.end, args.include_current)
    print(f"[fetch] seasons={seasons}")

    # schedule
    print("[fetch] schedule ...")
    sched = fetch_schedule(seasons)
    _save_csv(sched, "cfb_schedule.csv")

    # lines
    print("[fetch] lines ...")
    lines = fetch_lines(seasons)
    _save_csv(lines, "cfb_lines.csv")

    # team stats
    print("[fetch] team game stats ...")
    team_stats = fetch_team_stats(seasons, sched)
    _save_csv(team_stats, "cfb_game_team_stats.csv")

    # venues, teams, talent
    print("[fetch] venues ...")
    venues = fetch_venues()
    _save_csv(venues, "cfb_venues.csv")

    print("[fetch] teams ...")
    teams = fetch_teams(seasons)
    _save_csv(teams, "cfb_teams.csv")

    print("[fetch] talent ...")
    talent = fetch_talent(seasons)
    _save_csv(talent, "cfb_talent.csv")

    # Snapshot
    snap = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seasons": {"min": min(seasons), "max": max(seasons)},
        "rows": {
            "cfb_schedule.csv": len(sched),
            "cfb_lines.csv": len(lines),
            "cfb_game_team_stats.csv": len(team_stats),
            "cfb_venues.csv": len(venues),
            "cfb_teams.csv": len(teams),
            "cfb_talent.csv": len(talent),
        },
    }
    os.makedirs("docs/data", exist_ok=True)
    with open("docs/data/fetch_snapshot.json", "w") as f:
        json.dump(snap, f, indent=2)
    print("[fetch] snapshot saved to docs/data/fetch_snapshot.json")

    return 0

if __name__ == "__main__":
    sys.exit(main())
