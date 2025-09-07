#!/usr/bin/env python3
# scripts/fetch_cfbd.py
#
# Fetch ALL required raw data from the CollegeFootballData API and save CSVs to data/raw/cfbd/.
# Endpoints: /games, /lines, /games/teams (loop weeks), /venues, /teams/fbs (per season), /talent (per season)
#
# Usage examples:
#   python -m scripts.fetch_cfbd --start 2014 --end 2025 --include-current
#   START_SEASON=2014 END_SEASON=2025 INCLUDE_CURRENT=true python -m scripts.fetch_cfbd
#
# Required packages: requests, pandas

import os
import sys
import argparse
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
import pandas as pd
import json

# ---------------------- YOUR CFBD API KEY (hardcoded by request) ----------------------
API_KEY = "B0R+lKakGg0vl2SFDvbmhYpY0M0YHz0OXKVhQnYQ2cwPdOuFLsvU5T4oDUmz2YY/"
# -------------------------------------------------------------------------------------

DEST_DIR = "data/raw/cfbd"
API_BASE = "https://api.collegefootballdata.com"

# ---------- Helpers ----------

def _bool_env(val: Optional[str], default: bool) -> bool:
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "y"}

def _headers() -> Dict[str, str]:
    if not API_KEY or API_KEY.strip() == "":
        print("[fetch] ERROR: API_KEY is empty in the script.", file=sys.stderr)
        sys.exit(1)
    return {"Authorization": f"Bearer {API_KEY}"}

def _get(path: str, params: Dict[str, Any], retries: int = 4, backoff: float = 1.0) -> Any:
    """GET with simple retries/backoff; returns parsed JSON (list/dict) or raises."""
    url = f"{API_BASE}{path}"
    last = None
    for attempt in range(1, retries + 1):
        r = requests.get(url, headers=_headers(), params=params, timeout=30)
        if r.status_code == 200:
            try:
                return r.json()
            except Exception:
                return []
        last = (r.status_code, r.text[:300])
        time.sleep(backoff * attempt)
    code, body = last if last else ("?", "?")
    raise RuntimeError(f"GET {url} failed {code}: {body}")

def _season_range(start: int, end: int, include_current: bool) -> List[int]:
    this_year = datetime.now(timezone.utc).year
    last = max(end, this_year) if include_current else end
    return list(range(start, last + 1))

def _save_csv(df: pd.DataFrame, name: str):
    os.makedirs(DEST_DIR, exist_ok=True)
    path = os.path.join(DEST_DIR, name)
    df.to_csv(path, index=False)
    print(f"[fetch] wrote {path} (rows={len(df)}, cols={len(df.columns)})")

# ---------- Fetchers ----------

def fetch_schedule(seasons: List[int]) -> pd.DataFrame:
    """/games for regular + postseason."""
    rows: List[Dict[str, Any]] = []
    for y in seasons:
        # Regular season
        rows.extend(_get("/games", {"year": y, "division": "fbs"}) or [])
        # Postseason
        rows.extend(_get("/games", {"year": y, "seasonType": "postseason", "division": "fbs"}) or [])
    df = pd.json_normalize(rows)
    # Normalize key columns used downstream
    if "id" in df.columns:
        df.rename(columns={"id": "game_id"}, inplace=True)
    for col in ["game_id", "season", "week", "home_points", "away_points"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "neutral_site" not in df.columns:
        df["neutral_site"] = 0
    return df

def fetch_lines(seasons: List[int]) -> pd.DataFrame:
    """/lines by year; flatten provider lines."""
    flat: List[Dict[str, Any]] = []
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
                rec["spread_home"] = ln.get("spread")        # home perspective
                rec["over_under"] = ln.get("overUnder")
                rec["openingSpread"] = ln.get("openingSpread")
                rec["formattedSpread"] = ln.get("formattedSpread")
                flat.append(rec)
    return pd.DataFrame(flat)

def fetch_weeks(year: int) -> List[Tuple[str, int]]:
    """
    Get list of (seasonType, week) pairs for a given year via /weeks.
    seasonType ~ 'regular' or 'postseason'.
    """
    weeks = []
    data = _get("/weeks", {"year": year}) or []
    for w in data:
        st = (w.get("seasonType") or "").strip().lower()
        wk = w.get("week")
        if st in {"regular", "postseason"} and isinstance(wk, int):
            weeks.append((st, wk))
    # De-dupe and sort
    weeks = sorted(set(weeks), key=lambda t: (t[0], t[1]))
    return weeks

def fetch_team_game_stats(seasons: List[int]) -> pd.DataFrame:
    """
    /games/teams requires one of week/team/conference. We loop official weeks.
    We also include division=fbs to match schedule scope.
    """
    flat: List[Dict[str, Any]] = []
    for y in seasons:
        season_weeks = fetch_weeks(y)
        if not season_weeks:
            print(f"[fetch]  • no weeks found for {y}, skipping team stats", file=sys.stderr)
            continue
        for season_type, wk in season_weeks:
            params = {"year": y, "week": wk, "seasonType": season_type, "division": "fbs"}
            data = _get("/games/teams", params) or []
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
                    for s in (t.get("stats") or []):
                        cat = s.get("category")
                        if cat:
                            rec[cat] = s.get("stat")
                    flat.append(rec)
            # be polite with rate limits
            time.sleep(0.15)
    return pd.DataFrame(flat)

def fetch_venues() -> pd.DataFrame:
    return pd.json_normalize(_get("/venues", {}) or [])

def fetch_teams(seasons: List[int]) -> pd.DataFrame:
    parts = []
    for y in seasons:
        parts.append(pd.json_normalize(_get("/teams/fbs", {"year": y}) or []).assign(season=y))
        time.sleep(0.05)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

def fetch_talent(seasons: List[int]) -> pd.DataFrame:
    parts = []
    for y in seasons:
        parts.append(pd.json_normalize(_get("/talent", {"year": y}) or []).assign(season=y))
        time.sleep(0.05)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

# ---------- Main ----------

def main() -> int:
    # Defaults read from env, but CLI can override
    env_start = int(os.environ.get("START_SEASON", "2014"))
    env_end = int(os.environ.get("END_SEASON", str(datetime.now().year)))
    env_inc = _bool_env(os.environ.get("INCLUDE_CURRENT"), True)

    p = argparse.ArgumentParser(description="Fetch CFBD data (schedule, lines, team stats, venues, teams, talent)")
    p.add_argument("--start", type=int, default=env_start)
    p.add_argument("--end", type=int, default=env_end)
    p.add_argument("--include-current", action="store_true", default=env_inc)
    args = p.parse_args()

    seasons = _season_range(args.start, args.end, args.include_current)
    print(f"[fetch] seasons={seasons}")

    # Schedule
    print("[fetch] schedule ...")
    sched = fetch_schedule(seasons)
    _save_csv(sched, "cfb_schedule.csv")

    # Lines
    print("[fetch] lines ...")
    lines = fetch_lines(seasons)
    _save_csv(lines, "cfb_lines.csv")

    # Team game stats (loop weeks)
    print("[fetch] team game stats ... (looping weeks via /weeks)")
    team_stats = fetch_team_game_stats(seasons)
    _save_csv(team_stats, "cfb_game_team_stats.csv")

    # Venues / Teams / Talent
    print("[fetch] venues ...")
    venues = fetch_venues()
    _save_csv(venues, "cfb_venues.csv")

    print("[fetch] teams ...")
    teams = fetch_teams(seasons)
    _save_csv(teams, "cfb_teams.csv")

    print("[fetch] talent ...")
    talent = fetch_talent(seasons)
    _save_csv(talent, "cfb_talent.csv")

    # Snapshot for sanity
    snap = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seasons": {"min": min(seasons), "max": max(seasons)},
        "rows": {
            "cfb_schedule.csv": int(len(sched)),
            "cfb_lines.csv": int(len(lines)),
            "cfb_game_team_stats.csv": int(len(team_stats)),
            "cfb_venues.csv": int(len(venues)),
            "cfb_teams.csv": int(len(teams)),
            "cfb_talent.csv": int(len(talent)),
        },
    }
    os.makedirs("docs/data", exist_ok=True)
    with open("docs/data/fetch_snapshot.json", "w") as f:
        json.dump(snap, f, indent=2)
    print("[fetch] snapshot saved to docs/data/fetch_snapshot.json")

    return 0

if __name__ == "__main__":
    sys.exit(main())
