#!/usr/bin/env python3
# scripts/fetch_cfbd.py
#
# Robust CFBD fetcher with fallbacks:
# - Supports env/CLI: START_SEASON, END_SEASON, INCLUDE_CURRENT
# - Tries multiple candidate URLs for each file (monolithic and per-season patterns)
# - Merges season files when needed
# - Normalizes schedule columns expected downstream
#
# Usage examples:
#   python -m scripts.fetch_cfbd
#   START_SEASON=2014 END_SEASON=2025 INCLUDE_CURRENT=true python -m scripts.fetch_cfbd
#   python -m scripts.fetch_cfbd --start 2014 --end 2025 --include-current

import os
import sys
import argparse
import time
from typing import Dict, List, Tuple

import pandas as pd

DEST_DIR = "data/raw/cfbd"

RAW_BASE = "https://raw.githubusercontent.com/moneyball-ab/cfb-data/master/csv"
# We’ll try these in order; the first that works “wins”.
# For “seasonized” patterns, we put a {season} placeholder and stitch them together.
CANDIDATES: Dict[str, List[str]] = {
    # SCHEDULE
    "cfb_schedule.csv": [
        f"{RAW_BASE}/cfb_schedule.csv",
        f"{RAW_BASE}/schedule.csv",
        f"{RAW_BASE}/schedule/{{season}}.csv",
        f"{RAW_BASE}/cfb_schedule_{{season}}.csv",
    ],

    # LINES (many repos don’t expose a flat cfb_lines.csv at root)
    "cfb_lines.csv": [
        f"{RAW_BASE}/cfb_lines.csv",
        f"{RAW_BASE}/cfb_lines_history.csv",
        f"{RAW_BASE}/lines.csv",
        f"{RAW_BASE}/lines/{{season}}.csv",
        f"{RAW_BASE}/cfb_lines_{{season}}.csv",
        f"{RAW_BASE}/betting_lines/{{season}}.csv",
    ],

    # TEAM STATS
    "cfb_game_team_stats.csv": [
        f"{RAW_BASE}/cfb_game_team_stats.csv",
        f"{RAW_BASE}/team_stats.csv",
        f"{RAW_BASE}/team_stats/{{season}}.csv",
        f"{RAW_BASE}/cfb_game_team_stats_{{season}}.csv",
    ],

    # VENUES (note: 'cfbd_venues.csv' often DOESN'T exist; 'cfb_venues.csv' usually does)
    "cfb_venues.csv": [
        f"{RAW_BASE}/cfb_venues.csv",
        f"{RAW_BASE}/venues.csv",
        f"{RAW_BASE}/venues/{{season}}.csv",
        f"{RAW_BASE}/cfbd_venues.csv",  # keep as a last-try fallback
    ],

    # TEAMS
    "cfb_teams.csv": [
        f"{RAW_BASE}/cfb_teams.csv",
        f"{RAW_BASE}/teams.csv",
        f"{RAW_BASE}/teams/{{season}}.csv",
        f"{RAW_BASE}/cfbd_teams.csv",
    ],

    # TALENT
    "cfb_talent.csv": [
        f"{RAW_BASE}/cfb_talent.csv",
        f"{RAW_BASE}/talent.csv",
        f"{RAW_BASE}/talent/{{season}}.csv",
        f"{RAW_BASE}/recruiting_talent/{{season}}.csv",
    ],
}

# Files the model expects to exist for a full build
REQUIRED_FOR_MODEL = [
    "cfb_schedule.csv",
    "cfb_lines.csv",
    "cfb_game_team_stats.csv",
    "cfb_venues.csv",
    "cfb_teams.csv",
    "cfb_talent.csv",
]


def _read_csv(url: str) -> pd.DataFrame:
    return pd.read_csv(url)


def _read_csv_with_retries(url: str, retries: int = 3, sleep_sec: float = 1.2) -> pd.DataFrame:
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            return _read_csv(url)
        except Exception as e:
            last_err = e
            print(f"[fetch] attempt {attempt} failed for {url}: {e}", file=sys.stderr)
            time.sleep(sleep_sec)
    raise RuntimeError(f"Failed to fetch {url} after {retries} attempts") from last_err


def _normalize_schedule(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure columns we reference later exist
    must_have = ["game_id", "season", "week", "home_team", "away_team", "home_points", "away_points"]
    for c in must_have:
        if c not in df.columns:
            df[c] = pd.NA

    # Coerce some numerics
    for c in ["game_id", "season", "week", "home_points", "away_points"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "neutral_site" not in df.columns:
        df["neutral_site"] = 0
    return df


def _save_csv(df: pd.DataFrame, dest_path: str) -> None:
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    df.to_csv(dest_path, index=False)


def _fetch_single_candidate(url: str, start: int, end: int, include_current: bool) -> Tuple[pd.DataFrame, str]:
    """
    Try to fetch either a monolithic CSV or a seasonized pattern (if '{season}' in URL).
    Returns (df, 'mono' | 'seasonized').
    Raises on failure.
    """
    if "{season}" not in url:
        df = _read_csv_with_retries(url)
        return df, "mono"

    # Seasonized: stitch together
    parts = []
    last_season = end
    if include_current:
        last_season = max(end, last_season)
    for season in range(start, last_season + 1):
        season_url = url.replace("{season}", str(season))
        try:
            df_season = _read_csv_with_retries(season_url)
            if not df_season.empty:
                parts.append(df_season)
                print(f"[fetch]  • pulled {len(df_season)} rows from {season_url}")
        except Exception as e:
            # Not fatal; just continue to next pattern or season
            print(f"[fetch]  • no data at {season_url}: {e}", file=sys.stderr)

    if not parts:
        raise RuntimeError(f"No season files resolved for pattern {url}")

    stitched = pd.concat(parts, ignore_index=True)
    return stitched, "seasonized"


def _fetch_with_candidates(name: str, candidates: List[str], start: int, end: int, include_current: bool) -> pd.DataFrame:
    """
    Try all URL candidates for a given logical file name, returning the first that works.
    For seasonized patterns, we stitch multiple files together.
    """
    last_err = None
    for url in candidates:
        try:
            print(f"[fetch] trying {name} <= {url}")
            df, mode = _fetch_single_candidate(url, start, end, include_current)
            print(f"[fetch] resolved {name} via {mode}: rows={len(df)}, cols={len(df.columns)}")
            return df
        except Exception as e:
            last_err = e
            print(f"[fetch] candidate failed for {name}: {url} => {e}", file=sys.stderr)
    raise RuntimeError(f"All candidates failed for {name}") from last_err


def _parse_bool_env(v: str | None, default: bool = False) -> bool:
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch CFBD raw CSVs into data/raw/cfbd/ with fallbacks.")
    parser.add_argument("--start", type=int, default=int(os.environ.get("START_SEASON", "2014")))
    parser.add_argument("--end", type=int, default=int(os.environ.get("END_SEASON", "2025")))
    parser.add_argument("--include-current", action="store_true", default=_parse_bool_env(os.environ.get("INCLUDE_CURRENT"), True))
    args = parser.parse_args()

    start_season = args.start
    end_season = args.end
    include_current = args.include_current

    print(f"Fetching CFBD raw data into '{DEST_DIR}' (start={start_season}, end={end_season}, include_current={include_current})")

    os.makedirs(DEST_DIR, exist_ok=True)

    successes, failures = [], []

    for logical_name, url_list in CANDIDATES.items():
        try:
            df = _fetch_with_candidates(logical_name, url_list, start_season, end_season, include_current)

            # Schedule normalization for downstream scripts
            if logical_name == "cfb_schedule.csv":
                df = _normalize_schedule(df)

            # For venues, we want the saved filename to be cfb_venues.csv (not cfbd_*).
            dest_name = logical_name
            if logical_name == "cfb_venues.csv":
                dest_name = "cfb_venues.csv"

            dest_path = os.path.join(DEST_DIR, dest_name)
            _save_csv(df, dest_path)
            print(f"[fetch] wrote {dest_path} (rows={len(df)}, cols={len(df.columns)})")
            successes.append(dest_name)
        except Exception as e:
            print(f"[fetch] ERROR for {logical_name}: {e}", file=sys.stderr)
            failures.append(logical_name)

    # Post-process: if a required logical file is missing, flag it
    missing = []
    name_map = {
        # map logical names to saved filenames
        "cfb_schedule.csv": "cfb_schedule.csv",
        "cfb_lines.csv": "cfb_lines.csv",
        "cfb_game_team_stats.csv": "cfb_game_team_stats.csv",
        "cfb_venues.csv": "cfb_venues.csv",
        "cfb_teams.csv": "cfb_teams.csv",
        "cfb_talent.csv": "cfb_talent.csv",
    }
    for req in REQUIRED_FOR_MODEL:
        saved = name_map[req]
        if not os.path.exists(os.path.join(DEST_DIR, saved)):
            missing.append(saved)

    if missing:
        print(f"[fetch] WARNING: missing files after all fallbacks: {missing}", file=sys.stderr)
        return 1

    print(f"[fetch] Done. success={len(successes)}, failed={len(failures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
