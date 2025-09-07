#!/usr/bin/env python3
# scripts/fetch_cfbd.py
#
# Fetches CFBD raw CSVs into data/raw/cfbd/.
# By default downloads ALL required files for training/prediction.
# Use --games-only to fetch just the schedule.

import os
import sys
import argparse
import time
from typing import Dict, Tuple

import pandas as pd

RAW_BASE = "https://raw.githubusercontent.com/moneyball-ab/cfb-data/master/csv"
DEST_DIR = "data/raw/cfbd"

# Map local filenames -> remote URLs
FULL_FILES: Dict[str, str] = {
    "cfb_schedule.csv":         f"{RAW_BASE}/cfb_schedule.csv",
    "cfb_lines.csv":            f"{RAW_BASE}/cfb_lines.csv",
    "cfb_game_team_stats.csv":  f"{RAW_BASE}/cfb_game_team_stats.csv",
    "cfbd_venues.csv":          f"{RAW_BASE}/cfbd_venues.csv",
    "cfb_teams.csv":            f"{RAW_BASE}/cfb_teams.csv",
    "cfb_talent.csv":           f"{RAW_BASE}/cfb_talent.csv",
}

GAMES_ONLY: Dict[str, str] = {
    "cfb_schedule.csv": f"{RAW_BASE}/cfb_schedule.csv",
}

REQUIRED_FOR_MODEL = set(FULL_FILES.keys())  # all of them


def _read_csv_with_retries(url: str, retries: int = 3, sleep_sec: float = 1.5) -> pd.DataFrame:
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            df = pd.read_csv(url)
            return df
        except Exception as e:
            last_err = e
            print(f"[fetch] attempt {attempt} failed for {url}: {e}", file=sys.stderr)
            time.sleep(sleep_sec)
    raise RuntimeError(f"Failed to fetch {url} after {retries} attempts") from last_err


def _normalize_schedule(df: pd.DataFrame) -> pd.DataFrame:
    """Light normalization so downstream scripts are happy."""
    # Make sure core columns exist; fill if missing
    must_have = ["game_id", "season", "week", "home_team", "away_team", "home_points", "away_points"]
    for c in must_have:
        if c not in df.columns:
            df[c] = pd.NA

    # Coerce numerics where obvious
    for c in ["game_id", "season", "week", "home_points", "away_points"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Neutral site default if absent
    if "neutral_site" not in df.columns:
        df["neutral_site"] = 0

    return df


def _save_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def fetch_files(files: Dict[str, str]) -> Tuple[int, int]:
    ok, fail = 0, 0
    for name, url in files.items():
        dest = os.path.join(DEST_DIR, name)
        print(f"[fetch] {name}  <=  {url}")
        try:
            df = _read_csv_with_retries(url)
            if name == "cfb_schedule.csv":
                df = _normalize_schedule(df)
            _save_csv(df, dest)
            print(f"[fetch] wrote {dest} (rows={len(df)}, cols={len(df.columns)})")
            ok += 1
        except Exception as e:
            print(f"[fetch] ERROR saving {name}: {e}", file=sys.stderr)
            fail += 1
    return ok, fail


def main():
    parser = argparse.ArgumentParser(description="Fetch CFBD raw CSVs into data/raw/cfbd/")
    parser.add_argument("--games-only", action="store_true", help="Fetch only cfb_schedule.csv")
    args = parser.parse_args()

    files = GAMES_ONLY if args.games_only else FULL_FILES

    print(f"Fetching CFBD raw data into '{DEST_DIR}' ({'games-only' if args.games_only else 'full'}) ...")
    ok, fail = fetch_files(files)

    # Warn if we’re missing any model-critical files after a full run
    if not args.games_only:
        missing = [f for f in REQUIRED_FOR_MODEL if not os.path.exists(os.path.join(DEST_DIR, f))]
        if missing:
            print(f"[fetch] WARNING: missing files after fetch: {missing}", file=sys.stderr)

    print(f"[fetch] Done. success={ok}, failed={fail}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
