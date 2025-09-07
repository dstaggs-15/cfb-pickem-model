#!/usr/bin/env python3
# scripts/fetch_cfbd.py
#
# Downloads CFBD schedule data into data/raw/cfbd/cfb_schedule.csv
# Includes multiple seasons and ONLY completed games (with points).
#
# Env vars:
#   CFBD_API_KEY    (required)
#   START_SEASON    (optional, default: 2014)
#   END_SEASON      (optional, default: current year)
#   INCLUDE_CURRENT (optional, "true"/"false", default: true) -> include current year weeks that have final scores

import os
import sys
import time
import json
import math
import datetime as dt
from typing import List, Dict, Any

import pandas as pd
import requests


RAW_DIR = "data/raw/cfbd"
OUT_PATH = os.path.join(RAW_DIR, "cfbd_schedule.csv")
BASE = "https://api.collegefootballdata.com/v2"   # v2 base
GAMES_ENDPOINT = f"{BASE}/games"                   # /games?year=YYYY&division=fbs

SESSION = requests.Session()


def _headers() -> Dict[str, str]:
    key = os.getenv("CFBD_API_KEY")
    if not key:
        print("ERROR: CFBD_API_KEY is not set in env.", file=sys.stderr)
        sys.exit(1)
    return {"Authorization": f"Bearer {key}"}


def _years_to_fetch() -> List[int]:
    today = dt.date.today()
    start = int(os.getenv("START_SEASON", "2014"))
    end = int(os.getenv("END_SEASON", str(today.year)))
    include_current = os.getenv("INCLUDE_CURRENT", "true").strip().lower() in {"1", "true", "yes", "y"}
    if not include_current:
        end = min(end, today.year - 1)
    if end < start:
        end = start
    return list(range(start, end + 1))


def _fetch_games_for_year(year: int) -> List[Dict[str, Any]]:
    params = {
        "year": year,
        "division": "fbs",
        # v2 supports filtering by gameStatus; we want only finals
        # if the API you have doesn't accept this, we'll filter client-side below.
        "gameStatus": "final"
    }
    r = SESSION.get(GAMES_ENDPOINT, params=params, headers=_headers(), timeout=30)
    if r.status_code == 404:
        # Some v2 deployments may not support gameStatus filter; try without and filter locally
        r = SESSION.get(GAMES_ENDPOINT, params={"year": year, "division": "fbs"}, headers=_headers(), timeout=30)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and "games" in data:
        data = data["games"]  # some variants wrap results
    return data


def _normalize_rows(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Normalize to a flat table with the columns our pipeline expects.
    We keep only completed games with numeric points.
    """
    if not rows:
        return pd.DataFrame(columns=[
            "game_id","season","week","start_date","date",
            "home_team","away_team","home_points","away_points"
        ])

    df = pd.json_normalize(rows, sep="_")

    # Column name variants across API versions:
    # id -> game_id
    if "id" in df.columns and "game_id" not in df.columns:
        df = df.rename(columns={"id": "game_id"})
    # date fields
    date_col = None
    for cand in ["start_date", "startDate", "start_time_tbd", "startTimeTBD", "startTime", "start_time"]:
        if cand in df.columns:
            date_col = cand
            break
    if date_col is None and "start" in df.columns:
        date_col = "start"
    if date_col is None:
        # Some v2 payloads use "game_date"
        if "game_date" in df.columns:
            date_col = "game_date"

    # Derive our 'date' column as pandas Timestamp
    if date_col:
        df["date"] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    else:
        df["date"] = pd.NaT
    if "start_date" not in df.columns:
        df["start_date"] = df.get(date_col, pd.NaT)

    # Points variants
    # Prefer home_points/away_points; if missing, try nested score fields
    if "home_points" not in df.columns:
        for cand in ["home_points", "home_score", "homeTeam_score", "home_points_total"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "home_points"})
                break
    if "away_points" not in df.columns:
        for cand in ["away_points", "away_score", "awayTeam_score", "away_points_total"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "away_points"})
                break

    # Teams variants
    if "home_team" not in df.columns:
        for cand in ["home_team", "homeTeam", "home_school", "home_name"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "home_team"})
                break
    if "away_team" not in df.columns:
        for cand in ["away_team", "awayTeam", "away_school", "away_name"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "away_team"})
                break

    # Week/Season
    if "season" not in df.columns:
        for cand in ["season", "year"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "season"})
                break
        else:
            df["season"] = pd.NA
    if "week" not in df.columns:
        df["week"] = df.get("week", pd.NA)

    # Coerce points to numeric
    for c in ["home_points", "away_points"]:
        if c not in df.columns:
            df[c] = pd.NA
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Filter to completed games w/ at least one points value present
    # Most finals have both points; being permissive here in case of rare nulls.
    if "status" in df.columns:
        status_col = "status"
    elif "gameStatus" in df.columns:
        status_col = "gameStatus"
    else:
        status_col = None

    if status_col:
        finals_mask = df[status_col].astype(str).str.lower().isin(
            ["final", "completed", "complete", "post", "postgame"]
        )
        df = df[finals_mask]

    # If status wasn't available, still keep rows where points are populated
    points_mask = df["home_points"].notna() | df["away_points"].notna()
    df = df[points_mask]

    keep = ["game_id","season","week","start_date","date","home_team","away_team","home_points","away_points"]
    for col in keep:
        if col not in df.columns:
            df[col] = pd.NA

    return df[keep].drop_duplicates(subset=["game_id"])


def main() -> int:
    os.makedirs(RAW_DIR, exist_ok=True)
    years = _years_to_fetch()
    print(f"[FETCH] Years: {years}")

    all_frames = []
    for y in years:
        try:
            rows = _fetch_games_for_year(y)
            df = _normalize_rows(rows)
            print(f"[FETCH] {y}: {df.shape[0]} completed games")
            all_frames.append(df)
            # polite pause
            time.sleep(0.25)
        except requests.HTTPError as e:
            print(f"[FETCH] ERROR {y}: HTTP {e}", file=sys.stderr)
        except Exception as e:
            print(f"[FETCH] ERROR {y}: {e}", file=sys.stderr)

    if not all_frames:
        print("[FETCH] No data fetched; not writing output.", file=sys.stderr)
        return 1

    out = pd.concat(all_frames, ignore_index=True).drop_duplicates(subset=["game_id"])
    # Sort for sanity
    out = out.sort_values(["season","week","date"], na_position="last")
    out.to_csv(OUT_PATH, index=False)
    print(f"[FETCH] Wrote {OUT_PATH} with {out.shape[0]} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
