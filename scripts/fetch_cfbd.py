#!/usr/bin/env python3
# scripts/fetch_cfbd.py
#
# Pull FINAL FBS games (labels) from CFBD HTTP API and write:
#   data/raw/cfbd/cfb_schedule.csv
#
# If this fails, it will print the HTTP STATUS and the first 400 chars of the body.
# That immediately tells us if your CFBD_API_KEY is missing/invalid (401/403) or you hit 429, etc.

from __future__ import annotations

import os
import sys
import time
import datetime as dt
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

RAW_DIR = "data/raw/cfbd"
OUT_SCHED = os.path.join(RAW_DIR, "cfb_schedule.csv")

BASE = "https://api.collegefootballdata.com"
GAMES_URL = f"{BASE}/games"

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "cfb-pickem-model/1.0",
    "Accept": "application/json",
})

def _get_api_key() -> str:
    raw = os.getenv("CFBD_API_KEY", "")
    key = raw.strip().strip('"').strip("'")
    if not key:
        print("[FETCH] ERROR: CFBD_API_KEY env var is missing/empty.", file=sys.stderr)
        sys.exit(1)
    return key

def _years() -> List[int]:
    today = dt.date.today()
    start = int(os.getenv("START_SEASON", "2014"))
    end = int(os.getenv("END_SEASON", str(today.year)))
    include_current = os.getenv("INCLUDE_CURRENT", "true").strip().lower() in {"1","true","yes","y"}
    if not include_current:
        end = min(end, today.year - 1)
    if end < start:
        end = start
    return list(range(start, end + 1))

def _sleep_backoff(attempt: int, retry_after: Optional[str]) -> None:
    if retry_after and retry_after.isdigit():
        wait = int(retry_after)
    else:
        wait = min(30, 2 ** attempt) + (attempt % 3)
    print(f"[FETCH] 429 rate limit. Sleeping {wait}s...")
    time.sleep(wait)

def _get_json(url: str, params: Dict[str, Any], max_retries: int = 4) -> Optional[Any]:
    headers = {
        "Authorization": f"Bearer {_get_api_key()}",
        "Accept": "application/json",
        "User-Agent": "cfb-pickem-model/1.0",
    }
    for attempt in range(1, max_retries + 1):
        r = SESSION.get(url, params=params, headers=headers, timeout=45)
        status = r.status_code
        if status == 429:
            _sleep_backoff(attempt, r.headers.get("Retry-After"))
            continue
        body_preview = (r.text or "")[:400].replace("\n", " ")
        if 200 <= status < 300:
            try:
                return r.json()
            except Exception as e:
                print(f"[FETCH] ERROR: JSON parse failed (status {status}): {e}; body[:400]={body_preview}", file=sys.stderr)
                return None
        else:
            print(f"[FETCH] HTTP {status} url={url} params={params} body[:400]={body_preview}", file=sys.stderr)
            if status in (401, 403):
                print("[FETCH] HINT: Your CFBD_API_KEY is missing/invalid, or the header format is wrong. It must be Authorization: Bearer <key>.", file=sys.stderr)
            return None
    print(f"[FETCH] ERROR: exceeded retries for {url} params={params}", file=sys.stderr)
    return None

def _fetch_games_year(year: int) -> pd.DataFrame:
    params = {"year": year, "division": "fbs", "seasonType": "both"}
    data = _get_json(GAMES_URL, params) or []
    if isinstance(data, dict) and "games" in data:
        data = data["games"]
    df = pd.json_normalize(data, sep="_") if data else pd.DataFrame()

    if df.empty:
        return df

    # Normalize columns we need
    if "id" in df.columns and "game_id" not in df.columns:
        df = df.rename(columns={"id": "game_id"})

    # date
    date_col = next((c for c in ["start_date","startDate","start","game_date","start_time","startTime"] if c in df.columns), None)
    if date_col:
        df["date"] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    else:
        df["date"] = pd.NaT

    # home/away team
    if "home_team" not in df.columns:
        for c in ["homeTeam","home_school","home_name"]:
            if c in df.columns:
                df = df.rename(columns={c: "home_team"})
                break
    if "away_team" not in df.columns:
        for c in ["awayTeam","away_school","away_name"]:
            if c in df.columns:
                df = df.rename(columns={c: "away_team"})
                break

    # points
    def _first(*names):
        for n in names:
            if n in df.columns:
                return n
        return None
    hp = _first("home_points","homePoints","home_score","homeScore")
    ap = _first("away_points","awayPoints","away_score","awayScore")
    if hp and hp != "home_points":
        df = df.rename(columns={hp: "home_points"})
    if ap and ap != "away_points":
        df = df.rename(columns={ap: "away_points"})
    df["home_points"] = pd.to_numeric(df.get("home_points", pd.NA), errors="coerce")
    df["away_points"] = pd.to_numeric(df.get("away_points", pd.NA), errors="coerce")

    # status
    status_col = next((c for c in ["status","gameStatus","game_status","status_type"] if c in df.columns), None)
    if status_col and status_col != "status":
        df = df.rename(columns={status_col: "status"})
    if "status" not in df.columns:
        df["status"] = pd.NA
    df["status"] = df["status"].astype(str).str.lower()

    keep = ["game_id","season","week","date","home_team","away_team","home_points","away_points","status"]
    for k in keep:
        if k not in df.columns:
            df[k] = pd.NA
    out = df[keep].drop_duplicates(subset=["game_id"])
    finals = out["status"].isin(["final","final/ot","completed","complete","post","postgame"])
    has_pts = out["home_points"].notna() & out["away_points"].notna()
    return out[finals | has_pts].copy()

def main() -> int:
    os.makedirs(RAW_DIR, exist_ok=True)
    years = _years()
    print(f"[FETCH] Seasons: {years}")

    frames: List[pd.DataFrame] = []
    for y in years:
        try:
            df = _fetch_games_year(y)
            print(f"[FETCH] {y}: rows={len(df)}")
            if not df.empty:
                frames.append(df)
            time.sleep(0.2)
        except Exception as e:
            print(f"[FETCH] ERROR year {y}: {e}", file=sys.stderr)

    if not frames:
        print("[FETCH] No data fetched; not writing.", file=sys.stderr)
        return 1

    out = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["game_id"])
    out = out.sort_values(["season","week","date"], na_position="last")
    out.to_csv(OUT_SCHED, index=False)
    labeled = (out["home_points"].notna() & out["away_points"].notna()).sum()
    print(f"[FETCH] Wrote {OUT_SCHED} rows={len(out)} labeled={labeled}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
