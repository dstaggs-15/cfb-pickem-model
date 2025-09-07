#!/usr/bin/env python3
# scripts/fetch_cfbd.py
#
# Pulls COMPLETED (final) FBS games across seasons using raw HTTP requests to CFBD.
# Writes: data/raw/cfbd/cfb_schedule.csv
#
# ENV:
#   CFBD_API_KEY     (required)  -> Bearer token
#   START_SEASON     (optional)  -> default 2014
#   END_SEASON       (optional)  -> default current year
#   INCLUDE_CURRENT  (optional)  -> "true"/"false" (default true)

import os
import sys
import time
import datetime as dt
from typing import List, Dict, Any

import pandas as pd
import requests

RAW_DIR = "data/raw/cfbd"
OUT_PATH = os.path.join(RAW_DIR, "cfbd_schedule.csv")

V2_BASE = "https://api.collegefootballdata.com/v2"
V1_BASE = "https://api.collegefootballdata.com"  # v1 root

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "cfb-pickem-model/1.0"})


def _bearer_headers() -> Dict[str, str]:
    key = os.getenv("CFBD_API_KEY")
    if not key:
        print("ERROR: CFBD_API_KEY is not set", file=sys.stderr)
        sys.exit(1)
    # Per CFBD docs/email: Authorization: Bearer <key>
    return {"Authorization": f"Bearer {key}"}


def _years() -> List[int]:
    today = dt.date.today()
    start = int(os.getenv("START_SEASON", "2014"))
    end = int(os.getenv("END_SEASON", str(today.year)))
    include_current = os.getenv("INCLUDE_CURRENT", "true").strip().lower() in {"1", "true", "yes", "y"}
    if not include_current:
        end = min(end, today.year - 1)
    if end < start:
        end = start
    return list(range(start, end + 1))


def _get_json(url: str, params: dict):
    r = SESSION.get(url, params=params, headers=_bearer_headers(), timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()


def _fetch_games_year(year: int):
    """
    Try v2 first (with server-side final filter); then v2 without filter; then v1.
    """
    # v2 with gameStatus filter (best case)
    p = {"year": year, "division": "fbs", "seasonType": "both", "gameStatus": "final"}
    data = _get_json(f"{V2_BASE}/games", p)
    if isinstance(data, dict) and "games" in data:
        data = data["games"]
    if isinstance(data, list):
        return data

    # v2 without gameStatus
    p.pop("gameStatus", None)
    data = _get_json(f"{V2_BASE}/games", p)
    if isinstance(data, dict) and "games" in data:
        data = data["games"]
    if isinstance(data, list):
        return data

    # v1 fallback
    p = {"year": year, "division": "fbs", "seasonType": "both"}
    data = _get_json(f"{V1_BASE}/games", p)
    if data is None:
        return []
    return data


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize raw payload to a flat, labeled schedule.
    """
    if df.empty:
        return pd.DataFrame(columns=[
            "game_id","season","week","start_date","date",
            "home_team","away_team","home_points","away_points","status"
        ])

    # --- IDs
    if "id" in df.columns and "game_id" not in df.columns:
        df = df.rename(columns={"id": "game_id"})
    if "gameId" in df.columns and "game_id" not in df.columns:
        df = df.rename(columns={"gameId": "game_id"})

    # --- Date
    date_col = None
    for cand in ["start_date", "startDate", "game_date", "start", "startTime", "start_time"]:
        if cand in df.columns:
            date_col = cand
            break
    if date_col is not None:
        df["start_date"] = df[date_col]
    if "start_date" not in df.columns:
        df["start_date"] = pd.NaT
    df["date"] = pd.to_datetime(df["start_date"], errors="coerce", utc=True)

    # --- Teams
    rename_map = {}
    for src in ["home_team", "homeTeam", "home_school", "home_name"]:
        if src in df.columns:
            rename_map[src] = "home_team"
            break
    for src in ["away_team", "awayTeam", "away_school", "away_name"]:
        if src in df.columns:
            rename_map[src] = "away_team"
            break
    if rename_map:
        df = df.rename(columns=rename_map)

    # --- Season / week
    if "season" not in df.columns:
        for cand in ["season", "year"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "season"})
                break
        else:
            df["season"] = pd.NA
    if "week" not in df.columns:
        df["week"] = df.get("week", pd.NA)

    # --- Points: many variants exist
    def first_existing(*names):
        for n in names:
            if n in df.columns:
                return n
        return None

    hp = first_existing("home_points", "homePoints", "home_score", "homeScore", "homeTeam_score", "home_points_total")
    ap = first_existing("away_points", "awayPoints", "away_score", "awayScore", "awayTeam_score", "away_points_total")

    if hp and hp != "home_points":
        df = df.rename(columns={hp: "home_points"})
    if ap and ap != "away_points":
        df = df.rename(columns={ap: "away_points"})

    if "home_points" not in df.columns:
        df["home_points"] = pd.NA
    if "away_points" not in df.columns:
        df["away_points"] = pd.NA

    df["home_points"] = pd.to_numeric(df["home_points"], errors="coerce")
    df["away_points"] = pd.to_numeric(df["away_points"], errors="coerce")

    # --- Status
    status_col = None
    for cand in ["status", "gameStatus", "game_status", "status_type"]:
        if cand in df.columns:
            status_col = cand
            break
    if status_col is None:
        df["status"] = pd.NA
    else:
        if status_col != "status":
            df = df.rename(columns={status_col: "status"})
        df["status"] = df["status"].astype(str).str.lower()

    keep = ["game_id","season","week","start_date","date","home_team","away_team","home_points","away_points","status"]
    for k in keep:
        if k not in df.columns:
            df[k] = pd.NA

    out = df[keep].drop_duplicates(subset=["game_id"])

    # Final/completed OR has numeric points
    finals = out["status"].isin(["final", "completed", "complete", "post", "postgame", "final/ot"])
    has_pts = out["home_points"].notna() & out["away_points"].notna()
    out = out[finals | has_pts].copy()

    return out


def main() -> int:
    os.makedirs(RAW_DIR, exist_ok=True)
    years = _years()
    print(f"[FETCH] Seasons: {years}")

    frames = []
    for y in years:
        try:
            raw = _fetch_games_year(y)
            df_raw = pd.json_normalize(raw, sep="_") if raw else pd.DataFrame()
            df = _norm(df_raw)
            labeled = (df["home_points"].notna() & df["away_points"].notna()).sum()
            print(f"[FETCH] {y}: rows={len(df)} labeled={labeled}")
            frames.append(df)
            time.sleep(0.2)  # be polite
        except requests.HTTPError as e:
            print(f"[FETCH] HTTP {e} for year {y}", file=sys.stderr)
        except Exception as e:
            print(f"[FETCH] ERROR year {y}: {e}", file=sys.stderr)

    if not frames:
        print("[FETCH] No data fetched; not writing.", file=sys.stderr)
        return 1

    out = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["game_id"])
    out = out.sort_values(["season", "week", "date"], na_position="last")
    out.to_csv(OUT_PATH, index=False)

    labeled_total = (out["home_points"].notna() & out["away_points"].notna()).sum()
    print(f"[FETCH] Wrote {OUT_PATH} rows={out.shape[0]} labeled={labeled_total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
