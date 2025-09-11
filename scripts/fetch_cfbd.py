#!/usr/bin/env python3
# scripts/fetch_cfbd.py
#
# CFBD fetcher with API KEY **hard-coded** (per your request).
# - Uses existing CSVs in data/raw/cfbd by default (skip-if-present)
# - Fetches only missing files
# - Gracefully handles 429 (quota exceeded) / API failures (doesn't crash pipeline)
# - Optional fallback to public mirror for schedule/teams/talent
# - Writes docs/data/fetch_snapshot.json if anything changed
#
# Usage (env or flags both work for dates only):
#   START_SEASON=2014 END_SEASON=2025 INCLUDE_CURRENT=true python -m scripts.fetch_cfbd
# or:
#   python -m scripts.fetch_cfbd --start 2014 --end 2025 --include-current
#
# Dependencies: requests, pandas

import os
import sys
import argparse
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple, Optional

import json
import requests
import pandas as pd

API_BASE = "https://api.collegefootballdata.com"

# ***** HARD-CODED KEY (as requested) *****
API_KEY = "B0R+lKakGg0vl2SFDvbmhYpY0M0YHz0OXKVhQnYQ2cwPdOuFLsvU5T4oDUmz2YY/"
# *****************************************

DEST_DIR = "data/raw/cfbd"
SNAP_PATH = "docs/data/fetch_snapshot.json"

# Optional public mirror (best effort; some files may 404 there)
MIRROR_BASE = "https://raw.githubusercontent.com/moneyball-ab/cfb-data/master/csv"

CSV_NAMES = {
    "schedule": "cfb_schedule.csv",
    "lines": "cfb_lines.csv",
    "team_stats": "cfb_game_team_stats.csv",
    "venues": "cfb_venues.csv",
    "teams": "cfb_teams.csv",
    "talent": "cfb_talent.csv",
}

def _bool_env(val: Optional[str], default: bool) -> bool:
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "y"}

def _headers() -> Dict[str, str]:
    key = (API_KEY or "").strip()
    if not key:
        raise RuntimeError("Hard-coded CFBD API KEY is empty.")
    return {"Authorization": f"Bearer {key}"}

def _season_range(start: int, end: int, include_current: bool) -> List[int]:
    this_year = datetime.now(timezone.utc).year
    last = max(end, this_year) if include_current else end
    return list(range(start, last + 1))

def _save_csv(df: pd.DataFrame, name: str) -> str:
    os.makedirs(DEST_DIR, exist_ok=True)
    path = os.path.join(DEST_DIR, name)
    df.to_csv(path, index=False)
    print(f"[fetch] wrote {path} (rows={len(df)}, cols={len(df.columns)})")
    return path

def _exists_nonempty(name: str) -> bool:
    path = os.path.join(DEST_DIR, name)
    return os.path.exists(path) and os.path.getsize(path) > 0

def _get(endpoint: str, params: Dict[str, Any], retries: int = 3, backoff: float = 1.0) -> Any:
    url = f"{API_BASE}{endpoint}"
    last_status, last_text = None, None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=_headers(), params=params, timeout=30)
        except Exception as e:
            last_status, last_text = "EXC", str(e)[:200]
            time.sleep(backoff * attempt)
            continue
        if resp.status_code == 200:
            try:
                return resp.json()
            except Exception:
                return []
        last_status, last_text = resp.status_code, resp.text[:200]
        # If quota exceeded, stop retrying—carry on with cache/mirror
        if resp.status_code == 429:
            break
        time.sleep(backoff * attempt)
    raise RuntimeError(f"GET {url} failed {last_status}: {last_text}")

def _get_text(url: str, retries: int = 1) -> Optional[str]:
    last = None
    for _ in range(retries):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                return r.text
            last = f"{r.status_code}: {r.text[:120]}"
        except Exception as e:
            last = str(e)[:120]
        time.sleep(0.2)
    print(f"[fallback] miss {url} ({last})")
    return None

# ---------------- API fetchers ----------------

def fetch_schedule_api(seasons: List[int]) -> pd.DataFrame:
    rows = []
    for y in seasons:
        rows.extend(_get("/games", {"year": y, "division": "fbs"}) or [])
        rows.extend(_get("/games", {"year": y, "seasonType": "postseason", "division": "fbs"}) or [])
    df = pd.json_normalize(rows)
    if "id" in df.columns:
        df.rename(columns={"id": "game_id"}, inplace=True)
    if "season_type" in df.columns and "seasonType" not in df.columns:
        df.rename(columns={"season_type": "seasonType"}, inplace=True)
    for col in ["game_id", "season", "week", "home_points", "away_points"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "neutral_site" not in df.columns:
        df["neutral_site"] = 0
    return df

def fetch_lines_api(seasons: List[int]) -> pd.DataFrame:
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
                rec["openingSpread"] = ln.get("openingSpread")
                rec["formattedSpread"] = ln.get("formattedSpread")
                flat.append(rec)
    return pd.DataFrame(flat)

def _derive_weeks(schedule: pd.DataFrame, year: int) -> List[Tuple[str, int]]:
    if schedule.empty or "season" not in schedule.columns or "week" not in schedule.columns:
        return []
    st_col = "seasonType" if "seasonType" in schedule.columns else ("season_type" if "season_type" in schedule.columns else None)
    if not st_col:
        return []
    df = schedule[schedule["season"] == year].copy()
    if df.empty:
        return []
    df[st_col] = df[st_col].astype(str).str.lower().str.strip()
    df["week"] = pd.to_numeric(df["week"], errors="coerce").astype("Int64")
    df = df[df["week"].notna()]
    pairs = sorted({(st, int(wk)) for st, wk in df[[st_col, "week"]].itertuples(index=False, name=None) if st in {"regular", "postseason"}},
                   key=lambda t: (t[0], t[1]))
    return pairs

def fetch_team_stats_api(seasons: List[int], schedule: pd.DataFrame) -> pd.DataFrame:
    all_parts = []
    for y in seasons:
        pairs = _derive_weeks(schedule, y)
        if not pairs:
            print(f"[fetch]  • no weeks for {y}, skip team stats")
            continue
        for st, wk in pairs:
            data = _get("/games/teams", {"year": y, "week": wk, "seasonType": st, "division": "fbs"}) or []
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
                    all_parts.append(rec)
            time.sleep(0.1)
    return pd.DataFrame(all_parts)

def fetch_venues_api() -> pd.DataFrame:
    return pd.json_normalize(_get("/venues", {}) or [])

def fetch_teams_api(seasons: List[int]) -> pd.DataFrame:
    parts = []
    for y in seasons:
        parts.append(pd.json_normalize(_get("/teams/fbs", {"year": y}) or []).assign(season=y))
        time.sleep(0.05)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

def fetch_talent_api(seasons: List[int]) -> pd.DataFrame:
    parts = []
    for y in seasons:
        parts.append(pd.json_normalize(_get("/talent", {"year": y}) or []).assign(season=y))
        time.sleep(0.05)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

# --------------- Fallback (best effort) ---------------

def fetch_from_mirror(csv_name: str) -> Optional[pd.DataFrame]:
    url = f"{MIRROR_BASE}/{csv_name}"
    txt = _get_text(url, retries=1)
    if not txt:
        return None
    from io import StringIO
    try:
        df = pd.read_csv(StringIO(txt))
        print(f"[fallback] loaded {csv_name} from mirror")
        return df
    except Exception as e:
        print(f"[fallback] parse failed {csv_name}: {e}")
        return None

# --------------- Orchestration helpers ---------------

def _load_existing(name: str) -> Optional[pd.DataFrame]:
    path = os.path.join(DEST_DIR, name)
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[fetch] WARN could not read existing {name}: {e}")
        return None

def ensure_dataset(label: str,
                   seasons: List[int],
                   skip_if_present: bool,
                   allow_partial: bool,
                   schedule_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    label in {"schedule","lines","team_stats","venues","teams","talent"}.
    Returns a DataFrame (possibly from cache, API, or mirror). May be empty if unavailable.
    """
    csv_name = CSV_NAMES[label]

    # 1) Use existing (cache) if allowed
    if skip_if_present and _exists_nonempty(csv_name):
        df = _load_existing(csv_name)
        if df is not None:
            print(f"[fetch] cache hit: {csv_name} (rows={len(df)})")
            return df

    # 2) Try API
    try:
        if label == "schedule":
            df = fetch_schedule_api(seasons)
        elif label == "lines":
            df = fetch_lines_api(seasons)
        elif label == "team_stats":
            df = fetch_team_stats_api(seasons, schedule_df if schedule_df is not None else pd.DataFrame())
        elif label == "venues":
            df = fetch_venues_api()
        elif label == "teams":
            df = fetch_teams_api(seasons)
        elif label == "talent":
            df = fetch_talent_api(seasons)
        else:
            df = pd.DataFrame()
        if df is None:
            df = pd.DataFrame()
        _save_csv(df, csv_name)
        return df
    except Exception as e:
        emsg = str(e)
        print(f"[fetch] API failed for {label}: {emsg}")

    # 3) Fallback mirror for a few datasets (best effort)
    if label in {"schedule", "teams", "talent"}:
        mdf = fetch_from_mirror(csv_name)
        if mdf is not None:
            _save_csv(mdf, csv_name)
            return mdf

    # 4) Final: keep existing (even if empty) or return empty
    existing = _load_existing(csv_name)
    if existing is not None:
        print(f"[fetch] keep existing {csv_name} (rows={len(existing)})")
        return existing

    if allow_partial:
        print(f"[fetch] WARN: proceeding without {csv_name}")
        return pd.DataFrame()

    raise RuntimeError(f"Required dataset {csv_name} not available")

# ---------------- Main ----------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch raw CFBD data (quota-aware).")
    parser.add_argument("--start", type=int, default=int(os.environ.get("START_SEASON", "2014")))
    parser.add_argument("--end", type=int, default=int(os.environ.get("END_SEASON", str(datetime.now().year))))
    parser.add_argument("--include-current", action="store_true",
                        default=_bool_env(os.environ.get("INCLUDE_CURRENT"), True))
    parser.add_argument("--skip-if-present", action="store_true",
                        default=_bool_env(os.environ.get("SKIP_IF_PRESENT"), True))
    parser.add_argument("--allow-partial", action="store_true",
                        default=_bool_env(os.environ.get("ALLOW_PARTIAL"), True))
    args = parser.parse_args()

    seasons = _season_range(args.start, args.end, args.include_current)
    print(f"[fetch] seasons={seasons}")
    print(f"[fetch] options: skip_if_present={args.skip_if_present} allow_partial={args.allow_partial}")

    old_mtimes = {k: os.path.getmtime(os.path.join(DEST_DIR, v)) if os.path.exists(os.path.join(DEST_DIR, v)) else None
                  for k, v in CSV_NAMES.items()}

    # Schedule first
    try:
        sched = ensure_dataset("schedule", seasons, args.skip_if_present, args.allow_partial)
    except Exception as e:
        print(f"[fetch] FATAL: schedule unavailable: {e}")
        return 1

    # Others (graceful on failures)
    lines      = ensure_dataset("lines", seasons, args.skip_if_present, args.allow_partial, schedule_df=sched)
    teamstats  = ensure_dataset("team_stats", seasons, args.skip_if_present, args.allow_partial, schedule_df=sched)
    venues     = ensure_dataset("venues", seasons, args.skip_if_present, args.allow_partial)
    teams      = ensure_dataset("teams", seasons, args.skip_if_present, args.allow_partial)
    talent     = ensure_dataset("talent", seasons, args.skip_if_present, args.allow_partial)

    new_mtimes = {k: os.path.getmtime(os.path.join(DEST_DIR, v)) if os.path.exists(os.path.join(DEST_DIR, v)) else None
                  for k, v in CSV_NAMES.items()}
    changed_any = any(old_mtimes[k] != new_mtimes[k] for k in CSV_NAMES)

    if changed_any:
        os.makedirs(os.path.dirname(SNAP_PATH), exist_ok=True)
        snap = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "seasons": {"min": min(seasons), "max": max(seasons)},
            "rows": {
                CSV_NAMES["schedule"]: len(sched),
                CSV_NAMES["lines"]: len(lines),
                CSV_NAMES["team_stats"]: len(teamstats),
                CSV_NAMES["venues"]: len(venues),
                CSV_NAMES["teams"]: len(teams),
                CSV_NAMES["talent"]: len(talent),
            },
        }
        with open(SNAP_PATH, "w") as f:
            json.dump(snap, f, indent=2)
        print(f"[fetch] snapshot saved to {SNAP_PATH}")
    else:
        print("[fetch] nothing changed; no snapshot written")

    return 0

if __name__ == "__main__":
    sys.exit(main())
