#!/usr/bin/env python3
"""
Fetch team season + advanced stats from CFBD for all teams referenced in docs/input/games.txt,
and write them to docs/data/team_stats.json.

Env:
  CFBD_API_KEY = your RAW CFBD key (no 'Bearer', no '***').

This script uses plain requests to mirror the same authentication that works in your other repo scripts:
    Authorization: Bearer <KEY>
"""

import os
import re
import sys
import json
import argparse
from typing import Dict, List, Tuple
import requests

BASE = "https://api.collegefootballdata.com"

GAMES_FILE = "docs/input/games.txt"
OUT_JSON = "docs/data/team_stats.json"
CFRANK_JSON = "docs/data/cfbrank.json"


def die(msg: str, code: int = 1):
    print(msg, flush=True)
    sys.exit(code)


def get_headers() -> Tuple[Dict[str, str], str]:
    """Return headers with the correct Bearer token (no stars)."""
    raw = os.environ.get("CFBD_API_KEY", "").strip()
    if not raw:
        die("[error] CFBD_API_KEY is missing. Set it to your RAW key (no 'Bearer', no '***').")
    headers = {"Authorization": f"Bearer {raw}"}
    return headers, raw


def http_get(path: str, headers: Dict[str, str], params: Dict = None) -> requests.Response:
    url = f"{BASE}{path}"
    r = requests.get(url, headers=headers, params=params or {}, timeout=30)
    return r


def load_games(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    teams = set()
    splitter = re.compile(r"\s*(?:,|\||vs\.?|@)\s*", re.I)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            parts = [p.strip() for p in splitter.split(raw) if p.strip()]
            for p in parts:
                if re.fullmatch(r"[0-9/:\-\s]+", p):
                    continue
                teams.add(p)
    return sorted(teams)


def load_cfbrank(path: str) -> Dict[str, int]:
    if not os.path.exists(path):
        return {}
    try:
        data = json.load(open(path, encoding="utf-8"))
    except Exception:
        return {}

    ranks = {}
    if isinstance(data, dict):
        if "ranks" in data and isinstance(data["ranks"], dict):
            for rk, tm in data["ranks"].items():
                try:
                    ranks[str(tm)] = int(rk)
                except Exception:
                    pass
        for k, v in data.items():
            if str(k).isdigit():
                if isinstance(v, str):
                    try:
                        ranks[v] = int(k)
                    except Exception:
                        pass
                elif isinstance(v, dict):
                    tm = v.get("team") or v.get("Team")
                    if tm:
                        try:
                            ranks[str(tm)] = int(k)
                        except Exception:
                            pass
    elif isinstance(data, list):
        for obj in data:
            if isinstance(obj, dict):
                rk = obj.get("rank") or obj.get("Rank")
                tm = obj.get("team") or obj.get("Team")
                if rk is not None and tm:
                    try:
                        ranks[str(tm)] = int(rk)
                    except Exception:
                        pass
    return ranks


def fetch_basic(headers: Dict[str, str], team: str, year: int) -> Dict:
    r = http_get("/stats/season", headers, {"year": year, "team": team})
    r.raise_for_status()
    simple = {}
    for row in r.json():
        cat = (row.get("category") or "overall").lower()
        stat = (row.get("statName") or "").lower().replace(" ", "_")
        key = f"{cat}__{stat}"
        val = row.get("statValue")
        try:
            val = float(val)
        except Exception:
            pass
        simple[key] = val
    return simple


def fetch_advanced(headers: Dict[str, str], team: str, year: int) -> Dict:
    r = http_get("/stats/season/advanced", headers, {"year": year, "team": team})
    r.raise_for_status()
    arr = r.json()
    return arr[0] if arr else {}


def fetch_fpi(headers: Dict[str, str], team: str, year: int):
    r = http_get("/ratings/fpi", headers, {"year": year})
    r.raise_for_status()
    for row in r.json():
        if row.get("team") == team:
            return {
                "fpi": row.get("fpi"),
                "ranking": row.get("rank"),
                "resume_rank": row.get("resumeRank"),
            }
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    args = ap.parse_args()

    headers, raw_key = get_headers()
    print(f"[debug] Using CFBD_API_KEY (len={len(raw_key)})")

    teams_raw = load_games(GAMES_FILE)
    if not teams_raw:
        die(f"[error] No teams found in {GAMES_FILE}.", 2)
    print(f"[info] Found {len(teams_raw)} raw team tokens in {GAMES_FILE}")

    # Confirm authorization works first
    sanity = http_get("/teams", headers, {"year": args.year})
    if sanity.status_code != 200:
        die(f"[error] CFBD sanity check failed (HTTP {sanity.status_code}). "
            "Make sure CFBD_API_KEY is correct and available in Actions.", 4)

    name_map = {}
    for t in sanity.json():
        school = t.get("school")
        aliases = {school, t.get("abbreviation"), t.get("alt_name_1"),
                   t.get("alt_name_2"), t.get("alt_name_3")}
        for a in [x for x in aliases if x]:
            name_map[a.lower()] = school

    teams = sorted({name_map.get(t.lower(), t) for t in teams_raw})
    print(f"[info] Normalized to {len(teams)} CFBD team names")
    for t in teams:
        print("  -", t)

    ranks = load_cfbrank(CFRANK_JSON)
    out = []

    for team in teams:
        print(f"[info] Fetching {team} ({args.year})")
        rec = {"team": team, "year": args.year, "simple": {}, "advanced": {}, "fpi": None}

        try:
            rec["simple"] = fetch_basic(headers, team, args.year)
        except Exception as e:
            print(f"[warn] basic stats failed for {team}: {e}")

        try:
            rec["advanced"] = fetch_advanced(headers, team, args.year)
        except Exception as e:
            print(f"[warn] advanced stats failed for {team}: {e}")

        try:
            rec["fpi"] = fetch_fpi(headers, team, args.year)
        except Exception as e:
            print(f"[warn] fpi failed for {team}: {e}")

        if team in ranks:
            rec["model_rank"] = ranks[team]
        out.append(rec)

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[ok] Wrote {OUT_JSON} with {len(out)} teams")


if __name__ == "__main__":
    main()
