#!/usr/bin/env python3
"""
Fetch team season and advanced stats from CFBD for all teams found in docs/input/games.txt.
Output: docs/data/team_stats.json

Usage:
  # In GitHub Actions (with CFBD_API_KEY secret) – no extra flags needed
  python scripts/fetch_team_stats.py --year 2025

  # LOCAL run examples (pick ONE of these):
  CFBD_API_KEY=your_real_key_here python scripts/fetch_team_stats.py --year 2025
  python scripts/fetch_team_stats.py --year 2025 --api-key your_real_key_here

Notes:
  - Do NOT prefix your key with 'Bearer' in env/flag; this script adds the header prefix internally.
  - We do NOT touch your existing model code.
"""

import os
import re
import json
import argparse
from typing import Dict, List
import cfbd
from cfbd.rest import ApiException

GAMES_FILE   = "docs/input/games.txt"
OUT_JSON     = "docs/data/team_stats.json"
CFRANK_JSON  = "docs/data/cfbrank.json"   # optional enrich
FPI_FALLBACK = "docs/data/fpi.json"       # optional fallback/context if you keep this file updated

# ---------- utilities ----------
def log(msg: str) -> None:
    print(msg, flush=True)

def load_games(path: str) -> List[str]:
    teams = set()
    if not os.path.exists(path):
        log(f"[warn] {path} not found; returning empty team set.")
        return []
    # split on comma, pipe, 'vs', '@'
    splitter = re.compile(r"\s*(?:,|\||\svs\.?\s|@)\s*", flags=re.IGNORECASE)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            parts = [p.strip() for p in splitter.split(raw) if p.strip()]
            for p in parts:
                # ignore dates / numeric junk
                if re.fullmatch(r"[0-9/:\-\s]+", p):
                    continue
                teams.add(p)
    return sorted(teams)

def mask_key(k: str) -> str:
    if not k:
        return ""
    if len(k) <= 6:
        return "*" * len(k)
    return k[:3] + "*"*(len(k)-6) + k[-3:]

# ---------- cfbrank.json parser (robust to multiple shapes) ----------
def load_cfbrank(path: str) -> Dict[str, int]:
    """
    Accepts any of these shapes and returns {team_name: rank_int} for the most recent season:
    A) {"2025":{"1":"Georgia","2":"Ohio State",...}}
    B) [{"season":2025,"ranks":{"1":"Georgia","2":"Ohio State",...}}, ...]
    C) {"season":2025,"ranks":{"1":"Georgia",...}}  (single object)
    D) [{"season":2025,"rankings":[{"rank":1,"team":"Georgia"}, ...]}, ...]
    E) {"2025":[{"rank":1,"team":"Georgia"}, ...]}
    """
    if not os.path.exists(path):
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        log(f"[warn] could not parse {path}: {e}")
        return {}

    # Helper to convert a rankings container to {team:rank}
    def build_map_from_list(lst):
        out = {}
        for row in lst:
            try:
                r = int(row.get("rank") or row.get("ranking") or row.get("Rank") or row.get("R") or 0)
            except Exception:
                continue
            t = row.get("team") or row.get("Team") or row.get("name")
            if r and t:
                out[t] = r
        return out

    # A/E: dict keyed by season string
    if isinstance(data, dict):
        # Try to detect dict-of-seasons with {rank:team} OR list of {rank,team}
        seasons = []
        for k, v in data.items():
            try:
                seasons.append(int(k))
            except Exception:
                # maybe it's shape C
                pass
        if seasons:
            latest = str(max(seasons))
            v = data[latest]
            if isinstance(v, dict):
                # { "1":"Team", ... }
                out = {}
                for rk, tm in v.items():
                    try:
                        out[tm] = int(rk)
                    except Exception:
                        pass
                return out
            if isinstance(v, list):
                return build_map_from_list(v)

        # Shape C: single object with season + ranks
        if "season" in data and ("ranks" in data or "rankings" in data):
            ranks = data.get("ranks") or data.get("rankings")
            if isinstance(ranks, dict):
                out = {}
                for rk, tm in ranks.items():
                    try: out[tm] = int(rk)
                    except Exception: pass
                return out
            if isinstance(ranks, list):
                return build_map_from_list(ranks)

    # B/D: list of season objects
    if isin
