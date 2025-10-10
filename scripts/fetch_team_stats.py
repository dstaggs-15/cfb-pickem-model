#!/usr/bin/env python3
"""
Fetch team season and advanced stats from CFBD for all teams found in docs/input/games.txt.
Output is written to docs/data/team_stats.json.

Usage:
  python scripts/fetch_team_stats.py --year 2025

Requires:
  - env var CFBD_API_KEY to be set (Bearer token)
  - pip install cfbd

Notes:
  - We do NOT touch your existing model scripts.
  - This script is safe to run in a GitHub Action or locally.
"""

import os
import re
import json
import argparse
from collections import defaultdict

import cfbd
from cfbd.rest import ApiException


GAMES_FILE = "docs/input/games.txt"
OUT_JSON   = "docs/data/team_stats.json"
CFRANK_JSON = "docs/data/cfbrank.json"  # optional: enrich with your custom rank if present


def load_games(path: str) -> list[str]:
    teams = set()
    if not os.path.exists(path):
        print(f"[warn] {path} not found; returning empty team set.")
        return []

    splitter = re.compile(r"\s*(?:,|\||\svs\.?\s|@)\s*", flags=re.IGNORECASE)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            # Split on common separators: comma, pipe, "vs", "@"
            parts = [p.strip() for p in splitter.split(raw) if p.strip()]
            # Heuristic: Accept 1–2 tokens per line (team or matchup)
            for p in parts:
                # Filter out obvious junk (weeks, dates, numeric-only)
                if re.fullmatch(r"[0-9/:\-\s]+", p):
                    continue
                teams.add(p)
    return sorted(teams)


def load_cfbrank(path: str) -> dict[str, int]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Expect something like {"2025":{"1":"Georgia","2":"Ohio State",...}}
        # Build reverse lookup: team -> rank
        latest_year = max(data.keys(), key=lambda y: int(y))
        reverse = {}
        for rank_str, team in data[latest_year].items():
            try:
                reverse[team] = int(rank_str)
            except Exception:
                pass
        return reverse
    except Exception as e:
        print(f"[warn] could not parse {path}: {e}")
        return {}


def init_cfbd_client() -> tuple[cfbd.StatsApi, cfbd.RatingsApi]:
    api_key = os.environ.get("CFBD_API_KEY")
    if not api_key:
        raise RuntimeError("CFBD_API_KEY env var not set")

    config = cfbd.Configuration()
    config.api_key["Authorization"] = api_key
    config.api_key_prefix["Authorization"] = "Bearer"
    client = cfbd.ApiClient(config)

    return cfbd.StatsApi(client), cfbd.RatingsApi(client)


def gather_team_stats(stats_api: cfbd.StatsApi, ratings_api: cfbd.RatingsApi, team: str, year: int) -> dict:
    """
    Pull aggregated team season stats + advanced + FPI for a single team.
    Returns a dict ready to serialize.
    """
    out = {
        "team": team,
        "year": year,
        "simple": {},
        "advanced": {},
        "fpi": None
    }

    # --- Aggregated season stats
    try:
        ts = stats_api.get_team_stats(year=year, team=team)
        # API returns a list with items containing .stat_name and .stat_value, grouped by categories
        # We’ll flatten to a simple dict.
        simple = {}
        for item in ts or []:
            # item.category could be 'offense','defense','overall', we prefix keys
            cat = (item.category or "overall").lower()
            name = (item.stat_name or "").lower().replace(" ", "_")
            try:
                val = float(item.stat_value)
            except Exception:
                val = item.stat_value
            key = f"{cat}__{name}" if name else cat
            simple[key] = val
        out["simple"] = simple
    except ApiException as e:
        print(f"[warn] team_stats failed for {team}: {e}")

    # --- Advanced season stats (offense/defense efficiency, ppa, success rate, explosiveness, etc.)
    try:
        adv = stats_api.get_advanced_season_stats(year=year, team=team)
        # Returns a list with .offense and .defense sub-objects (each having ppa, success_rate, explosiveness, etc.)
        if adv:
            adv0 = adv[0]
            def flatten_advanced(prefix, node):
                out_adv = {}
                if node is None:
                    return out_adv
                for k, v in node.to_dict().items():
                    if isinstance(v, dict):
                        for k2, v2 in v.items():
                            out_adv[f"{prefix}__{k}__{k2}"] = v2
                    else:
                        out_adv[f"{prefix}__{k}"] = v
                return out_adv

            adv_dict = {}
            adv_dict.update(flatten_advanced("offense", adv0.offense))
            adv_dict.update(flatten_advanced("defense", adv0.defense))
            out["advanced"] = adv_dict
    except ApiException as e:
        print(f"[warn] advanced_season_stats failed for {team}: {e}")

    # --- FPI
    try:
        fpi = ratings_api.get_fpi(year=year)
        # fpi is a list of TeamFPI items; find our team
        for row in fpi or []:
            if row.team == team:
                out["fpi"] = {
                    "fpi": row.fpi,
                    "ranking": row.rank,
                    "resume_rank": getattr(row, "resume_rank", None),
                    "efficiencies": getattr(row, "efficiencies", None)
                }
                break
    except ApiException as e:
        print(f"[warn] FPI failed for {team}: {e}")

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True, help="Season year, e.g., 2025")
    args = parser.parse_args()
    year = args.year

    teams = load_games(GAMES_FILE)
    if not teams:
        print("[warn] No teams discovered from docs/input/games.txt — nothing to do.")
        return

    # Optional: attach your custom model rank per team if present
    custom_rank = load_cfbrank(CFRANK_JSON)

    stats_api, ratings_api = init_cfbd_client()

    results = []
    for t in teams:
        print(f"[info] fetching stats for {t} ({year})")
        rec = gather_team_stats(stats_api, ratings_api, t, year)
        if t in custom_rank:
            rec["model_rank"] = custom_rank[t]
        results.append(rec)

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[ok] wrote {OUT_JSON} with {len(results)} team records")


if __name__ == "__main__":
    main()
