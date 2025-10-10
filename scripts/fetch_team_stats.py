#!/usr/bin/env python3
"""
Fetch team season and advanced stats from CFBD for all teams in docs/input/games.txt.
Writes: docs/data/team_stats.json

Requires env CFBD_API_KEY = your RAW key (no 'Bearer', no '***').
"""

import os
import re
import json
import argparse
import sys
import cfbd
from cfbd.rest import ApiException


# ---------- CONSTANTS ----------
GAMES_FILE = "docs/input/games.txt"
OUT_JSON = "docs/data/team_stats.json"
CFRANK_JSON = "docs/data/cfbrank.json"


# ---------- UTILITIES ----------
def mask_key(k: str) -> str:
    if not k:
        return ""
    return k[:3] + "*" * (len(k) - 6) + k[-3:]


def die(msg, code=1):
    print(msg, flush=True)
    sys.exit(code)


def load_games(path: str):
    """Parse team names from docs/input/games.txt."""
    teams = set()
    if not os.path.exists(path):
        return []

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


# ---------- RANKING LOADER ----------
def load_cfbrank(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        data = json.load(open(path, encoding="utf-8"))
    except Exception:
        return {}

    ranks = {}
    if isinstance(data, dict):
        for k, v in data.items():
            if str(k).isdigit() and isinstance(v, dict):
                for rk, tm in v.items():
                    try:
                        ranks[str(tm)] = int(rk)
                    except Exception:
                        continue
        if "ranks" in data and isinstance(data["ranks"], dict):
            for rk, tm in data["ranks"].items():
                try:
                    ranks[str(tm)] = int(rk)
                except Exception:
                    continue
    elif isinstance(data, list):
        for obj in data:
            if not isinstance(obj, dict):
                continue
            rk = obj.get("rank") or obj.get("Rank")
            tm = obj.get("team") or obj.get("Team")
            try:
                if rk and tm:
                    ranks[str(tm)] = int(rk)
            except Exception:
                continue
    return ranks


# ---------- CFBD CLIENT ----------
def make_cfbd_client():
    """Build a CFBD ApiClient that uses Bearer ***KEY authentication."""
    raw_key = os.environ.get("CFBD_API_KEY", "").strip()
    if not raw_key:
        die("[error] CFBD_API_KEY missing. Provide it as a secret or env var.")

    cfg = cfbd.Configuration()
    cfg.api_key["Authorization"] = f"***{raw_key}"  # CFBD now requires the *** prefix
    cfg.api_key_prefix["Authorization"] = "Bearer"

    print(f"[debug] Using CFBD_API_KEY (masked): {mask_key(raw_key)}")
    return cfbd.ApiClient(cfg)


# ---------- TEAM NORMALIZATION ----------
def build_name_map(teams_api, year: int):
    """Return {alias_lower: canonical_school_name} for normalization."""
    mapping = {}
    try:
        teams = teams_api.get_teams(year=year)
    except ApiException as e:
        print(f"[warn] Could not fetch team list: {e}")
        return mapping

    for t in teams or []:
        names = {t.school}
        for alt in [t.abbreviation, t.alt_name_1, t.alt_name_2, t.alt_name_3]:
            if alt:
                names.add(alt)
        for n in names:
            mapping[n.lower()] = t.school
    return mapping


def normalize_team(name_map, raw):
    return name_map.get(raw.lower().strip(), raw.strip())


# ---------- FETCH STATS ----------
def fetch_team(stats_api, ratings_api, team, year):
    result = {"team": team, "year": year, "simple": {}, "advanced": {}, "fpi": None}
    unauthorized = False

    # Basic stats
    try:
        stats = stats_api.get_team_stats(year=year, team=team)
        for item in stats or []:
            cat = (item.category or "overall").lower()
            stat = (item.stat_name or "").lower().replace(" ", "_")
            key = f"{cat}__{stat}"
            try:
                val = float(item.stat_value)
            except Exception:
                val = item.stat_value
            result["simple"][key] = val
    except ApiException as e:
        if e.status == 401:
            unauthorized = True
        print(f"[warn] team_stats failed for {team}: {e}")

    # Advanced stats
    try:
        advs = stats_api.get_advanced_season_stats(year=year, team=team)
        if advs:
            adv = advs[0]
            def flatten(prefix, node):
                res = {}
                if not node:
                    return res
                nd = node.to_dict()
                for k, v in nd.items():
                    if isinstance(v, dict):
                        for subk, subv in v.items():
                            res[f"{prefix}__{k}__{subk}"] = subv
                    else:
                        res[f"{prefix}__{k}"] = v
                return res

            result["advanced"].update(flatten("offense", adv.offense))
            result["advanced"].update(flatten("defense", adv.defense))
    except ApiException as e:
        if e.status == 401:
            unauthorized = True
        print(f"[warn] advanced stats failed for {team}: {e}")

    # FPI
    try:
        fpi_rows = ratings_api.get_fpi(year=year)
        for row in fpi_rows or []:
            if row.team == team:
                result["fpi"] = {
                    "fpi": row.fpi,
                    "ranking": row.rank,
                    "resume_rank": getattr(row, "resume_rank", None),
                }
                break
    except ApiException as e:
        if e.status == 401:
            unauthorized = True
        print(f"[warn] FPI failed for {team}: {e}")

    return result, unauthorized


# ---------- MAIN ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    args = ap.parse_args()

    games = load_games(GAMES_FILE)
    if not games:
        die(f"[error] No teams found in {GAMES_FILE}.", 2)

    print(f"[info] Found {len(games)} raw team entries in games.txt")

    client = make_cfbd_client()
    stats_api = cfbd.StatsApi(client)
    ratings_api = cfbd.RatingsApi(client)
    teams_api = cfbd.TeamsApi(client)

    name_map = build_name_map(teams_api, args.year)
    teams = sorted({normalize_team(name_map, t) for t in games})
    print(f"[info] Normalized to {len(teams)} valid team names.")
    for t in teams:
        print("  -", t)

    ranks = load_cfbrank(CFRANK_JSON)

    data = []
    unauthorized_count = 0
    nonempty_count = 0

    for team in teams:
        print(f"[info] Fetching {team} stats...")
        rec, unauth = fetch_team(stats_api, ratings_api, team, args.year)
        if unauth:
            unauthorized_count += 1
        if rec["simple"] or rec["advanced"] or rec["fpi"]:
            nonempty_count += 1
        if team in ranks:
            rec["model_rank"] = ranks[team]
        data.append(rec)

    if unauthorized_count == len(teams):
        die("[error] All API calls unauthorized — fix CFBD_API_KEY formatting.", 4)
    if nonempty_count == 0:
        die("[error] No data returned for any team.", 5)

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"[ok] Wrote {OUT_JSON} with {len(data)} teams ({nonempty_count} with real data)")


if __name__ == "__main__":
    main()
