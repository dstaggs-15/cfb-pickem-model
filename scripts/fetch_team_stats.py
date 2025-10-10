#!/usr/bin/env python3
"""
Fetch team season and advanced stats from CFBD for all teams found in docs/input/games.txt.
Writes: docs/data/team_stats.json

Exit codes:
  0 = success (>=1 team with data)
  2 = games.txt missing/empty
  3 = CFBD key missing
  4 = All API calls unauthorized (bad/missing key)
  5 = No usable stats returned for any team
"""

import os, re, json, argparse, sys
import cfbd
from cfbd.rest import ApiException

GAMES_FILE  = "docs/input/games.txt"
OUT_JSON    = "docs/data/team_stats.json"
CFRANK_JSON = "docs/data/cfbrank.json"

# ---------- small utils ----------
def mask_key(k: str) -> str:
    if not k: return ""
    if len(k) <= 6: return "*" * len(k)
    return k[:3] + "*"*(len(k)-6) + k[-3:]

def die(code: int, msg: str):
    print(msg, flush=True)
    sys.exit(code)

def load_games(path: str):
    teams = set()
    if not os.path.exists(path):
        return []
    splitter = re.compile(r"\s*(?:,|\||\svs\.?\s|@)\s*", re.I)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            parts = [p.strip() for p in splitter.split(raw) if p.strip()]
            for p in parts:
                # ignore pure date/number fragments
                if re.fullmatch(r"[0-9/:\-\s]+", p):
                    continue
                teams.add(p)
    return sorted(teams)

# ---------- robust cfbrank.json loader ----------
def load_cfbrank(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        data = json.load(open(path, encoding="utf-8"))
    except Exception as e:
        print(f"[warn] couldn't parse {path}: {e}")
        return {}

    def from_rank_dict(d: dict) -> dict:
        out = {}
        for rk, tm in d.items():
            try: out[str(tm)] = int(rk)
            except: pass
        return out

    def from_rank_list(lst: list) -> dict:
        out = {}
        for row in lst:
            if not isinstance(row, dict): continue
            team = row.get("team") or row.get("Team") or row.get("name")
            rk = row.get("rank") or row.get("Rank") or row.get("r") or row.get("R")
            try: rk = int(rk)
            except: rk = None
            if team and rk: out[str(team)] = rk
        return out

    if isinstance(data, dict):
        # seasons as keys
        season_keys = [k for k in data.keys() if str(k).isdigit()]
        if season_keys:
            latest = max(season_keys, key=lambda s: int(s))
            payload = data[latest]
            if isinstance(payload, dict):  return from_rank_dict(payload)
            if isinstance(payload, list):  return from_rank_list(payload)
        # single object with ranks / rankings
        if "ranks" in data:
            return from_rank_dict(data["ranks"]) if isinstance(data["ranks"], dict) else from_rank_list(data["ranks"])
        if "rankings" in data and isinstance(data["rankings"], list):
            return from_rank_list(data["rankings"])
        return {}

    if isinstance(data, list) and data:
        latest = None; latest_season = -10**9
        for obj in data:
            if not isinstance(obj, dict): continue
            try: s = int(obj.get("season"))
            except: continue
            if s > latest_season: latest_season, latest = s, obj
        if latest:
            if "ranks" in latest:
                return from_rank_dict(latest["ranks"]) if isinstance(latest["ranks"], dict) else from_rank_list(latest["ranks"])
            if "rankings" in latest and isinstance(latest["rankings"], list):
                return from_rank_list(latest["rankings"])
    return {}

# ---------- CFBD setup ----------
def init_cfbd_clients(raw_key: str):
    if not raw_key:
        die(3, "[error] CFBD_API_KEY missing. Provide via env CFBD_API_KEY or --api-key.")
    cfg = cfbd.Configuration()
    cfg.api_key["Authorization"] = raw_key          # raw key only
    cfg.api_key_prefix["Authorization"] = "Bearer"  # library adds 'Bearer <key>'
    client = cfbd.ApiClient(cfg)
    return cfbd.StatsApi(client), cfbd.RatingsApi(client), cfbd.TeamsApi(client)

# ---------- resolve team names to CFBD canon ----------
def build_name_map(teams_api: cfbd.TeamsApi, year: int) -> dict:
    """Return {lower_name: canonical_name} including aliases."""
    canon = {}
    try:
        rows = teams_api.get_teams(year=year)
    except ApiException as e:
        print(f"[warn] get_teams failed: {e}")
        rows = []
    for r in rows or []:
        if not r.school: continue
        names = {r.school}
        if r.abbreviation: names.add(r.abbreviation)
        if r.alt_name_1:  names.add(r.alt_name_1)
        if r.alt_name_2:  names.add(r.alt_name_2)
        if r.alt_name_3:  names.add(r.alt_name_3)
        for n in names:
            canon[n.lower()] = r.school
    return canon

def normalize_team(name_map: dict, raw: str) -> str:
    k = raw.strip().lower()
    return name_map.get(k, raw.strip())

# ---------- per-team fetch ----------
def fetch_team(stats_api, ratings_api, team: str, year: int) -> dict:
    out = {"team": team, "year": year, "simple": {}, "advanced": {}, "fpi": None}
    unauthorized = False

    # simple stats
    try:
        data = stats_api.get_team_stats(year=year, team=team)
        for item in data or []:
            cat = (item.category or "overall").lower()
            name = (item.stat_name or "").lower().replace(" ", "_")
            key = f"{cat}__{name}" if name else cat
            try:
                val = float(item.stat_value)
            except Exception:
                val = item.stat_value
            out["simple"][key] = val
    except ApiException as e:
        if e.status == 401: unauthorized = True
        print(f"[warn] team_stats failed for {team}: {e}")

    # advanced stats
    try:
        adv = stats_api.get_advanced_season_stats(year=year, team=team)
        if adv:
            adv0 = adv[0]
            def flatten(prefix, node):
                res = {}
                if not node: return res
                nd = node.to_dict()
                for k, v in nd.items():
                    if isinstance(v, dict):
                        for k2, v2 in v.items():
                            res[f"{prefix}__{k}__{k2}"] = v2
                    else:
                        res[f"{prefix}__{k}"] = v
                return res
            out["advanced"].update(flatten("offense", adv0.offense))
            out["advanced"].update(flatten("defense", adv0.defense))
    except ApiException as e:
        if e.status == 401: unauthorized = True
        print(f"[warn] advanced stats failed for {team}: {e}")

    # FPI
    try:
        fpi_rows = ratings_api.get_fpi(year=year)
        for row in fpi_rows or []:
            if row.team == team:
                out["fpi"] = {"fpi": row.fpi, "ranking": row.rank,
                              "resume_rank": getattr(row, "resume_rank", None)}
                break
    except ApiException as e:
        if e.status == 401: unauthorized = True
        print(f"[warn] FPI failed for {team}: {e}")

    return out, unauthorized

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--api-key", type=str, default=None)
    args = ap.parse_args()

    key = args.api_key or os.getenv("CFBD_API_KEY") or ""
    print(f"[debug] CFBD_API_KEY present: {bool(key)} len={len(key)} value(masked)={mask_key(key)}")

    stats_api, ratings_api, teams_api = init_cfbd_clients(key)

    teams_raw = load_games(GAMES_FILE)
    if not teams_raw:
        die(2, f"[error] No teams found. Ensure {GAMES_FILE} exists and lists teams (e.g., 'Alabama vs Georgia').")

    print(f"[info] discovered {len(teams_raw)} team tokens from {GAMES_FILE}")

    # map to canonical CFBD names
    name_map = build_name_map(teams_api, args.year)
    teams = sorted({normalize_team(name_map, t) for t in teams_raw})
    print(f"[info] normalized to {len(teams)} CFBD team names")
    for t in teams:
        print(f"  - {t}")

    ranks = load_cfbrank(CFRANK_JSON)

    results = []
    unauthorized_count = 0
    nonempty_count = 0

    for t in teams:
        print(f"[info] fetching {t} ({args.year})")
        rec, unauth = fetch_team(stats_api, ratings_api, t, args.year)
        if unauth:
            unauthorized_count += 1
        if rec.get("simple") or rec.get("advanced") or rec.get("fpi"):
            nonempty_count += 1
        if t in ranks:
            rec["model_rank"] = ranks[t]
        results.append(rec)

    if unauthorized_count == len(teams):
        die(4, "[error] All requests returned 401 Unauthorized. Your CFBD API key is missing/invalid or formatted incorrectly. In GitHub Secrets store the RAW key (no 'Bearer').")

    if nonempty_count == 0:
        die(5, "[error] No stats returned for any team. Check team names in games.txt and the season year.")

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[ok] wrote {OUT_JSON} with {len(results)} teams and {nonempty_count} with data")

if __name__ == "__main__":
    main()
