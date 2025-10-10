#!/usr/bin/env python3
"""
Fetch team season and advanced stats from CFBD for all teams found in docs/input/games.txt.
Outputs: docs/data/team_stats.json

Usage examples:
  python scripts/fetch_team_stats.py --year 2025
  CFBD_API_KEY=your_key_here python scripts/fetch_team_stats.py --year 2025
  python scripts/fetch_team_stats.py --year 2025 --api-key your_key_here
"""

import os, re, json, argparse
import cfbd
from cfbd.rest import ApiException

GAMES_FILE = "docs/input/games.txt"
OUT_JSON = "docs/data/team_stats.json"
CFRANK_JSON = "docs/data/cfbrank.json"


# ------------------- HELPERS -------------------
def load_games(path):
    teams = set()
    if not os.path.exists(path):
        print(f"[warn] {path} not found.")
        return []
    splitter = re.compile(r"\s*(?:,|\||\svs\.?\s|@)\s*", re.I)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            for part in splitter.split(raw):
                if not part or re.fullmatch(r"[0-9/:\-\s]+", part):
                    continue
                teams.add(part.strip())
    return sorted(teams)


def load_cfbrank(path):
    """Robust parser for cfbrank.json with various shapes."""
    if not os.path.exists(path):
        return {}
    try:
        data = json.load(open(path, encoding="utf-8"))
    except Exception as e:
        print(f"[warn] couldn't parse {path}: {e}")
        return {}

    def parse_rank_dict(d):
        out = {}
        for rk, tm in d.items():
            try:
                out[tm] = int(rk)
            except Exception:
                pass
        return out

    # Format A: {"2025": {"1":"Georgia"}}
    if isinstance(data, dict):
        for key, val in data.items():
            if key.isdigit() and isinstance(val, dict):
                return parse_rank_dict(val)
        # Format C: {"season":2025, "ranks":{...}}
        if "ranks" in data:
            return parse_rank_dict(data["ranks"])
        if "rankings" in data and isinstance(data["rankings"], list):
            return {row["team"]: row["rank"] for row in data["rankings"] if "team" in row}
    # Format B: [{"season":2025,"ranks":{...}}]
    if isinstance(data, list) and data:
        latest = max(data, key=lambda x: x.get("season", 0))
        if "ranks" in latest:
            return parse_rank_dict(latest["ranks"])
        if "rankings" in latest:
            return {r["team"]: r["rank"] for r in latest["rankings"] if "team" in r}
    return {}


def init_cfbd(api_key):
    if not api_key:
        raise RuntimeError("CFBD_API_KEY missing.")
    cfg = cfbd.Configuration()
    cfg.api_key["Authorization"] = api_key
    cfg.api_key_prefix["Authorization"] = "Bearer"
    client = cfbd.ApiClient(cfg)
    return cfbd.StatsApi(client), cfbd.RatingsApi(client)


# ------------------- FETCH LOGIC -------------------
def fetch_team(stats_api, ratings_api, team, year):
    out = {"team": team, "year": year, "simple": {}, "advanced": {}, "fpi": None}

    # --- season stats ---
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
        print(f"[warn] team_stats failed for {team}: {e}")

    # --- advanced ---
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
        print(f"[warn] advanced stats failed for {team}: {e}")

    # --- FPI ---
    try:
        fpi_data = ratings_api.get_fpi(year=year)
        for row in fpi_data or []:
            if row.team == team:
                out["fpi"] = {
                    "fpi": row.fpi,
                    "ranking": row.rank,
                    "resume_rank": getattr(row, "resume_rank", None)
                }
                break
    except ApiException as e:
        print(f"[warn] FPI failed for {team}: {e}")

    return out


# ------------------- MAIN -------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--api-key", type=str, help="Optional manual CFBD key override")
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("CFBD_API_KEY")
    stats_api, ratings_api = init_cfbd(api_key)

    teams = load_games(GAMES_FILE)
    ranks = load_cfbrank(CFRANK_JSON)
    results = []
    for t in teams:
        print(f"[info] fetching {t}")
        rec = fetch_team(stats_api, ratings_api, t, args.year)
        if t in ranks:
            rec["model_rank"] = ranks[t]
        results.append(rec)

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[ok] wrote {OUT_JSON} with {len(results)} teams")


if __name__ == "__main__":
    main()
