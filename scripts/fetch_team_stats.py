#!/usr/bin/env python3
"""
Fetch team season + advanced stats from CFBD for all teams referenced in docs/input/games.txt,
and write them to docs/data/team_stats.json.

Env:
  CFBD_API_KEY = your RAW CFBD key (no 'Bearer', no '***').

Auth detail:
  CFBD now requires the header "Authorization: Bearer ***<KEY>" (note the three asterisks).
"""

import os
import re
import json
import argparse
import sys
import cfbd
from cfbd.rest import ApiException

# ---------- paths ----------
GAMES_FILE = "docs/input/games.txt"
OUT_JSON = "docs/data/team_stats.json"
CFRANK_JSON = "docs/data/cfbrank.json"


# ---------- utils ----------
def mask_key(k: str) -> str:
    if not k:
        return ""
    if len(k) <= 6:
        return "*" * len(k)
    return k[:3] + "*" * (len(k) - 6) + k[-3:]


def die(msg: str, code: int = 1):
    print(msg, flush=True)
    sys.exit(code)


def load_games(path: str):
    """Scrape team tokens from games.txt lines like 'Texas vs Oklahoma | 9/7'."""
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
                # skip obvious date/score fragments
                if re.fullmatch(r"[0-9/:\-\s]+", p):
                    continue
                teams.add(p)
    return sorted(teams)


# ---------- cfbrank loader (robust to dict or list shapes) ----------
def load_cfbrank(path: str) -> dict:
    """Returns {team_name: rank} or {}."""
    if not os.path.exists(path):
        return {}
    try:
        data = json.load(open(path, encoding="utf-8"))
    except Exception:
        return {}

    ranks = {}

    # Case A: dict with possible "ranks" sub-dict or numeric keys
    if isinstance(data, dict):
        if "ranks" in data and isinstance(data["ranks"], dict):
            for rk, tm in data["ranks"].items():
                try:
                    ranks[str(tm)] = int(rk)
                except Exception:
                    pass
        # also support {"1":"Georgia","2":"Michigan",...} or {"1":{"team":"..."}}
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

    # Case B: list of objects like [{"rank":1,"team":"..."}, ...]
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


# ---------- CFBD client ----------
def make_cfbd_client():
    """
    Build an authorized cfbd.ApiClient that sends:
        Authorization: Bearer ***<KEY>
    where CFBD_API_KEY env var is the RAW key (no 'Bearer', no '***').
    """
    raw_key = os.environ.get("CFBD_API_KEY", "").strip()
    if not raw_key:
        die("[error] CFBD_API_KEY is missing. Set it to your RAW key (no 'Bearer', no '***').")

    cfg = cfbd.Configuration()
    # New requirement: three asterisks prefix lives in the api_key value
    cfg.api_key["Authorization"] = f"***{raw_key}"
    cfg.api_key_prefix["Authorization"] = "Bearer"

    print(f"[debug] CFBD_API_KEY present: True len={len(raw_key)} value(masked)={mask_key(raw_key)}")
    return cfbd.ApiClient(cfg)


# ---------- team normalization ----------
def build_name_map(teams_api, year: int):
    """Return {alias_lower: canonical_school_name} so 'OU' → 'Oklahoma' etc."""
    mapping = {}
    try:
        teams = teams_api.get_teams(year=year)
    except ApiException as e:
        print(f"[warn] get_teams failed: {e}")
        return mapping

    for t in teams or []:
        aliases = {t.school}
        for alt in (t.abbreviation, t.alt_name_1, t.alt_name_2, t.alt_name_3):
            if alt:
                aliases.add(alt)
        for a in aliases:
            mapping[a.lower()] = t.school
    return mapping


def normalize_team(name_map: dict, raw: str) -> str:
    return name_map.get(raw.lower().strip(), raw.strip())


# ---------- fetchers ----------
def fetch_team(stats_api, ratings_api, team: str, year: int):
    """Return (record_dict, unauthorized_flag)."""
    rec = {"team": team, "year": year, "simple": {}, "advanced": {}, "fpi": None}
    unauthorized = False

    # basic team stats
    try:
        rows = stats_api.get_team_stats(year=year, team=team)
        for item in rows or []:
            cat = (item.category or "overall").lower()
            stat = (item.stat_name or "").lower().replace(" ", "_")
            key = f"{cat}__{stat}"
            try:
                val = float(item.stat_value)
            except Exception:
                val = item.stat_value
            rec["simple"][key] = val
    except ApiException as e:
        if e.status == 401:
            unauthorized = True
        print(f"[warn] team_stats failed for {team}: {e}")

    # advanced season stats
    try:
        advs = stats_api.get_advanced_season_stats(year=year, team=team)
        if advs:
            adv = advs[0]

            def flatten(prefix, node):
                out = {}
                if not node:
                    return out
                d = node.to_dict()
                for k, v in d.items():
                    if isinstance(v, dict):
                        for sk, sv in v.items():
                            out[f"{prefix}__{k}__{sk}"] = sv
                    else:
                        out[f"{prefix}__{k}"] = v
                return out

            rec["advanced"].update(flatten("offense", adv.offense))
            rec["advanced"].update(flatten("defense", adv.defense))
    except ApiException as e:
        if e.status == 401:
            unauthorized = True
        print(f"[warn] advanced stats failed for {team}: {e}")

    # FPI (ratings)
    try:
        fpi_rows = ratings_api.get_fpi(year=year)
        for row in fpi_rows or []:
            if row.team == team:
                rec["fpi"] = {
                    "fpi": row.fpi,
                    "ranking": row.rank,
                    "resume_rank": getattr(row, "resume_rank", None),
                }
                break
    except ApiException as e:
        if e.status == 401:
            unauthorized = True
        print(f"[warn] FPI failed for {team}: {e}")

    return rec, unauthorized


# ---------- main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    args = parser.parse_args()

    teams_raw = load_games(GAMES_FILE)
    if not teams_raw:
        die(f"[error] No teams found in {GAMES_FILE}.", 2)

    print(f"[info] discovered {len(teams_raw)} team tokens from {GAMES_FILE}")

    client = make_cfbd_client()
    stats_api = cfbd.StatsApi(client)
    ratings_api = cfbd.RatingsApi(client)
    teams_api = cfbd.TeamsApi(client)

    # normalize team names to CFBD canonical form
    name_map = build_name_map(teams_api, args.year)
    teams = sorted({normalize_team(name_map, t) for t in teams_raw})
    print(f"[info] normalized to {len(teams)} CFBD team names")
    for t in teams:
        print(f"  - {t}")

    # optional model ranks
    ranks = load_cfbrank(CFRANK_JSON)

    out = []
    unauthorized_count = 0
    nonempty_count = 0

    for team in teams:
        print(f"[info] fetching {team} ({args.year})")
        rec, unauth = fetch_team(stats_api, ratings_api, team, args.year)
        if unauth:
            unauthorized_count += 1
        if rec["simple"] or rec["advanced"] or rec["fpi"]:
            nonempty_count += 1
        if team in ranks:
            rec["model_rank"] = ranks[team]
        out.append(rec)

    if unauthorized_count == len(teams):
        die("[error] All requests returned 401 Unauthorized. Your CFBD API key is missing/invalid or formatted incorrectly. In GitHub Secrets store the RAW key (no 'Bearer').", 4)
    if nonempty_count == 0:
        die("[error] No data returned for any team. Aborting to avoid writing [].", 5)

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"[ok] wrote {OUT_JSON} with {len(out)} teams ({nonempty_count} with real data)")


if __name__ == "__main__":
    main()
