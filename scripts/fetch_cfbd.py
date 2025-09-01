import os, time, csv, json
from typing import List, Dict, Any
import requests
import pandas as pd
from tqdm import tqdm

# --------------------------
# CONFIG: adjust years here
# --------------------------
YEAR_START = 2005     # CFBD coverage is good from ~2005+
YEAR_END   = 2025     # inclusive; set to current season
SEASON_TYPES = ["regular", "postseason"]  # "both" is not accepted; fetch both
DIVISION = "fbs"

OUT_DIR = "data/raw/cfbd"
SCHEDULE_CSV = os.path.join(OUT_DIR, "cfb_schedule.csv")
TEAM_STATS_CSV = os.path.join(OUT_DIR, "cfb_game_team_stats.csv")
LINES_CSV = os.path.join(OUT_DIR, "cfb_lines.csv")  # optional extra

API_BASE = "https://api.collegefootballdata.com"
API_KEY = os.environ.get("CFBD_API_KEY", "").strip()

HEADERS = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}

def get_json(path: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    url = f"{API_BASE}{path}"
    for attempt in range(5):
        r = requests.get(url, params=params, headers=HEADERS, timeout=60)
        if r.status_code == 200:
            return r.json()
        # CFBD rate limits; backoff
        time.sleep(1 + attempt * 1.5)
    r.raise_for_status()
    return []

def fetch_games(year: int, season_type: str) -> List[Dict[str, Any]]:
    return get_json("/games", {"year": year, "seasonType": season_type, "division": DIVISION})

def fetch_game_team_stats(year: int, season_type: str) -> List[Dict[str, Any]]:
    # returns one record per game, with a "teams" array; each entry has "team", "homeAway", "stats" (list of {category, stat})
    return get_json("/games/teams", {"year": year, "seasonType": season_type})

def fetch_lines(year: int, season_type: str) -> List[Dict[str, Any]]:
    # Book lines (optional)
    return get_json("/lines", {"year": year, "seasonType": season_type})

def flatten_schedule(games: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for g in games:
        rows.append({
            "game_id": g.get("id"),
            "season": g.get("season"),
            "week": g.get("week"),
            "season_type": g.get("seasonType"),
            "neutral_site": bool(g.get("neutralSite", False)),
            "home_team": g.get("homeTeam"),
            "away_team": g.get("awayTeam"),
            "home_points": g.get("homePoints"),
            "away_points": g.get("awayPoints"),
            "venue": g.get("venue"),
            "conference_game": g.get("conferenceGame"),
            "date": g.get("startDate")
        })
    return rows

# We’ll save a long-form team game stats CSV compatible with our trainer
def flatten_team_stats(stats: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for rec in stats:
        gid = rec.get("id")
        for t in rec.get("teams", []):
            team_name = t.get("school") or t.get("team")
            home_away = t.get("homeAway")
            for s in t.get("stats", []):
                cat = s.get("category")
                val = s.get("stat")
                rows.append({
                    "game_id": gid,
                    "team": team_name,
                    "homeAway": home_away,
                    "category": cat,
                    "stat_value": val
                })
    return rows

def flatten_lines(lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for rec in lines:
        gid = rec.get("id")
        for l in rec.get("lines", []):
            book = l.get("provider", {}).get("name")
            if not book: 
                book = l.get("provider", {}).get("id")
            rows.append({
                "game_id": gid,
                "season": rec.get("season"),
                "week": rec.get("week"),
                "season_type": rec.get("seasonType"),
                "home_team": rec.get("homeTeam"),
                "away_team": rec.get("awayTeam"),
                "spread": l.get("spread"),
                "over_under": l.get("overUnder"),
                "provider": book,
                "updated": l.get("lastUpdated")
            })
    return rows

def main():
    if not API_KEY:
        raise SystemExit("CFBD_API_KEY not set. Add a repo secret and expose it to this job.")

    os.makedirs(OUT_DIR, exist_ok=True)
    sched_rows: List[Dict[str, Any]] = []
    stat_rows: List[Dict[str, Any]] = []
    line_rows: List[Dict[str, Any]] = []

    years = list(range(YEAR_START, YEAR_END + 1))
    for y in tqdm(years, desc="Years"):
        for st in SEASON_TYPES:
            # schedule
            games = fetch_games(y, st)
            sched_rows.extend(flatten_schedule(games))

            # per-team stats for each game
            tstats = fetch_game_team_stats(y, st)
            stat_rows.extend(flatten_team_stats(tstats))

            # betting lines (optional; some years sparse)
            try:
                ldata = fetch_lines(y, st)
                line_rows.extend(flatten_lines(ldata))
            except Exception:
                pass

    pd.DataFrame(sched_rows).to_csv(SCHEDULE_CSV, index=False)
    pd.DataFrame(stat_rows).to_csv(TEAM_STATS_CSV, index=False)
    if line_rows:
        pd.DataFrame(line_rows).to_csv(LINES_CSV, index=False)

    print(f"Wrote:\n  {SCHEDULE_CSV}\n  {TEAM_STATS_CSV}\n  {LINES_CSV if line_rows else '(no lines)'}")

if __name__ == "__main__":
    main()
