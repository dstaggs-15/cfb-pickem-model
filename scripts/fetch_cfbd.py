import os, time
from typing import List, Dict, Any, Optional
import requests
import pandas as pd
from tqdm import tqdm

# --------------------------
# CONFIG: adjust years here
# --------------------------
YEAR_START = 2005     # CFBD coverage is good from ~2005+
YEAR_END   = 2025     # inclusive; set to current season
SEASON_TYPES = ["regular", "postseason"]  # fetch both
DIVISION = "fbs"

OUT_DIR = "data/raw/cfbd"
SCHEDULE_CSV = os.path.join(OUT_DIR, "cfb_schedule.csv")
TEAM_STATS_CSV = os.path.join(OUT_DIR, "cfb_game_team_stats.csv")
LINES_CSV = os.path.join(OUT_DIR, "cfb_lines.csv")
TEAMS_CSV = os.path.join(OUT_DIR, "cfbd_teams.csv")
VENUES_CSV = os.path.join(OUT_DIR, "cfbd_venues.csv")
TALENT_CSV = os.path.join(OUT_DIR, "cfbd_talent.csv")  # preseason prior proxy

API_BASE = "https://api.collegefootballdata.com"
API_KEY = os.environ.get("CFBD_API_KEY", "").strip()
HEADERS = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}

def get_json(path: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    url = f"{API_BASE}{path}"
    for attempt in range(6):
        r = requests.get(url, params=params, headers=HEADERS, timeout=60)
        if r.status_code == 200:
            return r.json()
        time.sleep(1.0 + attempt * 1.5)  # polite backoff
    r.raise_for_status()
    return []

def fetch_games(year: int, season_type: str) -> List[Dict[str, Any]]:
    return get_json("/games", {"year": year, "seasonType": season_type, "division": DIVISION})

def fetch_game_team_stats(year: int, season_type: str) -> List[Dict[str, Any]]:
    return get_json("/games/teams", {"year": year, "seasonType": season_type})

def fetch_lines(year: int, season_type: str) -> List[Dict[str, Any]]:
    return get_json("/lines", {"year": year, "seasonType": season_type})

def fetch_venues() -> List[Dict[str, Any]]:
    # No params → all venues
    return get_json("/venues", {})

def fetch_teams_fbs(year: int) -> List[Dict[str, Any]]:
    # FBS teams for a given year
    return get_json("/teams/fbs", {"year": year})

def fetch_talent(year: int) -> List[Dict[str, Any]]:
    # Team talent composite per year (recruiting proxy)
    return get_json("/talent", {"year": year})

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
            "venue_id": g.get("venueId"),
            "venue": g.get("venue"),
            "conference_game": g.get("conferenceGame"),
            "date": g.get("startDate")
        })
    return rows

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
        season = rec.get("season")
        week = rec.get("week")
        stype = rec.get("seasonType")
        h = rec.get("homeTeam")
        a = rec.get("awayTeam")
        for l in rec.get("lines", []):
            provider = l.get("provider", {}) or {}
            book = provider.get("name") or provider.get("id")
            rows.append({
                "game_id": gid,
                "season": season,
                "week": week,
                "season_type": stype,
                "home_team": h,
                "away_team": a,
                "spread": l.get("spread"),
                "over_under": l.get("overUnder"),
                "provider": book,
                "updated": l.get("lastUpdated")
            })
    return rows

def flatten_venues(data: List[Dict[str, Any]]) -> pd.DataFrame:
    out = []
    for v in data:
        out.append({
            "venue_id": v.get("id"),
            "venue_name": v.get("name"),
            "city": v.get("city"),
            "state": v.get("state"),
            "capacity": v.get("capacity"),
            "elevation": v.get("elevation"),
            "latitude": v.get("location", {}).get("latitude") if isinstance(v.get("location"), dict) else v.get("latitude"),
            "longitude": v.get("location", {}).get("longitude") if isinstance(v.get("location"), dict) else v.get("longitude"),
        })
    return pd.DataFrame(out)

def flatten_teams(data: List[Dict[str, Any]], year: int) -> pd.DataFrame:
    # Typical keys: school, mascot, conference, location:{latitude,longitude}, venueId
    out = []
    for t in data:
        loc = t.get("location") or {}
        out.append({
            "year": year,
            "school": t.get("school"),
            "mascot": t.get("mascot"),
            "abbreviation": t.get("abbreviation"),
            "conference": t.get("conference"),
            "division": t.get("division"),
            "venue_id": t.get("venueId"),
            "latitude": loc.get("latitude"),
            "longitude": loc.get("longitude"),
        })
    return pd.DataFrame(out)

def flatten_talent(data: List[Dict[str, Any]], year: int) -> pd.DataFrame:
    # Keys: year, school, talent
    out = []
    for r in data:
        out.append({"year": year, "school": r.get("school"), "talent": r.get("talent")})
    return pd.DataFrame(out)

def main():
    if not API_KEY:
        raise SystemExit("CFBD_API_KEY not set. Add a repo secret and expose it to this job.")

    os.makedirs(OUT_DIR, exist_ok=True)

    sched_rows, stat_rows, line_rows = [], [], []
    for y in tqdm(range(YEAR_START, YEAR_END + 1), desc="Years"):
        for st in SEASON_TYPES:
            try:
                games = fetch_games(y, st)
                sched_rows.extend(flatten_schedule(games))
            except Exception:
                pass
            try:
                tstats = fetch_game_team_stats(y, st)
                stat_rows.extend(flatten_team_stats(tstats))
            except Exception:
                pass
            try:
                ldata = fetch_lines(y, st)
                line_rows.extend(flatten_lines(ldata))
            except Exception:
                pass

    if sched_rows:
        pd.DataFrame(sched_rows).to_csv(SCHEDULE_CSV, index=False)
    if stat_rows:
        pd.DataFrame(stat_rows).to_csv(TEAM_STATS_CSV, index=False)
    if line_rows:
        pd.DataFrame(line_rows).to_csv(LINES_CSV, index=False)

    # Venues (once)
    try:
        venues = fetch_venues()
        flatten_venues(venues).to_csv(VENUES_CSV, index=False)
    except Exception:
        pass

    # Teams (grab for last year to get latest coords/venue)
    try:
        teams = fetch_teams_fbs(YEAR_END)
        flatten_teams(teams, YEAR_END).to_csv(TEAMS_CSV, index=False)
    except Exception:
        pass

    # Talent (preseason priors) for a range of years
    talent_frames = []
    for y in tqdm(range(max(2015, YEAR_START), YEAR_END + 1), desc="Talent"):
        try:
            t = fetch_talent(y)
            if t:
                talent_frames.append(flatten_talent(t, y))
        except Exception:
            time.sleep(0.5)
    if talent_frames:
        pd.concat(talent_frames, ignore_index=True).to_csv(TALENT_CSV, index=False)

    print("Done.")
    print("Wrote:")
    for p in [SCHEDULE_CSV, TEAM_STATS_CSV, LINES_CSV, VENUES_CSV, TEAMS_CSV, TALENT_CSV]:
        if os.path.exists(p):
            print("  ", p)

if __name__ == "__main__":
    main()
