import os, time
from typing import List, Dict, Any
import requests
import pandas as pd
from tqdm import tqdm

# CONFIG: Set the range of years for historical data
YEAR_START = 2005
YEAR_END = 2025 # Inclusive; set to current season

OUT_DIR = "data/raw/cfbd"
SCHEDULE_CSV = os.path.join(OUT_DIR, "cfb_schedule.csv")
TEAM_STATS_CSV = os.path.join(OUT_DIR, "cfb_game_team_stats.csv")
LINES_CSV = os.path.join(OUT_DIR, "cfb_lines.csv")
TEAMS_CSV = os.path.join(OUT_DIR, "cfbd_teams.csv")
VENUES_CSV = os.path.join(OUT_DIR, "cfbd_venues.csv")
TALENT_CSV = os.path.join(OUT_DIR, "cfbd_talent.csv")

API_BASE = "https://api.collegefootballdata.com"
API_KEY = os.environ.get("CFBD_API_KEY", "").strip()
HEADERS = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}

def get_json(path: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Makes a request to the CFBD API with an exponential backoff retry strategy."""
    url = f"{API_BASE}{path}"
    for attempt in range(6):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=60)
            if r.status_code == 200:
                return r.json()
            print(f"  Warning: API returned status {r.status_code} for {url}. Retrying...")
        except requests.exceptions.RequestException as e:
            print(f"  Warning: Request failed for {url}. Error: {e}. Retrying...")
        time.sleep(1.0 + attempt * 1.5)
    
    print(f"  Error: Failed to fetch data from {url} after multiple attempts.")
    return []

def flatten_and_save(data: List[Dict], path: str):
    """Flattens nested JSON data and saves to a CSV."""
    if not data:
        return
    df = pd.json_normalize(data)
    df.to_csv(path, index=False)
    print(f"   Wrote {len(df)} rows to {path}")

def main():
    if not API_KEY:
        raise SystemExit("CFBD_API_KEY not set. Add a repo secret and expose it to this job.")

    os.makedirs(OUT_DIR, exist_ok=True)
    
    # --- Fetch data year-by-year ---
    all_games, all_stats, all_lines, all_talent = [], [], [], []
    years = range(YEAR_START, YEAR_END + 1)
    
    print("Fetching yearly data (Games, Stats, Lines, Talent)...")
    for year in tqdm(years, desc="Years"):
        all_games.extend(get_json("/games", {"year": year, "seasonType": "both", "division": "fbs"}))
        all_stats.extend(get_json("/stats/game/advanced", {"year": year, "seasonType": "both"}))
        all_lines.extend(get_json("/lines", {"year": year, "seasonType": "both"}))
        if year >= 2015: # Talent data is available from 2015 onwards
            all_talent.extend(get_json("/talent", {"year": year}))
    
    # --- Fetch non-yearly data (once) ---
    print("\nFetching non-yearly data (Venues, Teams)...")
    all_venues = get_json("/venues", {})
    all_teams = get_json("/teams/fbs", {}) # Get all current FBS teams

    # --- Save all data to CSV files ---
    print("\nSaving data to CSV files...")
    flatten_and_save(all_games, SCHEDULE_CSV)
    flatten_and_save(all_stats, TEAM_STATS_CSV)
    flatten_and_save(all_lines, LINES_CSV)
    flatten_and_save(all_talent, TALENT_CSV)
    flatten_and_save(all_venues, VENUES_CSV)
    flatten_and_save(all_teams, TEAMS_CSV)

    print("\nData fetching complete.")

if __name__ == "__main__":
    main()
