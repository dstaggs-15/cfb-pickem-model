import os
import sys
import time
import datetime as dt
import requests
import pandas as pd
from tqdm import tqdm

# --- Configuration ---
YEAR_START = 2005  # The first year of historical data to fetch
YEAR_END = dt.datetime.now().year  # Automatically get the current year
OUT_DIR = "data/raw/cfbd"
API_BASE = "https://api.collegefootballdata.com"
API_KEY = os.environ.get("CFBD_API_KEY", "").strip()
HEADERS = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}

def get_json(path: str, params: dict) -> list:
    """Makes a request to the CFBD API with rate limiting and retries."""
    url = f"{API_BASE}{path}"
    for attempt in range(5):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=30)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            print(f"  API Error: {e}. Retrying in {2 ** attempt}s...")
            time.sleep(2 ** attempt)
    print(f"Failed to fetch {url} after multiple attempts.")
    return []

def main():
    """Fetches all necessary data from the CFBD API and saves it to CSV files."""
    if not API_KEY:
        print("Error: CFBD_API_KEY is not set. Please add it to your repository's secrets.")
        sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)
    
    all_games, all_stats, all_lines, all_talent = [], [], [], []

    print("Fetching yearly data (Games, Stats, Lines, Talent)...")
    years = range(YEAR_START, YEAR_END + 1)
    
    for year in tqdm(years, desc="Years"):
        # Fetch regular and postseason games for the year
        for season_type in ["regular", "postseason"]:
            games = get_json("/games", {"year": year, "seasonType": season_type, "division": "fbs"})
            if games:
                all_games.extend(games)
            
            # Advanced stats are available from 2013 onwards
            if year >= 2013:
                stats = get_json("/stats/game/advanced", {"year": year, "seasonType": season_type})
                if stats:
                    all_stats.extend(stats)
            
            # Betting lines
            lines = get_json("/lines", {"year": year, "seasonType": season_type})
            if lines:
                all_lines.extend(lines)

        # Team talent ratings are available from 2015 onwards
        if year >= 2015:
            talent = get_json("/talent", {"year": year})
            if talent:
                all_talent.extend(talent)

    print("Fetching non-yearly data (Venues, Teams)...")
    venues = get_json("/venues", {})
    teams = get_json("/teams/fbs", {"year": YEAR_END}) # Get latest team info for coordinates, etc.

    print("Saving data to CSV files...")
    # --- Convert to DataFrames and save ---
    if all_games:
        pd.DataFrame(all_games).to_csv(f"{OUT_DIR}/cfb_schedule.csv", index=False)
    if all_stats:
        pd.DataFrame(all_stats).to_csv(f"{OUT_DIR}/cfb_game_team_stats.csv", index=False)
    if all_lines:
        pd.DataFrame(all_lines).to_csv(f"{OUT_DIR}/cfb_lines.csv", index=False)
    if all_talent:
        pd.DataFrame(all_talent).to_csv(f"{OUT_DIR}/cfbd_talent.csv", index=False)
    if venues:
        pd.DataFrame(venues).to_csv(f"{OUT_DIR}/cfbd_venues.csv", index=False)
    if teams:
        pd.DataFrame(teams).to_csv(f"{OUT_DIR}/cfbd_teams.csv", index=False)
        
    print("Data fetching complete.")

if __name__ == "__main__":
    main()

