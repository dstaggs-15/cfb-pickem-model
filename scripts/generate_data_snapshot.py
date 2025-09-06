import pandas as pd
import numpy as np
import json
import os
import sys
import datetime as dt

# Import the same parsing helpers used by other scripts for consistency
from .lib.parsing import ensure_schedule_columns

# Define file paths
LOCAL_DIR = "data/raw/cfbd"
LOCAL_SCHEDULE = f"{LOCAL_DIR}/cfb_schedule.csv"
LOCAL_TEAM_STATS = f"{LOCAL_DIR}/cfb_game_team_stats.csv"
OUTPUT_JSON = "docs/data/data_snapshot.json"

# Define key stats we want to display for each team
TEAM_STATS_OF_INTEREST = [
    'ppa',
    'successRate',
    'explosiveness',
    'rushingPPA',
    'passingPPA',
    'turnovers',
    'totalYards'
]

def main():
    """
    Creates a JSON snapshot of the most recent week's games and season-to-date team stats.
    """
    print("Generating data snapshot for the website...")

    # Load the raw schedule and stats data, failing loudly if they don't exist
    if not os.path.exists(LOCAL_SCHEDULE) or not os.path.exists(LOCAL_TEAM_STATS):
        print("Error: Raw data files not found in 'data/raw/cfbd/'.")
        print("Ensure the 'Restore Raw Data from Cache' step ran successfully.")
        sys.exit(1) # Exit with a non-zero code to fail the workflow

    schedule = pd.read_csv(LOCAL_SCHEDULE)
    team_stats_raw = pd.read_csv(LOCAL_TEAM_STATS)
    
    # Use the helper to ensure date and column names are consistent
    schedule = ensure_schedule_columns(schedule)

    # --- 1. Find the most recent week's games ---
    # The 'date' column is now guaranteed to be a datetime object
    
    # Filter for completed games in the current season
    current_season = schedule['season'].max()
    completed_games = schedule[
        (schedule['season'] == current_season) &
        (schedule['home_points'].notna())
    ].copy()

    # Determine the most recent week number that has completed games
    if completed_games.empty:
        print("No completed games found for the current season. Creating empty snapshot.")
        # Create an empty but valid snapshot to avoid breaking the website
        snapshot = {
            "last_updated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "most_recent_week": 0,
            "recent_games": [],
            "team_stats": []
        }
    else:
        most_recent_week = completed_games['week'].max()
        recent_games_df = completed_games[completed_games['week'] == most_recent_week].sort_values(by='date')
        recent_games_list = recent_games_df[['home_team', 'home_points', 'away_team', 'away_points']].to_dict(orient='records')

        # --- 2. Calculate season-to-date team stats ---
        team_stats_long = team_stats_raw.pivot_table(index=['game_id', 'team'], columns='stat', values='value').reset_index()
        
        for stat in TEAM_STATS_OF_INTEREST:
            if stat not in team_stats_long.columns:
                team_stats_long[stat] = 0.0
        
        team_stats_long[TEAM_STATS_OF_INTEREST] = team_stats_long[TEAM_STATS_OF_INTEREST].apply(pd.to_numeric, errors='coerce').fillna(0)

        team_stats_with_season = team_stats_long.merge(schedule[['game_id', 'season']], on='game_id', how='left')
        current_season_stats = team_stats_with_season[team_stats_with_season['season'] == current_season]
        avg_team_stats = current_season_stats.groupby('team')[TEAM_STATS_OF_INTEREST].mean().reset_index()
        
        wins = completed_games.apply(lambda row: row['home_team'] if row['home_points'] > row['away_points'] else row['away_team'], axis=1)
        losses = completed_games.apply(lambda row: row['away_team'] if row['home_points'] > row['away_points'] else row['home_team'], axis=1)
        
        win_counts = wins.value_counts().reset_index(name='wins')
        loss_counts = losses.value_counts().reset_index(name='losses')
        
        avg_team_stats = avg_team_stats.merge(win_counts, on='team', how='left')
        avg_team_stats = avg_team_stats.merge(loss_counts, on='team', how='left')
        avg_team_stats[['wins', 'losses']] = avg_team_stats[['wins', 'losses']].fillna(0).astype(int)
        avg_team_stats['record'] = avg_team_stats['wins'].astype(str) + '-' + avg_team_stats['losses'].astype(str)
        
        team_stats_list = avg_team_stats.sort_values(by='team').to_dict(orient='records')

        snapshot = {
            "last_updated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "most_recent_week": int(most_recent_week),
            "recent_games": recent_games_list,
            "team_stats": team_stats_list
        }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(snapshot, f, indent=4)

    print(f"Successfully created data snapshot at {OUTPUT_JSON}")

if __name__ == "__main__":
    main()

