import pandas as pd
import json

def load_aliases(path):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def ensure_schedule_columns(df):
    required_cols = ['game_id', 'season', 'week', 'date', 'home_team', 'away_team', 'home_points', 'away_points', 'neutral_site']
    for col in required_cols:
        if col not in df.columns:
            df[col] = None
    return df

def parse_games_txt(path, aliases={}):
    """
    Parses a text file of game matchups with robust handling for different separators.
    - "Away @ Home" denotes a standard game.
    - "Team1 vs Team2" or "Team1, Team2" denotes a neutral site game.
    """
    games = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            home_team, away_team, neutral_site = None, None, False

            if '@' in line:
                parts = [p.strip() for p in line.split('@')]
                if len(parts) == 2:
                    away_team, home_team = parts[0], parts[1]
                    neutral_site = False
            elif 'vs' in line:
                parts = [p.strip() for p in line.split('vs')]
                if len(parts) == 2:
                    away_team, home_team = parts[0], parts[1]
                    neutral_site = True
            elif ',' in line:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) == 2:
                    away_team, home_team = parts[0], parts[1]
                    neutral_site = True # Assume ambiguous comma is neutral

            if home_team and away_team:
                home_team = aliases.get(home_team, home_team)
                away_team = aliases.get(away_team, away_team)
                
                games.append({
                    'home_team': home_team,
                    'away_team': away_team,
                    'neutral_site': neutral_site
                })
    return games
