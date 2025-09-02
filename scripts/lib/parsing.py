# scripts/lib/parsing.py

import json
import re
import pandas as pd
import numpy as np

def parse_ratio_val(val):
    """
    Parses a string that might be a ratio (e.g., "3-of-9", "3-9") into a float.
    Handles existing floats/ints and non-string values gracefully.
    """
    if not isinstance(val, str):
        return val # Assume it's already numeric or NaN
    
    val = val.lower().replace('-of-', '-').strip()
    
    if '-' in val:
        try:
            num, den = map(float, val.split('-'))
            return num / den if den != 0 else 0.0
        except (ValueError, TypeError):
            return np.nan
    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan

def ensure_schedule_columns(df):
    """
    Ensures a DataFrame has the required columns for a schedule,
    renaming and casting types as necessary.
    """
    column_map = {
        'Game ID': 'game_id', 'Season': 'season', 'Week': 'week',
        'Date': 'date', 'Home Team': 'home_team', 'Away Team': 'away_team',
        'Home Points': 'home_points', 'Away Points': 'away_points',
        'Season Type': 'season_type', 'Neutral Site': 'neutral_site',
        'Venue ID': 'venue_id'
    }
    df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})

    required_cols = list(column_map.values())
    for col in required_cols:
        if col not in df.columns:
            df[col] = pd.NA

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    numeric_cols = ['season', 'week', 'home_points', 'away_points']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    if 'neutral_site' in df.columns:
        # FIX: Fill missing values with False, then convert to boolean type.
        df['neutral_site'] = df['neutral_site'].fillna(False).astype(bool)

    return df

def parse_games_txt(filepath, aliases={}):
    """
    Parses a text file of weekly matchups into a structured list.
    """
    games = []
    neutral_pattern = re.compile(r"^(.*) vs (.*) \(N\)$", re.IGNORECASE)
    away_home_pattern = re.compile(r"^(.*) @ (.*)$", re.IGNORECASE)
    home_away_pattern = re.compile(r"^(.*) vs (.*)$", re.IGNORECASE)
    comma_pattern = re.compile(r"^(.*), (.*)$", re.IGNORECASE)

    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                home, away, neutral = None, None, False

                match = neutral_pattern.match(line)
                if match:
                    home, away, neutral = match.group(1).strip(), match.group(2).strip(), True
                else:
                    match = away_home_pattern.match(line)
                    if match:
                        away, home = match.group(1).strip(), match.group(2).strip()
                    else:
                        match = home_away_pattern.match(line)
                        if match:
                            home, away = match.group(1).strip(), match.group(2).strip()
                        else:
                            match = comma_pattern.match(line)
                            if match:
                                home, away = match.group(1).strip(), match.group(2).strip()

                if home and away:
                    home_norm = aliases.get(home.lower(), home)
                    away_norm = aliases.get(away.lower(), away)
                    
                    games.append({
                        'home_team': home_norm,
                        'away_team': away_norm,
                        'neutral_site': neutral
                    })

    except FileNotFoundError:
        print(f"Warning: Input file not found at {filepath}")
        return []
        
    return games

def load_aliases(filepath):
    """
    Loads a JSON file of team name aliases.
    """
    try:
        with open(filepath, 'r') as f:
            return {k.lower(): v for k, v in json.load(f).items()}
    except FileNotFoundError:
        return {}
