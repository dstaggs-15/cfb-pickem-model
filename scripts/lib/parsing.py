import json
import re
import pandas as pd

def ensure_schedule_columns(df):
    """
    Ensures a DataFrame has the required columns for a schedule,
    renaming and casting types as necessary.
    
    Args:
        df (pd.DataFrame): The raw schedule DataFrame.

    Returns:
        pd.DataFrame: The cleaned schedule DataFrame.
    """
    # Mapping of potential column names to our standard names
    column_map = {
        'Game ID': 'game_id',
        'Season': 'season',
        'Week': 'week',
        'Date': 'date',
        'Home Team': 'home_team',
        'Away Team': 'away_team',
        'Home Points': 'home_points',
        'Away Points': 'away_points',
        'Season Type': 'season_type',
        'Neutral Site': 'neutral_site',
        'Venue ID': 'venue_id'
    }

    # Rename columns that exist in the mapping
    df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})

    # Ensure required columns exist, filling with NaN if not
    required_cols = list(column_map.values())
    for col in required_cols:
        if col not in df.columns:
            df[col] = pd.NA

    # Coerce data types
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    numeric_cols = ['season', 'week', 'home_points', 'away_points']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    if 'neutral_site' in df.columns:
        df['neutral_site'] = df['neutral_site'].astype(bool)

    return df

def parse_games_txt(filepath, aliases={}):
    """
    Parses a text file of weekly matchups into a structured list.
    Handles formats like "Away @ Home", "Home vs Away", and "Home vs Away (N)".

    Args:
        filepath (str): Path to the games.txt file.
        aliases (dict, optional): Dictionary to map nicknames to official names.

    Returns:
        list: A list of dictionaries, each representing a game.
    """
    games = []
    
    # Regex for "Team vs Team (N)"
    neutral_pattern = re.compile(r"^(.*) vs (.*) \(N\)$", re.IGNORECASE)
    # Regex for "Away @ Home"
    away_home_pattern = re.compile(r"^(.*) @ (.*)$", re.IGNORECASE)
    # Regex for "Home vs Away"
    home_away_pattern = re.compile(r"^(.*) vs (.*)$", re.IGNORECASE)
    # Regex for "Home, Away"
    comma_pattern = re.compile(r"^(.*), (.*)$", re.IGNORECASE)

    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                home, away, neutral = None, None, False

                # Try patterns in order of specificity
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
                    # Apply aliases to normalize team names
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

    Args:
        filepath (str): Path to the aliases.json file.

    Returns:
        dict: A dictionary of aliases. Returns an empty dict if file not found.
    """
    try:
        with open(filepath, 'r') as f:
            # Read and convert all keys to lowercase for case-insensitive matching
            return {k.lower(): v for k, v in json.load(f).items()}
    except FileNotFoundError:
        return {}
