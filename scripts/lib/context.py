# scripts/lib/context.py

import numpy as np
import pandas as pd
from haversine import haversine

def rest_and_travel(schedule, teams_df, venues_df, predict_df=None):
    """
    Engineers context features like rest, travel distance, and special game flags.
    """
    df = schedule if predict_df is None else pd.concat([schedule, predict_df])
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date')

    # Calculate days of rest
    df['last_game_date_home'] = df.groupby('home_team')['date'].shift(1)
    df['last_game_date_away'] = df.groupby('away_team')['date'].shift(1)
    df['rest_home'] = (df['date'] - df['last_game_date_home']).dt.days
    df['rest_away'] = (df['date'] - df['last_game_date_away']).dt.days

    # Merge location data
    if not teams_df.empty and 'latitude' in teams_df.columns:
        teams_loc = teams_df[['school', 'latitude', 'longitude']].rename(columns={'school': 'team'})
        home_loc = teams_loc.rename(columns={'team': 'home_team', 'latitude': 'home_lat', 'longitude': 'home_lon'})
        away_loc = teams_loc.rename(columns={'team': 'away_team', 'latitude': 'away_lat', 'longitude': 'away_lon'})
        df = df.merge(home_loc, on='home_team', how='left')
        df = df.merge(away_loc, on='away_team', how='left')

    if not venues_df.empty and 'latitude' in venues_df.columns:
        venues_loc = venues_df[['id', 'latitude', 'longitude']].rename(columns={'id': 'venue_id', 'latitude': 'venue_lat', 'longitude': 'venue_lon'})
        df = df.merge(venues_loc, on='venue_id', how='left')

    # Calculate travel distance
    def calculate_travel(row):
        # No travel for home team
        home_travel = 0
        
        # Away team travels from their location to the venue location
        if pd.notna(row['away_lat']) and pd.notna(row['venue_lat']):
            away_loc = (row['away_lat'], row['away_lon'])
            venue_loc = (row['venue_lat'], row['venue_lon'])
            away_travel = haversine(away_loc, venue_loc)
        else:
            away_travel = np.nan
        
        return home_travel, away_travel

    if 'away_lat' in df.columns and 'venue_lat' in df.columns:
        travel_distances = df.apply(calculate_travel, axis=1, result_type='expand')
        df['travel_home_km'] = travel_distances[0]
        df['travel_away_km'] = travel_distances[1]
    else:
        df['travel_home_km'] = 0
        df['travel_away_km'] = np.nan


    # Feature differences and flags
    df['rest_diff'] = df['rest_home'] - df['rest_away']
    df['travel_diff_km'] = df['travel_home_km'] - df['travel_away_km']
    df['shortweek_home'] = (df['rest_home'] < 7).astype(int)
    df['shortweek_away'] = (df['rest_away'] < 7).astype(int)
    df['shortweek_diff'] = df['shortweek_home'] - df['shortweek_away']
    df['bye_home'] = (df['rest_home'] > 9).astype(int)
    df['bye_away'] = (df['rest_away'] > 9).astype(int)
    df['bye_diff'] = df['bye_home'] - df['bye_away']
    df['is_postseason'] = df['season_type'].str.contains('post', case=False, na=False).astype(int)

    feature_cols = [
        'game_id', 'rest_diff', 'shortweek_diff', 'bye_diff', 'travel_diff_km', 
        'neutral_site', 'is_postseason'
    ]
    
    return df[feature_cols]
