# scripts/lib/market.py

import numpy as np
import pandas as pd
from scipy.optimize import minimize

def median_lines(lines_df):
    """
    Calculates median betting lines for each game, preserving all necessary keys.
    """
    if lines_df.empty or 'spread' not in lines_df.columns:
        return pd.DataFrame(columns=['game_id', 'home_team', 'away_team', 'spread_home', 'over_under'])

    # Standardize column names
    lines_df = lines_df.rename(columns={
        'spread': 'spread_home', 
        'overUnder': 'over_under',
        'homeTeam': 'home_team',
        'awayTeam': 'away_team'
    })

    # Ensure required columns for grouping exist
    required_keys = ['game_id', 'home_team', 'away_team']
    if not all(key in lines_df.columns for key in required_keys):
        print("Warning: Lines file is missing required keys (game_id, home_team, away_team).")
        return pd.DataFrame(columns=['game_id', 'home_team', 'away_team', 'spread_home', 'over_under'])

    lines_df['spread_home'] = pd.to_numeric(lines_df['spread_home'], errors='coerce')
    lines_df['over_under'] = pd.to_numeric(lines_df['over_under'], errors='coerce')
    
    # --- THIS IS THE FIX ---
    # Group by all keys to ensure the output is compatible with both training and prediction.
    median_lines = lines_df.groupby(required_keys)[['spread_home', 'over_under']].median()
    return median_lines.reset_index()

def _log_loss(params, spreads, outcomes):
    """Log loss function for fitting spread-to-probability mapping."""
    a, b = params
    probs = 1 / (1 + np.exp(-(a + b * (-spreads))))
    return -np.mean(outcomes * np.log(probs) + (1 - outcomes) * np.log(1 - probs))

def fit_market_mapping(spreads, outcomes):
    """
    Learns a logistic function to map betting spreads to win probabilities.
    p(win) = 1 / (1 + exp(-(a + b * (-spread))))
    """
    valid_indices = ~np.isnan(spreads) & ~np.isnan(outcomes)
    spreads = spreads[valid_indices]
    outcomes = outcomes[valid_indices]

    if len(spreads) < 10: # Need some data to fit
        return {'a': 0.0, 'b': 0.15} # Return default if not enough data

    initial_guess = [0.0, 0.15]
    result = minimize(_log_loss, initial_guess, args=(spreads, outcomes), method='L-BFGS-B')
    
    a, b = result.x
    return {'a': a, 'b': b}
