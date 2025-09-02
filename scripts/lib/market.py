# scripts/lib/market.py

import numpy as np
import pandas as pd
from scipy.optimize import minimize

def median_lines(lines_df):
    """
    Calculates median betting lines for each game.
    """
    if lines_df.empty or 'spread' not in lines_df.columns:
        # Return a DataFrame with all necessary columns for merging
        return pd.DataFrame(columns=['home_team', 'away_team', 'spread_home', 'over_under'])

    # Standardize column names from common sources
    lines_df = lines_df.rename(columns={'spread': 'spread_home', 'overUnder': 'over_under'})

    # Ensure required columns exist
    if 'home_team' not in lines_df.columns or 'away_team' not in lines_df.columns:
        if 'game_id' in lines_df.columns:
             # Fallback for historical data that is game_id based
             return lines_df.groupby('game_id')[['spread_home', 'over_under']].median().reset_index()
        else:
             return pd.DataFrame(columns=['home_team', 'away_team', 'spread_home', 'over_under'])

    lines_df['spread_home'] = pd.to_numeric(lines_df['spread_home'], errors='coerce')
    lines_df['over_under'] = pd.to_numeric(lines_df['over_under'], errors='coerce')
    
    # --- THIS IS THE FIX ---
    # Group by team names to keep them in the output, which is needed for prediction.
    median_lines = lines_df.groupby(['home_team', 'away_team'])[['spread_home', 'over_under']].median()
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
    # Filter out NaNs before fitting
    valid_indices = ~np.isnan(spreads) & ~np.isnan(outcomes)
    spreads = spreads[valid_indices]
    outcomes = outcomes[valid_indices]

    if len(spreads) == 0:
        return {'a': 0.0, 'b': 0.15} # Return default if no data

    initial_guess = [0.0, 0.15]
    result = minimize(_log_loss, initial_guess, args=(spreads, outcomes), method='L-BFGS-B')
    
    a, b = result.x
    return {'a': a, 'b': b}
