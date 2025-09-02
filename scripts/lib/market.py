# scripts/lib/market.py

import numpy as np
import pandas as pd
from scipy.optimize import minimize

def median_lines(lines_df):
    """
    Calculates median betting lines for each game.
    """
    if lines_df.empty or 'spread' not in lines_df.columns:
        return pd.DataFrame(columns=['game_id', 'spread_home', 'over_under'])

    lines_df['spread_home'] = pd.to_numeric(lines_df['spread'], errors='coerce')
    lines_df['over_under'] = pd.to_numeric(lines_df['overUnder'], errors='coerce')
    
    # Group by game and take the median line from all providers
    median_lines = lines_df.groupby('game_id')[['spread_home', 'over_under']].median()
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
    spreads = spreads[~np.isnan(spreads)]
    outcomes = outcomes[~np.isnan(spreads)]

    initial_guess = [0.0, 0.15] # Initial parameters for a and b
    result = minimize(_log_loss, initial_guess, args=(spreads, outcomes), method='L-BFGS-B')
    
    a, b = result.x
    return {'a': a, 'b': b}
