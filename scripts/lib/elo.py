# scripts/lib/elo.py

import numpy as np
import pandas as pd

# --- CONFIGURATION CONSTANTS ---
HFA = 65  # Home field advantage in Elo points
REGRESSION_FACTOR = 0.5 # Regress 50% to the mean each off-season
EARLY_SEASON_CUTOFF_WEEK = 4 # Use prior-season Elo for features up to and including this week

def elo_prob(elo1, elo2):
    """Calculates the probability of elo1 winning against elo2."""
    return 1 / (10 ** (-(elo1 - elo2) / 400) + 1)

def update_elo(winner_elo, loser_elo, week):
    """Updates Elo ratings for a winner and loser with a dynamic K-factor."""
    k_factor = 32 if week <= 4 else 25
    prob = elo_prob(winner_elo, loser_elo)
    update = k_factor * (1 - prob)
    return winner_elo + update, loser_elo - update

def pregame_probs(schedule, talent_df, predict_df=None):
    """
    Calculates pre-game Elo probabilities for all games using the two-phase approach.
    - Phase 1 (Early Season): Uses the final Elo from the *prior season* as a static feature.
    - Phase 2 (Mid/Late Season): Uses the dynamically updated *current season* Elo.
    """
    df = schedule.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date')

    # --- NEW: Dictionaries to store Elo ratings ---
    # `current_elos` will be dynamically updated throughout each season
    current_elos = {} 
    # `season_end_elos` will store the final Elo for each team at the end of a season
    season_end_elos = {}

    current_season = None

    for i, row in df.iterrows():
        # --- OFF-SEASON LOGIC (runs at the start of each new season) ---
        if row['season'] != current_season:
            # NEW: Store the final Elo ratings from the completed season
            if current_season is not None:
                season_end_elos[current_season] = current_elos.copy()
            
            current_season = row['season']
            
            # Regress the dynamic Elos to the mean for the new season's calculations
            for team in current_elos:
                current_elos[team] = (1 - REGRESSION_FACTOR) * current_elos[team] + REGRESSION_FACTOR * 1500

        home_team, away_team = row['home_team'], row['away_team']
        
        # --- NEW: TWO-PHASE FEATURE CALCULATION ---
        # This is the core of the new system. We decide which Elo value to use for the feature.
        
        feature_home_elo, feature_away_elo = 0, 0
        game_week = row['week']

        if game_week <= EARLY_SEASON_CUTOFF_WEEK:
            # PHASE 1: Use static Elo from the end of the prior season
            prior_season = row['season'] - 1
            if prior_season in season_end_elos:
                feature_home_elo = season_end_elos[prior_season].get(home_team, 1500)
                feature_away_elo = season_end_elos[prior_season].get(away_team, 1500)
            else: # Fallback for the very first season in the dataset
                feature_home_elo = 1500
                feature_away_elo = 1500
        else:
            # PHASE 2: Use the dynamically updating Elo from the current season
            feature_home_elo = current_elos.get(home_team, 1500)
            feature_away_elo = current_elos.get(away_team, 1500)

        # Apply HFA to the chosen feature Elo
        if not row['neutral_site']:
            feature_home_elo += HFA
            
        # Calculate the pre-game probability using the feature Elos
        df.loc[i, 'elo_home_prob'] = elo_prob(feature_home_elo, feature_away_elo)
        
        # --- DYNAMIC ELO UPDATE (Always runs in the background) ---
        # This ensures our `current_elos` are always up-to-date and ready for Phase 2.
        
        if pd.notna(row['home_points']):
            # Get the actual current Elos for the update calculation
            home_elo_for_update = current_elos.get(home_team, 1500)
            away_elo_for_update = current_elos.get(away_team, 1500)

            # Temporarily add HFA for the update calculation
            if not row['neutral_site']:
                home_elo_for_update += HFA

            if row['home_points'] > row['away_points']: # Home wins
                new_home_elo, new_away_elo = update_elo(home_elo_for_update, away_elo_for_update, game_week)
            else: # Away wins
                new_away_elo, new_home_elo = update_elo(away_elo_for_update, home_elo_for_update, game_week)
            
            # Remove the temporary HFA before saving the new base Elo
            if not row['neutral_site']:
                new_home_elo -= HFA

            current_elos[home_team] = new_home_elo
            current_elos[away_team] = new_away_elo
            
    # --- PREDICTION LOGIC (for new, unplayed games) ---
    if predict_df is not None:
        # Get the very last calculated season to determine the prior season
        last_historical_season = df['season'].max()
        
        for i, row in predict_df.iterrows():
            home_team, away_team = row['home_team'], row['away_team']
            
            # For predictions, we assume it's early in the current season (e.g., 2025)
            # and use the final Elos from the last fully completed season (2024).
            final_elos_for_prediction = season_end_elos.get(last_historical_season, current_elos)
            
            home_elo = final_elos_for_prediction.get(home_team, 1500)
            away_elo = final_elos_for_prediction.get(away_team, 1500)
            
            if not row['neutral_site']:
                home_elo += HFA
            
            predict_df.loc[i, 'elo_home_prob'] = elo_prob(home_elo, away_elo)
        return predict_df[['game_id', 'elo_home_prob']]
        
    return df[['game_id', 'elo_home_prob']]
