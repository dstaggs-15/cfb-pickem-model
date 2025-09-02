# scripts/lib/elo.py

import numpy as np
import pandas as pd

HFA = 65      # Home field advantage in Elo points
REGRESS = 0.5 # Regress 50% to the mean each off-season

def elo_prob(elo1, elo2):
    """Calculates the probability of elo1 winning against elo2."""
    return 1 / (10 ** (-(elo1 - elo2) / 400) + 1)

def update_elo(winner_elo, loser_elo, week):
    """Updates Elo ratings for a winner and loser with a dynamic K-factor."""
    # Use a higher K-factor in the first 4 weeks to allow for faster adjustments
    k_factor = 32 if week <= 4 else 25
    prob = elo_prob(winner_elo, loser_elo)
    update = k_factor * (1 - prob)
    return winner_elo + update, loser_elo - update

def pregame_probs(schedule, talent_df, predict_df=None):
    """
    Calculates pre-game Elo probabilities for all games.
    """
    df = schedule.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date')

    elos = {}
    current_season = None

    for i, row in df.iterrows():
        # Regress to the mean at the start of a new season
        if row['season'] != current_season:
            current_season = row['season']
            for team in elos:
                elos[team] = (1 - REGRESS) * elos[team] + REGRESS * 1500

        home_team, away_team = row['home_team'], row['away_team']
        home_elo = elos.get(home_team, 1500)
        away_elo = elos.get(away_team, 1500)

        # Apply HFA only if it's not a neutral site game
        if not row['neutral_site']:
            home_elo += HFA

        df.loc[i, 'elo_home_prob'] = elo_prob(home_elo, away_elo)
        
        # Update Elos based on game result
        if pd.notna(row['home_points']):
            game_week = row['week']
            if row['home_points'] > row['away_points']: # Home wins
                new_home_elo, new_away_elo = update_elo(home_elo, away_elo, game_week)
            else: # Away wins
                new_away_elo, new_home_elo = update_elo(away_elo, home_elo, game_week)
            
            elos[home_team] = new_home_elo
            elos[away_team] = new_away_elo
            
    if predict_df is not None:
        for i, row in predict_df.iterrows():
            home_team, away_team = row['home_team'], row['away_team']
            home_elo = elos.get(home_team, 1500)
            away_elo = elos.get(away_team, 1500)
            
            if not row['neutral_site']:
                home_elo += HFA
            
            predict_df.loc[i, 'elo_home_prob'] = elo_prob(home_elo, away_elo)
        return predict_df[['game_id', 'elo_home_prob']]
        
    return df[['game_id', 'elo_home_prob']]
