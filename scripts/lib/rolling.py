# scripts/lib/rolling.py

import pandas as pd
from .parsing import parse_ratio_val

# Define the list of statistical features to be used for rolling averages
STAT_FEATURES = [
    "ppa", "success_rate", "explosiveness", "power_success", "stuff_rate", 
    "line_yards", "second_level_yards", "open_field_yards", "points_per_opportunity",
    "havoc", "turnovers", "field_pos_avg_start"
]

def long_stats_to_wide(team_stats):
    """Pivots the long-format team stats to a wide format."""
    # Pivot the data, which creates a multi-level column index
    pivoted = team_stats.pivot(
        index="game_id", 
        columns="home_away", 
        values=[c for c in team_stats.columns if c not in ["game_id", "home_away", "team"]]
    )
    
    # --- FIX STARTS HERE ---
    # Flatten the multi-level columns (e.g., ('ppa', 'home')) into a single level ('ppa_home')
    pivoted.columns = [f'{col[0]}_{col[1]}' for col in pivoted.columns]
    # --- FIX ENDS HERE ---
    
    return pivoted

def _get_rollups(df, last_n):
    """Helper to compute rolling stats for a given DataFrame of one-sided stats."""
    # Ensure data is sorted by team and date to get correct rolling window
    df = df.sort_values(by=["team", "date"])
    
    # Group by team and calculate rolling average, shifting to prevent data leakage
    grp = df.groupby("team")[STAT_FEATURES]
    rollups = grp.rolling(window=last_n, min_periods=1).mean().shift(1)
    
    # Calculate rolling count
    counts = grp.cumcount().rename(f"R{last_n}_count") + 1
    counts[counts > last_n] = last_n
    
    # Rename columns to reflect the rolling window size
    rollups.columns = [f"R{last_n}_{c}" for c in STAT_FEATURES]
    
    # Combine the rolling stats and counts
    final = pd.concat([df[["game_id", "team"]], rollups, counts], axis=1)
    return final.reset_index(drop=True)

def build_sidewise_rollups(schedule, wide_stats, last_n, predict_df=None):
    """
    Builds rolling average features for teams based on their side (home/away).
    """
    # Merge schedule and stats
    schedule['date'] = pd.to_datetime(schedule['date'])
    full_df = schedule.merge(wide_stats, on="game_id", how="left")

    # Prepare home stats DataFrame
    home_df = full_df[["game_id", "date", "home_team"]].rename(columns={"home_team": "team"})
    home_stats_cols = [c for c in full_df.columns if c.endswith('_home')]
    home_stats = full_df[home_stats_cols]
    home_stats.columns = [c.replace('_home', '') for c in home_stats.columns]
    home_df = pd.concat([home_df, home_stats], axis=1)

    # Prepare away stats DataFrame
    away_df = full_df[["game_id", "date", "away_team"]].rename(columns={"away_team": "team"})
    away_stats_cols = [c for c in full_df.columns if c.endswith('_away')]
    away_stats = full_df[away_stats_cols]
    away_stats.columns = [c.replace('_away', '') for c in away_stats.columns]
    away_df = pd.concat([away_df, away_stats], axis=1)

    # If predicting, append historical data to the prediction set to form a complete timeline
    if predict_df is not None:
        predict_df['date'] = pd.to_datetime("2099-01-01") # Ensure predictions are last
        home_predict = predict_df[["game_id", "date", "home_team"]].rename(columns={"home_team": "team"})
        away_predict = predict_df[["game_id", "date", "away_team"]].rename(columns={"away_team": "team"})
        
        home_df = pd.concat([home_df, home_predict])
        away_df = pd.concat([away_df, away_predict])

    # Calculate rollups for each side
    home_rollups = _get_rollups(home_df, last_n)
    away_rollups = _get_rollups(away_df, last_n)

    # If predicting, filter to return only the stats for the prediction games
    if predict_df is not None:
        game_ids_to_predict = predict_df["game_id"].unique()
        home_rollups = home_rollups[home_rollups["game_id"].isin(game_ids_to_predict)]
        away_rollups = away_rollups[away_rollups["game_id"].isin(game_ids_to_predict)]

    return home_rollups, away_rollups
