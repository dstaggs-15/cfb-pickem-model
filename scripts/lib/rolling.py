import pandas as pd
import numpy as np
import os

# --- Define path for the season averages file ---
DERIVED = "data/derived"
SEASON_AVG_PARQUET = f"{DERIVED}/season_averages.parquet"

# Define the list of statistical features to be used for rolling averages
STAT_FEATURES = [
    "ppa", "success_rate", "explosiveness", "power_success", "stuff_rate",
    "line_yards", "second_level_yards", "open_field_yards", "points_per_opportunity",
    "havoc", "turnovers", "field_pos_avg_start"
]

def long_stats_to_wide(team_stats):
    """Pivots the long-format team stats to a wide format."""
    pivoted = team_stats.pivot(
        index="game_id",
        columns="home_away",
        values=[c for c in team_stats.columns if c not in ["game_id", "home_away", "team"]]
    )
    pivoted.columns = [f'{col[0]}_{col[1]}' for col in pivoted.columns]
    return pivoted

def _get_rollups(df, last_n, season_averages_df):
    """
    Helper to compute rolling stats, now with season-average carry-forward logic.
    """
    df = df.sort_values(by=["team", "date"]).reset_index(drop=True)
    
    # This function will be applied to each team's data
    def team_rollup(team_df):
        seasons = team_df['season'].unique()
        all_season_rollups = []

        for season in seasons:
            season_df = team_df[team_df['season'] == season].copy()
            prior_season = season - 1
            
            # Get the previous season's averages for this team
            prior_season_avg = season_averages_df[
                (season_averages_df['team'] == team_df['team'].iloc[0]) &
                (season_averages_df['season'] == prior_season)
            ]

            # Create "padding" rows from last season's averages
            padding_rows = []
            if not prior_season_avg.empty:
                avg_stats = prior_season_avg[STAT_FEATURES].iloc[0]
                for _ in range(last_n):
                    padding_rows.append(avg_stats)
            
            # If we have padding, prepend it to this season's data
            if padding_rows:
                padding_df = pd.DataFrame(padding_rows)
                padded_season_data = pd.concat([padding_df, season_df[STAT_FEATURES]], ignore_index=True)
            else:
                padded_season_data = season_df[STAT_FEATURES]
            
            # Now, calculate the rolling mean on the padded data
            rolling_stats = padded_season_data.rolling(window=last_n, min_periods=1).mean()
            
            # Slice the results to match the original number of games for this season
            # and apply the .shift(1) to prevent data leakage
            final_rolling_stats = rolling_stats.iloc[-len(season_df):].shift(1)
            final_rolling_stats.columns = [f"R{last_n}_{c}" for c in STAT_FEATURES]
            
            # Calculate the rolling count
            counts = pd.Series(range(1, len(season_df) + 1), index=season_df.index)
            counts[counts > last_n] = last_n
            final_rolling_stats[f"R{last_n}_count"] = counts
            
            # Combine with game_id and team
            season_rollup = pd.concat([season_df[['game_id', 'team']].reset_index(drop=True), final_rolling_stats.reset_index(drop=True)], axis=1)
            all_season_rollups.append(season_rollup)
        
        return pd.concat(all_season_rollups, ignore_index=True)

    # Apply the custom rollup function to each team
    final_rollups = df.groupby('team', group_keys=False).apply(team_rollup)
    
    return final_rollups

def build_sidewise_rollups(schedule, wide_stats, last_n, predict_df=None):
    """
    Builds rolling average features for teams based on their side (home/away).
    """
    # Load the pre-calculated season averages
    season_averages_df = pd.read_parquet(SEASON_AVG_PARQUET) if os.path.exists(SEASON_AVG_PARQUET) else pd.DataFrame()

    schedule['date'] = pd.to_datetime(schedule['date'])
    # Add season to wide_stats for the join
    full_df = schedule.merge(wide_stats, on="game_id", how="left")

    home_df = full_df[["game_id", "date", "season", "home_team"]].rename(columns={"home_team": "team"})
    home_stats_cols = [c for c in full_df.columns if c.endswith('_home')]
    home_stats = full_df[home_stats_cols]
    home_stats.columns = [c.replace('_home', '') for c in home_stats.columns]
    home_df = pd.concat([home_df, home_stats], axis=1)

    away_df = full_df[["game_id", "date", "season", "away_team"]].rename(columns={"away_team": "team"})
    away_stats_cols = [c for c in full_df.columns if c.endswith('_away')]
    away_stats = full_df[away_stats_cols]
    away_stats.columns = [c.replace('_away', '') for c in away_stats.columns]
    away_df = pd.concat([away_df, away_stats], axis=1)

    if predict_df is not None:
        predict_df['date'] = pd.to_datetime("2099-01-01")
        # Ensure predict_df has a season column
        if 'season' not in predict_df.columns:
            predict_df['season'] = schedule['season'].max()
            
        home_predict = predict_df[["game_id", "date", "season", "home_team"]].rename(columns={"home_team": "team"})
        away_predict = predict_df[["game_id", "date", "season", "away_team"]].rename(columns={"away_team": "team"})
        
        home_df = pd.concat([home_df, home_predict])
        away_df = pd.concat([away_df, away_predict])

    home_rollups = _get_rollups(home_df.dropna(subset=['team']), last_n, season_averages_df)
    away_rollups = _get_rollups(away_df.dropna(subset=['team']), last_n, season_averages_df)

    if predict_df is not None:
        game_ids_to_predict = predict_df["game_id"].unique()
        home_rollups = home_rollups[home_rollups["game_id"].isin(game_ids_to_predict)]
        away_rollups = away_rollups[away_rollups["game_id"].isin(game_ids_to_predict)]

    return home_rollups, away_rollups
