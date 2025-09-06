import pandas as pd
import numpy as np
import os
import datetime as dt

DERIVED = "data/derived"
SEASON_AVG_PARQUET = f"{DERIVED}/season_averages.parquet"

STAT_FEATURES = [
    'ppa',
    'success_rate',
    'explosiveness',
    'rushing_yards',
    'net_passing_yards',
    'turnovers',
    'possession_seconds'
]

def long_stats_to_wide(team_stats):
    """Pivots the home/away team stats to a single row per game."""
    pivoted = team_stats.pivot(
        index="game_id",
        columns="home_away",
        values=[c for c in team_stats.columns if c not in ["game_id", "home_away", "team"]]
    )
    pivoted.columns = [f'{col[0]}_{col[1]}' for col in pivoted.columns]
    return pivoted

def _get_rollups(df, last_n, season_averages_df):
    """
    Helper to compute rolling stats, with season-average carry-forward logic.
    """
    # This sort is now safe because all dates are standardized
    df = df.sort_values(by=["date"]).reset_index(drop=True)
    
    def team_rollup(team_df, team_name):
        seasons = team_df['season'].unique()
        all_season_rollups = []
        
        existing_stat_features = [feat for feat in STAT_FEATURES if feat in team_df.columns]

        for season in seasons:
            season_df = team_df[team_df['season'] == season].copy()
            prior_season = season - 1
            
            prior_season_avg = season_averages_df[
                (season_averages_df['team'] == team_name) &
                (season_averages_df['season'] == prior_season)
            ]

            padding_rows = []
            if not prior_season_avg.empty:
                avg_stats = prior_season_avg[existing_stat_features].iloc[0]
                for _ in range(last_n):
                    padding_rows.append(avg_stats)
            
            if padding_rows:
                padding_df = pd.DataFrame(padding_rows)
                padded_season_data = pd.concat([padding_df, season_df[existing_stat_features]], ignore_index=True)
            else:
                padded_season_data = season_df[existing_stat_features]
            
            rolling_stats = padded_season_data.rolling(window=last_n, min_periods=1).mean()
            
            final_rolling_stats = rolling_stats.iloc[-len(season_df):].shift(1)
            final_rolling_stats.columns = [f"R{last_n}_{c}" for c in existing_stat_features]
            
            counts = pd.Series(range(1, len(season_df) + 1), index=season_df.index)
            counts[counts > last_n] = last_n
            final_rolling_stats[f"R{last_n}_count"] = counts
            
            season_rollup = pd.concat([season_df[['game_id']].reset_index(drop=True), final_rolling_stats.reset_index(drop=True)], axis=1)
            season_rollup['team'] = team_name
            all_season_rollups.append(season_rollup)
        
        if not all_season_rollups:
            return pd.DataFrame()
        return pd.concat(all_season_rollups, ignore_index=True)

    all_teams_rollups = []
    for team_name, team_df in df.groupby('team'):
        team_rollups = team_rollup(team_df, team_name)
        all_teams_rollups.append(team_rollups)
    
    if not all_teams_rollups:
        return pd.DataFrame()
    final_rollups = pd.concat(all_teams_rollups, ignore_index=True)
    
    return final_rollups

def build_sidewise_rollups(schedule, wide_stats, last_n, predict_df=None):
    season_averages_df = pd.read_parquet(SEASON_AVG_PARQUET) if os.path.exists(SEASON_AVG_PARQUET) else pd.DataFrame()

    # --- FIX IS HERE ---
    # Convert date column to datetime objects and standardize to UTC.
    # This handles mixed (naive and aware) timezones by making them all consistent.
    schedule['date'] = pd.to_datetime(schedule['date']).dt.tz_convert('UTC')

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
        # For prediction, ensure the new date is also timezone-aware to match
        predict_df['date'] = pd.to_datetime(dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=365))
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
