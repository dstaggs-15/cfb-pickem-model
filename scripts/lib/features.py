import pandas as pd
import numpy as np
from .rolling import long_stats_to_wide, build_sidewise_rollups, STAT_FEATURES
from .context import rest_and_travel
from .market import median_lines, fit_market_mapping
from .elo import pregame_probs

def parse_possession_time(s):
    if not isinstance(s, str) or ':' not in s:
        return 0.0
    try:
        minutes, seconds = s.split(':')
        return int(minutes) * 60 + int(seconds)
    except (ValueError, TypeError):
        return 0.0

def create_feature_set(schedule, team_stats_long, venues_df, teams_df, talent_df, lines_df, games_to_predict_df=None):
    """
    This is the single source of truth for feature engineering.
    It takes raw dataframes and returns a clean feature set X.
    """
    print("  Creating feature set...")
    
    # 1. Pivot and clean raw team stats
    team_stats_long['stat_value'] = pd.to_numeric(team_stats_long['stat_value'], errors='coerce')
    team_stats = team_stats_long.pivot_table(
        index=['game_id', 'team'], columns='category', values='stat_value'
    ).reset_index()

    def camel_to_snake(name):
        return ''.join(['_' + c.lower() if c.isupper() else c for c in name]).lstrip('_')
    team_stats.columns = [camel_to_snake(col) for col in team_stats.columns]

    # 2. Create derived stats
    rushing_attempts = team_stats.get('rushing_attempts', pd.Series(0, index=team_stats.index))
    pass_attempts = team_stats.get('pass_attempts', pd.Series(0, index=team_stats.index))
    total_yards = team_stats.get('total_yards', pd.Series(0, index=team_stats.index))
    first_downs = team_stats.get('first_downs', pd.Series(0, index=team_stats.index))
    yards_per_pass = team_stats.get('yards_per_pass', pd.Series(0, index=team_stats.index))
    yards_per_rush_attempt = team_stats.get('yards_per_rush_attempt', pd.Series(0, index=team_stats.index))
    
    total_plays = rushing_attempts.fillna(0) + pass_attempts.fillna(0)
    
    team_stats['ppa'] = (total_yards.fillna(0) / total_plays.replace(0, np.nan)).fillna(0)
    team_stats['success_rate'] = (first_downs.fillna(0) / total_plays.replace(0, np.nan)).fillna(0)
    team_stats['explosiveness'] = (yards_per_pass.fillna(0) * 0.5 + yards_per_rush_attempt.fillna(0) * 0.5)
    
    if 'possession_time' in team_stats.columns:
        team_stats['possession_seconds'] = team_stats['possession_time'].apply(parse_possession_time)
        team_stats.drop(columns=['possession_time'], inplace=True, errors='ignore')
    else:
        team_stats['possession_seconds'] = 0.0

    # 3. Build rolling average features
    home_team_map = schedule[['game_id', 'home_team']]
    team_stats_sided = team_stats.merge(home_team_map, on='game_id', how='left')
    team_stats_sided['home_away'] = np.where(team_stats_sided['team'] == team_stats_sided['home_team'], 'home', 'away')
    team_stats_sided = team_stats_sided.drop(columns=['home_team'])
    
    wide_stats = long_stats_to_wide(team_stats_sided)
    home_roll, away_roll = build_sidewise_rollups(schedule, wide_stats, 5, games_to_predict_df)

    # 4. Merge all features together
    base_df = schedule if games_to_predict_df is None else games_to_predict_df
    X = base_df.merge(home_roll, left_on=['game_id', 'home_team'], right_on=['game_id', 'team'], how='left').drop(columns=['team'], errors='ignore')
    X = X.merge(away_roll, left_on=['game_id', 'away_team'], right_on=['game_id', 'team'], how='left').drop(columns=['team'], errors='ignore')
    
    diff_cols = []
    for c in STAT_FEATURES:
        hc, ac = f"home_R5_{c}", f"away_R5_{c}"
        dc = f"diff_R5_{c}"
        if hc in X.columns and ac in X.columns:
            X[dc] = X[hc] - X[ac]
            diff_cols.append(dc)

    eng = rest_and_travel(schedule, teams_df, venues_df, games_to_predict_df)
    X = X.merge(eng, on="game_id", how="left")
    
    elo_df = pregame_probs(schedule, talent_df, games_to_predict_df)
    X = X.merge(elo_df, on="game_id", how="left")

    # This logic is for training only
    if games_to_predict_df is None:
        X["home_win"] = (pd.to_numeric(X["home_points"], errors="coerce") > pd.to_numeric(X["away_points"], errors="coerce")).astype(int)
    
    # Market features (handle manual lines for prediction)
    if games_to_predict_df is not None and 'manual_lines_df' in locals():
         manual_lines_df.rename(columns={'spread': 'spread_home'}, inplace=True)
         X = X.merge(manual_lines_df[['home_team', 'away_team', 'spread_home']], on=['home_team', 'away_team'], how='left')
    else:
        med = median_lines(lines_df)
        X = X.merge(med, on="game_id", how="left")

    return X, diff_cols
