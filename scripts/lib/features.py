import pandas as pd
import numpy as np
from .rolling import long_stats_to_wide, build_sidewise_rollups, STAT_FEATURES
from .context import rest_and_travel
from .market import median_lines
from .elo import pregame_probs

def parse_possession_time(s):
    if not isinstance(s, str) or ':' not in s:
        return 0.0
    try:
        minutes, seconds = s.split(':')
        return int(minutes) * 60 + int(seconds)
    except (ValueError, TypeError):
        return 0.0

def create_feature_set(schedule, team_stats, venues_df, teams_df, talent_df, lines_df, games_to_predict_df=None):
    """
    Single source of truth for feature engineering.
    Takes raw dataframes and returns a clean feature set X and a list of feature columns.
    """
    print("  Creating feature set...")

    LAST_N = 5
    
    # 1. Prepare base stats DataFrame
    home_team_map = schedule[['game_id', 'home_team']]
    team_stats_sided = team_stats.merge(home_team_map, on='game_id', how='left')
    team_stats_sided['home_away'] = np.where(team_stats_sided['team'] == team_stats_sided['home_team'], 'home', 'away')
    team_stats_sided = team_stats_sided.drop(columns=['home_team'])
    
    wide_stats = long_stats_to_wide(team_stats_sided)

    # 2. Build rolling features
    home_roll, away_roll = build_sidewise_rollups(schedule, wide_stats, LAST_N, games_to_predict_df)

    # 3. Join all features together
    base_df = schedule if games_to_predict_df is None else games_to_predict_df
    X = base_df.merge(home_roll, left_on=['game_id', 'home_team'], right_on=['game_id', 'team'], how='left').drop(columns=['team'], errors='ignore')
    X = X.merge(away_roll, left_on=['game_id', 'away_team'], right_on=['game_id', 'team'], how='left').drop(columns=['team'], errors='ignore')

    # 4. Create difference columns
    diff_cols = []
    for c in STAT_FEATURES:
        hc, ac = f"home_R{LAST_N}_{c}", f"away_R{LAST_N}_{c}"
        dc = f"diff_R{LAST_N}_{c}"
        if hc in X.columns and ac in X.columns:
            X[dc] = X[hc] - X[ac]
            diff_cols.append(dc)

    # 5. Add other feature types
    eng = rest_and_travel(schedule, teams_df, venues_df, games_to_predict_df)
    X = X.merge(eng, on="game_id", how="left")
    
    elo_df = pregame_probs(schedule, talent_df, games_to_predict_df)
    X = X.merge(elo_df, on="game_id", how="left")

    # 6. Define final feature list
    count_features = [f"home_R{LAST_N}_count", f"away_R{LAST_N}_count"]
    ENG_FEATURES_BASE = ["rest_diff", "travel_away_km", "neutral_site", "is_postseason"]
    
    feature_list = diff_cols + count_features + ENG_FEATURES_BASE + ["elo_home_prob"]
    
    return X, feature_list
