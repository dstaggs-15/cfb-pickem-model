import pandas as pd
import numpy as np
import os
from .rolling import long_stats_to_wide, build_sidewise_rollups, STAT_FEATURES
from .context import rest_and_travel
from .market import median_lines
from .elo import pregame_probs

MANUAL_LINES_CSV = "docs/input/lines.csv"

def create_feature_set(schedule, team_stats, venues_df, teams_df, talent_df, lines_df, games_to_predict_df=None):
    """
    Single source of truth for feature engineering.
    """
    print("  Creating feature set...")
    LAST_N = 5
    
    team_stats = team_stats.drop_duplicates(subset=['game_id', 'team'])
    
    home_team_map = schedule[['game_id', 'home_team']]
    team_stats_sided = team_stats.merge(home_team_map, on='game_id', how='left')
    team_stats_sided['home_away'] = np.where(team_stats_sided['team'] == team_stats_sided['home_team'], 'home', 'away')
    team_stats_sided = team_stats_sided.drop(columns=['home_team'])
    
    wide_stats = long_stats_to_wide(team_stats_sided)

    home_roll, away_roll = build_sidewise_rollups(schedule, wide_stats, LAST_N, games_to_predict_df)

    base_df = schedule if games_to_predict_df is None else games_to_predict_df
    X = base_df.merge(home_roll, left_on=['game_id', 'home_team'], right_on=['game_id', 'team'], how='left').drop(columns=['team'], errors='ignore')
    X = X.merge(away_roll, left_on=['game_id', 'away_team'], right_on=['game_id', 'team'], how='left').drop(columns=['team'], errors='ignore')

    diff_cols = []
    for c in STAT_FEATURES:
        hc, ac = f"home_R{LAST_N}_{c}", f"away_R{LAST_N}_{c}"
        dc = f"diff_R{LAST_N}_{c}"
        if hc in X.columns and ac in X.columns:
            X[dc] = X[hc] - X[ac]
            diff_cols.append(dc)

    eng = rest_and_travel(schedule, teams_df, venues_df, games_to_predict_df)
    X = X.merge(eng, on="game_id", how="left")
    
    elo_df = pregame_probs(schedule, talent_df, games_to_predict_df)
    X = X.merge(elo_df, on="game_id", how="left")
    
    manual_lines_df = pd.read_csv(MANUAL_LINES_CSV) if os.path.exists(MANUAL_LINES_CSV) else pd.DataFrame()
    if games_to_predict_df is not None and not manual_lines_df.empty:
        manual_lines_df.rename(columns={'spread': 'spread_home'}, inplace=True)
        X = X.merge(manual_lines_df[['home_team', 'away_team', 'spread_home']], on=['home_team', 'away_team'], how='left')
    else:
        med = median_lines(lines_df)
        merge_keys = 'game_id' if games_to_predict_df is None else ['home_team', 'away_team']
        # Ensure merge keys exist before merging
        if all(key in med.columns for key in np.atleast_1d(merge_keys)):
             X = X.merge(med, on=merge_keys, how='left')
        else: # Fallback if keys are missing
             X['spread_home'] = np.nan
             X['over_under'] = np.nan


    count_features = [f"home_R{LAST_N}_count", f"away_R{LAST_N}_count"]
    ENG_FEATURES_BASE = ["rest_diff", "travel_away_km", "neutral_site", "is_postseason"]
    
    feature_list = diff_cols + count_features + ENG_FEATURES_BASE + ["elo_home_prob"]
    
    return X, feature_list
