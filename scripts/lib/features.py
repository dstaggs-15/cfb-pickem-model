import numpy as np
import pandas as pd
from typing import Optional, Tuple, List

from .rolling import long_stats_to_wide, build_sidewise_rollups
from .context import rest_and_travel
from .market import median_lines
from .elo import pregame_probs


def create_feature_set(
    schedule: pd.DataFrame,
    team_stats: pd.DataFrame,
    venues_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    talent_df: pd.DataFrame,
    lines_df: pd.DataFrame,
    manual_lines_df: Optional[pd.DataFrame] = None,
    games_to_predict_df: Optional[pd.DataFrame] = None,
    last_n: int = 5,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Single source of truth for feature engineering.
    """
    print("  Creating feature set...")

    # --- FIX IS HERE ---
    # The raw data can contain duplicate entries for games. We remove them here
    # to ensure the pipeline is robust to data quality issues from the source.
    schedule = schedule.drop_duplicates(subset=['game_id'])
    team_stats = team_stats.drop_duplicates(subset=['game_id', 'team'])

    # 1) Get home team for each game to determine 'home' vs 'away' side for stats
    home_team_map = schedule[['game_id', 'home_team']]
    team_stats_sided = team_stats.merge(
        home_team_map, on='game_id', how='left', validate='many_to_one'
    )
    team_stats_sided['home_away'] = np.where(
        team_stats_sided['team'] == team_stats_sided['home_team'], 'home', 'away'
    )
    team_stats_sided = team_stats_sided.drop(columns=['home_team'])

    # Pivot the stats from long format to wide format (one row per game)
    wide_stats = long_stats_to_wide(team_stats_sided)

    # 2) Calculate rolling features for each team
    home_roll, away_roll = build_sidewise_rollups(
        schedule,
        wide_stats,
        last_n,
        games_to_predict_df
    )

    # Normalize column names for safe merges
    if 'team' in home_roll.columns:
        home_roll = home_roll.rename(columns={'team': 'home_team'})
    if 'team' in away_roll.columns:
        away_roll = away_roll.rename(columns={'team': 'away_team'})

    # 3) Use the correct base dataframe (all historical games for training, or new games for prediction)
    base_df = schedule if games_to_predict_df is None else games_to_predict_df
    
    required = {'game_id', 'home_team', 'away_team'}
    if not required.issubset(base_df.columns):
        raise ValueError(f"Base dataframe missing one of required columns: {required - set(base_df.columns)}")

    # 4) Merge all engineered features together
    X = base_df.merge(
        home_roll, on=['game_id', 'home_team'], how='left'
    )
    X = X.merge(
        away_roll, on=['game_id', 'away_team'], how='left'
    )
    
    # Define the list of stat features dynamically from the rolling average columns
    STAT_FEATURES = [col.replace(f"home_R{last_n}_", "") for col in home_roll.columns if col.startswith(f"home_R{last_n}_") and "_count" not in col]

    diff_cols: List[str] = []
    for c in STAT_FEATURES:
        hc = f"home_R{last_n}_{c}"
        ac = f"away_R{last_n}_{c}"
        dc = f"diff_R{last_n}_{c}"
        if hc in X.columns and ac in X.columns:
            X[dc] = X[hc] - X[ac]
            diff_cols.append(dc)

    eng = rest_and_travel(schedule, teams_df, venues_df, games_to_predict_df)
    X = X.merge(eng, on="game_id", how="left")

    elo_df = pregame_probs(schedule, talent_df, games_to_predict_df)
    X = X.merge(elo_df, on="game_id", how="left")

    if games_to_predict_df is not None and manual_lines_df is not None and not manual_lines_df.empty:
        mlines = manual_lines_df.rename(columns={'spread': 'spread_home'}).copy()
        X = X.merge(mlines[['home_team', 'away_team', 'spread_home']], on=['home_team', 'away_team'], how='left')
    else:
        med = median_lines(lines_df)
        if 'game_id' in med.columns:
            X = X.merge(med, on='game_id', how='left')
        else:
            X = X.merge(med, on=['home_team', 'away_team'], how='left')

    # 5) Define the final list of features for the model
    count_features = [f"home_R{last_n}_count", f"away_R{last_n}_count"]
    ENG_FEATURES_BASE = ["rest_diff", "travel_away_km", "neutral_site", "is_postseason"]
    feature_list = diff_cols + count_features + ENG_FEATURES_BASE + ["elo_home_prob"]

    # Ensure identifier columns are always present
    id_cols = ['game_id', 'home_team', 'away_team', 'neutral_site']
    for col in id_cols:
        if col not in X.columns and col in base_df.columns:
            X[col] = base_df[col]

    return X, feature_list
