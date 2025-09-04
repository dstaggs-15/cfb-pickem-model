# lib/features.py

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List

from .rolling import long_stats_to_wide, build_sidewise_rollups, STAT_FEATURES
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

    Parameters
    ----------
    schedule : DataFrame
        Full schedule with at least: game_id, season, week, home_team, away_team,
        home_points, away_points, neutral_site, is_postseason (if available).
    team_stats : DataFrame
        Per-team per-game long/wide stats, must include: game_id, team, and stat columns.
    venues_df, teams_df, talent_df : DataFrame
        Contextual data used by rest_and_travel / pregame_probs.
    lines_df : DataFrame
        Historical market lines (used by median_lines()).
    manual_lines_df : DataFrame, optional
        For prediction mode, manual lines keyed by home_team/away_team (with 'spread' column).
    games_to_predict_df : DataFrame, optional
        If provided, we generate features for these games (prediction mode).
        Must include: game_id, home_team, away_team, season, (neutral_site if available).
    last_n : int
        Window size for rolling features.

    Returns
    -------
    X : DataFrame
        Feature matrix joined to the base (schedule or games_to_predict_df).
    feature_list : list[str]
        Names of core engineered features (does not include market features).
    """
    print("  Creating feature set...")

    # 1) Ensure (game_id, team) uniqueness, then side and go wide
    team_stats = team_stats.drop_duplicates(subset=['game_id', 'team'])

    # Map home team for each game, then mark home/away for each row
    home_team_map = schedule[['game_id', 'home_team']]
    team_stats_sided = team_stats.merge(
        home_team_map, on='game_id', how='left', validate='many_to_one'
    )
    team_stats_sided['home_away'] = np.where(
        team_stats_sided['team'] == team_stats_sided['home_team'], 'home', 'away'
    )
    team_stats_sided = team_stats_sided.drop(columns=['home_team'])

    # Wide (one row per game/team side with prefixed columns)
    wide_stats = long_stats_to_wide(team_stats_sided)

    # 2) Rolling features per side (home/away)
    home_roll, away_roll = build_sidewise_rollups(
        schedule=schedule,
        wide_stats=wide_stats,
        last_n=last_n,
        games_to_predict_df=games_to_predict_df,
    )

    # Normalize rollup keys to match merge keys
    if 'team' in home_roll.columns and 'home_team' not in home_roll.columns:
        home_roll = home_roll.rename(columns={'team': 'home_team'})
    if 'team' in away_roll.columns and 'away_team' not in away_roll.columns:
        away_roll = away_roll.rename(columns={'team': 'away_team'})

    # 3) Choose base DF: training (schedule) or prediction (games list)
    base_df = schedule if games_to_predict_df is None else games_to_predict_df

    # Sanity check required keys in base_df
    required = {'game_id', 'home_team', 'away_team'}
    missing = required - set(base_df.columns)
    if missing:
        raise ValueError(f"Base dataframe missing columns: {missing}")

    # Join rollups
    X = base_df.merge(
        home_roll, on=['game_id', 'home_team'], how='left', validate='one_to_one'
    )
    X = X.merge(
        away_roll, on=['game_id', 'away_team'], how='left', validate='one_to_one'
    )

    # 4) Diffs using last_n consistently
    diff_cols: List[str] = []
    for c in STAT_FEATURES:
        hc = f"home_R{last_n}_{c}"
        ac = f"away_R{last_n}_{c}"
        dc = f"diff_R{last_n}_{c}"
        if hc in X.columns and ac in X.columns:
            X[dc] = X[hc] - X[ac]
            diff_cols.append(dc)

    # 5) Context: rest/travel + ELO
    eng = rest_and_travel(schedule, teams_df, venues_df, games_to_predict_df)
    X = X.merge(eng, on="game_id", how="left", validate='one_to_one')

    elo_df = pregame_probs(schedule, talent_df, games_to_predict_df)
    X = X.merge(elo_df, on="game_id", how="left", validate='one_to_one')

    # 6) Market lines
    #    Prediction mode: prefer manual lines keyed by team names.
    #    Training mode: use median_lines (may be keyed by game_id or teams).
    if games_to_predict_df is not None and manual_lines_df is not None and not manual_lines_df.empty:
        mlines = manual_lines_df.rename(columns={'spread': 'spread_home'}).copy()
        need = {'home_team', 'away_team', 'spread_home'}
        miss = need - set(mlines.columns)
        if miss:
            raise ValueError(f"manual_lines_df missing columns: {miss}")
        X = X.merge(
            mlines[list(need)],
            on=['home_team', 'away_team'],
            how='left',
            validate='many_to_one'
        )
    else:
        med = median_lines(lines_df)
        if 'game_id' in med.columns:
            X = X.merge(med, on='game_id', how='left', validate='one_to_one')
        else:
            X = X.merge(
                med, on=['home_team', 'away_team'], how='left', validate='many_to_one'
            )

    # 7) Feature list (not including market features; those get appended downstream)
    count_features = [f"home_R{last_n}_count", f"away_R{last_n}_count"]
    ENG_FEATURES_BASE = ["rest_diff", "travel_away_km", "neutral_site", "is_postseason"]

    feature_list = diff_cols + count_features + ENG_FEATURES_BASE + ["elo_home_prob"]

    return X, feature_list
