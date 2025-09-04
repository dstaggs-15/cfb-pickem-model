import pandas as pd
import numpy as np
from .rolling import long_stats_to_wide, build_sidewise_rollups, STAT_FEATURES
from .context import rest_and_travel
from .market import median_lines
from .elo import pregame_probs

def create_feature_set(schedule, team_stats, venues_df, teams_df, talent_df, lines_df, manual_lines_df, games_to_predict_df=None):
    print("  Creating feature set...")

    LAST_N = 5

    # 1) Base stats -> sided -> wide
    team_stats = team_stats.drop_duplicates(subset=['game_id', 'team'])
    home_team_map = schedule[['game_id', 'home_team']]
    team_stats_sided = team_stats.merge(home_team_map, on='game_id', how='left', validate='many_to_one')
    team_stats_sided['home_away'] = np.where(team_stats_sided['team'] == team_stats_sided['home_team'], 'home', 'away')
    team_stats_sided = team_stats_sided.drop(columns=['home_team'])
    wide_stats = long_stats_to_wide(team_stats_sided)

    # 2) Rollups
    home_roll, away_roll = build_sidewise_rollups(schedule, wide_stats, LAST_N, games_to_predict_df)

    # Normalize rollup keys
    if 'team' in home_roll.columns and 'home_team' not in home_roll.columns:
        home_roll = home_roll.rename(columns={'team':'home_team'})
    if 'team' in away_roll.columns and 'away_team' not in away_roll.columns:
        away_roll = away_roll.rename(columns={'team':'away_team'})

    # 3) Base df + join rollups
    base_df = schedule if games_to_predict_df is None else games_to_predict_df

    req = {'game_id','home_team','away_team'}
    miss = req - set(base_df.columns)
    if miss:
        raise ValueError(f"Base dataframe missing columns: {miss}")

    X = base_df.merge(home_roll, on=['game_id','home_team'], how='left', validate='one_to_one')
    X = X.merge(away_roll, on=['game_id','away_team'], how='left', validate='one_to_one')

    # 4) Diffs (use LAST_N consistently)
    diff_cols = []
    for c in STAT_FEATURES:
        hc = f"home_R{LAST_N}_{c}"
        ac = f"away_R{LAST_N}_{c}"
        dc = f"diff_R{LAST_N}_{c}"
        if hc in X.columns and ac in X.columns:
            X[dc] = X[hc] - X[ac]
            diff_cols.append(dc)

    # 5) Engineering + ELO
    eng = rest_and_travel(schedule, teams_df, venues_df, games_to_predict_df)
    X = X.merge(eng, on="game_id", how="left", validate='one_to_one')

    elo_df = pregame_probs(schedule, talent_df, games_to_predict_df)
    X = X.merge(elo_df, on="game_id", how="left", validate='one_to_one')

    # 6) Market lines (no in-place rename; robust keys)
    if games_to_predict_df is not None and manual_lines_df is not None and not manual_lines_df.empty:
        mlines = manual_lines_df.rename(columns={'spread': 'spread_home'}).copy()
        need = {'home_team','away_team','spread_home'}
        miss = need - set(mlines.columns)
        if miss:
            raise ValueError(f"manual_lines_df missing columns: {miss}")
        X = X.merge(mlines[list(need)], on=['home_team','away_team'], how='left', validate='many_to_one')
    else:
        med = median_lines(lines_df)
        if 'game_id' in med.columns:
            X = X.merge(med, on='game_id', how='left', validate='one_to_one')
        else:
            X = X.merge(med, on=['home_team','away_team'], how='left', validate='many_to_one')

    # 7) Feature list (guarantee existence)
    count_features = [f"home_R{LAST_N}_count", f"away_R{LAST_N}_count"]
    ENG_FEATURES_BASE = ["rest_diff", "travel_away_km", "neutral_site", "is_postseason"]

    for col, default in [
        ('rest_diff', 0.0), ('travel_away_km', 0.0),
        ('neutral_site', 0), ('is_postseason', 0),
        ('elo_home_prob', np.nan),
    ]:
        if col not in X.columns:
            X[col] = default

    feature_list = diff_cols + count_features + ENG_FEATURES_BASE + ["elo_home_prob"]
    return X, feature_list
