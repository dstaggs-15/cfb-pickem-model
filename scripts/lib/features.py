# scripts/lib/features.py
# Creates usable numeric features so training and prediction don't choke.

from __future__ import annotations

import os
import pandas as pd
import numpy as np

RAW_DIR = "data/raw/cfbd"
SCHED_CSV = os.path.join(RAW_DIR, "cfb_schedule.csv")
LINES_CSV = os.path.join(RAW_DIR, "cfb_lines.csv")

def _load_schedule() -> pd.DataFrame:
    if not os.path.exists(SCHED_CSV):
        raise FileNotFoundError(
            f"[FEATURES] Missing required schedule CSV at {SCHED_CSV}. "
            "Run the fetch workflow first."
        )
    df = pd.read_csv(SCHED_CSV)
    # Ensure all expected columns exist
    need = [
        "game_id", "season", "week",
        "home_team", "away_team", "date",
        "home_points", "away_points",
        "neutral_site"
    ]
    for col in need:
        if col not in df.columns:
            df[col] = pd.NA
    # Coerce data types
    df["game_id"] = pd.to_numeric(df["game_id"], errors="coerce")
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df["week"] = pd.to_numeric(df["week"], errors="coerce")
    df["home_points"] = pd.to_numeric(df["home_points"], errors="coerce")
    df["away_points"] = pd.to_numeric(df["away_points"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df["neutral_site"] = df["neutral_site"].fillna(0)

    # Sort deterministically and drop rows without IDs
    df = df.sort_values(
        ["season", "week", "date", "game_id"],
        na_position="last"
    )
    df = df[df["game_id"].notna()].copy()
    df["game_id"] = df["game_id"].astype("int64")
    return df

def _melt_schedule_to_team_games(sched: pd.DataFrame) -> pd.DataFrame:
    """Create a per-team/per-game view to compute rolling features."""
    home = sched.rename(columns={
        "home_team": "team",
        "away_team": "opponent",
        "home_points": "points_for",
        "away_points": "points_against",
    }).assign(is_home=1)

    away = sched.rename(columns={
        "away_team": "team",
        "home_team": "opponent",
        "away_points": "points_for",
        "home_points": "points_against",
    }).assign(is_home=0)

    cols = [
        "game_id", "season", "week", "date",
        "team", "opponent", "is_home",
        "points_for", "points_against"
    ]
    long = pd.concat([home[cols], away[cols]], ignore_index=True)
    long["margin"] = pd.to_numeric(long["points_for"], errors="coerce") - pd.to_numeric(long["points_against"], errors="coerce")
    long["win"] = (long["margin"] > 0).astype("float")
    return long

def _add_rolling_form(long: pd.DataFrame) -> pd.DataFrame:
    """Calculate rolling averages of key stats (L3, L5, win rate)."""
    long = long.sort_values(["team", "date", "game_id"])
    grp = long.groupby("team", group_keys=False)

    def _roll(s: pd.Series, window: int) -> pd.Series:
        return s.shift(1).rolling(window, min_periods=1).mean()

    long["pf_l3"] = grp["points_for"].apply(lambda s: _roll(s, 3))
    long["pa_l3"] = grp["points_against"].apply(lambda s: _roll(s, 3))
    long["margin_l3"] = grp["margin"].apply(lambda s: _roll(s, 3))

    long["pf_l5"] = grp["points_for"].apply(lambda s: _roll(s, 5))
    long["pa_l5"] = grp["points_against"].apply(lambda s: _roll(s, 5))
    long["margin_l5"] = grp["margin"].apply(lambda s: _roll(s, 5))

    long["winrate_l5"] = grp["win"].apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean())

    # Home/away-specific rolling margin
    long["home_margin_l3"] = grp.apply(
        lambda g: g.assign(_hm=g["margin"].where(g["is_home"] == 1).shift(1).rolling(3, min_periods=1).mean())
    )["_hm"].values
    long["away_margin_l3"] = grp.apply(
        lambda g: g.assign(_am=g["margin"].where(g["is_home"] == 0).shift(1).rolling(3, min_periods=1).mean())
    )["_am"].values

    return long

def _pivot_back_to_game(long_with_rolls: pd.DataFrame) -> pd.DataFrame:
    """Pivot the per-team rolling metrics back to per-game rows."""
    home_side = long_with_rolls[long_with_rolls["is_home"] == 1].copy()
    away_side = long_with_rolls[long_with_rolls["is_home"] == 0].copy()

    keep_feats = [
        "pf_l3", "pa_l3", "margin_l3",
        "pf_l5", "pa_l5", "margin_l5",
        "winrate_l5",
        "home_margin_l3", "away_margin_l3",
    ]
    home_feats = home_side[["game_id", "team"] + keep_feats].add_prefix("home_")
    home_feats = home_feats.rename(columns={
        "home_game_id": "game_id",
        "home_team": "home_team_name"
    })

    away_feats = away_side[["game_id", "team"] + keep_feats].add_prefix("away_")
    away_feats = away_feats.rename(columns={
        "away_game_id": "game_id",
        "away_team": "away_team_name"
    })

    roll = pd.merge(home_feats, away_feats, on="game_id", how="inner")
    return roll

def _load_lines_features(schedule: pd.DataFrame) -> pd.DataFrame:
    """Parse cfb_lines.csv to extract closing spread/total."""
    if not os.path.exists(LINES_CSV):
        return pd.DataFrame()
    df = pd.read_csv(LINES_CSV)
    if df.empty:
        return pd.DataFrame()
    if "game_id" not in df.columns:
        if "id" in df.columns:
            df = df.rename(columns={"id": "game_id"})
        else:
            return pd.DataFrame()
    df["game_id"] = pd.to_numeric(df["game_id"], errors="coerce")
    df = df[df["game_id"].notna()].copy()
    df["game_id"] = df["game_id"].astype("int64")

    spread_cols = [c for c in df.columns if c.lower() in {"spread", "spread_close", "closing_spread", "close_spread"}]
    total_cols = [c for c in df.columns if c.lower() in {"overunder", "total", "closing_total", "total_close"}]

    # Keep the last record per game (closer to kickoff)
    df = df.sort_values(["game_id"]).groupby("game_id", as_index=False).tail(1)

    out = df[["game_id"]].copy()
    if spread_cols:
        out["home_closing_spread"] = pd.to_numeric(df[spread_cols[0]], errors="coerce")
    if total_cols:
        out["home_total"] = pd.to_numeric(df[total_cols[0]], errors="coerce")

    return out.drop_duplicates("game_id")

def create_feature_set(team_stats_sided: pd.DataFrame | None = None) -> tuple[pd.DataFrame, list[str]]:
    """
    Build the feature matrix from raw schedule + lines.

    Returns:
        X: DataFrame with id columns (game_id, season, week, home_team, away_team, neutral_site, home_points, away_points)
           and engineered numeric features.
        feature_list: list of predictor column names.
    """
    schedule = _load_schedule()
    print(f"[FEATURES] schedule (raw normalized): shape={schedule.shape} cols={list(schedule.columns)}")
    print(schedule.head().to_string(index=False))

    # Rolling form features
    team_long = _melt_schedule_to_team_games(schedule)
    team_long = _add_rolling_form(team_long)
    roll = _pivot_back_to_game(team_long)

    # Merge identifiers (include home/away teams and neutral site)
    id_cols = [
        "game_id", "season", "week",
        "home_team", "away_team", "neutral_site",
        "home_points", "away_points"
    ]
    base = schedule[id_cols].drop_duplicates("game_id")

    X = base.merge(roll, on="game_id", how="left")

    # Merge lines if present
    lines_feats = _load_lines_features(schedule)
    if not lines_feats.empty:
        X = X.merge(lines_feats, on="game_id", how="left")

    # Normalize/clean numeric feature columns
    feature_cols = [c for c in X.columns if c not in id_cols]
    for c in feature_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # Simple per-season mean imputation followed by global mean
    for c in feature_cols:
        X[c] = X.groupby("season")[c].transform(lambda s: s.fillna(s.mean()))
        X[c] = X[c].fillna(X[c].mean())

    # Drop all-NaN features
    kept_feats = [c for c in feature_cols if X[c].notna().any()]
    print(f"[FEATURES] engineered features: count={len(kept_feats)} sample={kept_feats[:10]}")
    return X[id_cols + kept_feats].copy(), kept_feats
