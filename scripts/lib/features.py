# scripts/lib/features.py

from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd

from .parsing import ensure_schedule_columns
from . import elo as elo_lib
from . import market as market_lib
from . import context as context_lib
from . import rolling as rolling_lib

# ---------- File locations ----------
RAW_DIR = "data/raw/cfbd"
SCHED_CSV = f"{RAW_DIR}/cfb_schedule.csv"
STATS_CSV = f"{RAW_DIR}/cfb_game_team_stats.csv"
LINES_CSV = f"{RAW_DIR}/cfb_lines.csv"
TEAMS_CSV = f"{RAW_DIR}/cfbd_teams.csv"
VENUES_CSV = f"{RAW_DIR}/cfbd_venues.csv"
TALENT_CSV = f"{RAW_DIR}/cfbd_talent.csv"

# Manual lines (optional)
MANUAL_LINES_CSV = "docs/input/manual_lines.csv"


def _load_csv(path: str) -> pd.DataFrame:
    # low_memory=False avoids mixed-type chunk inference warnings
    return pd.read_csv(path, low_memory=False) if os.path.exists(path) else pd.DataFrame()


def _load_raw(
    schedule: pd.DataFrame | None,
    team_stats: pd.DataFrame | None,
    lines_df: pd.DataFrame | None,
    teams_df: pd.DataFrame | None,
    venues_df: pd.DataFrame | None,
    talent_df: pd.DataFrame | None,
    manual_lines_df: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sched = schedule.copy() if schedule is not None else _load_csv(SCHED_CSV)
    stats = team_stats.copy() if team_stats is not None else _load_csv(STATS_CSV)
    lines = lines_df.copy() if lines_df is not None else _load_csv(LINES_CSV)
    teams = teams_df.copy() if teams_df is not None else _load_csv(TEAMS_CSV)
    venues = venues_df.copy() if venues_df is not None else _load_csv(VENUES_CSV)
    talent = talent_df.copy() if talent_df is not None else _load_csv(TALENT_CSV)
    manual_lines = manual_lines_df.copy() if manual_lines_df is not None else _load_csv(MANUAL_LINES_CSV)
    return sched, stats, lines, teams, venues, talent, manual_lines


def _prep_schedule(df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_schedule_columns(df.copy())
    # Normalize team names (trim spaces) and make sure dates are usable
    for c in ["home_team", "away_team"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    # Coerce numerics where appropriate
    for c in ["season", "week", "home_points", "away_points", "venue_id"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Unify to string game_id for all downstream merges
    if "game_id" in df.columns:
        df["game_id"] = df["game_id"].astype(str)
    return df


def _prep_team_stats(schedule: pd.DataFrame, team_stats_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Build a team-game long table with advanced stats, then label rows as 'home'/'away'
    based on the schedule. Output will have columns: game_id, team, home_away, <stats...>
    """
    if team_stats_raw.empty:
        return pd.DataFrame(columns=["game_id", "team", "home_away"])

    # The CFBD /stats/game/advanced data in this repo is stored as long form with 'stat'/'value'
    cols = [c.lower() for c in team_stats_raw.columns]
    rename = dict(zip(team_stats_raw.columns, cols))
    ts = team_stats_raw.rename(columns=rename).copy()

    # Standard field names
    if "game_id" not in ts.columns and "gameid" in ts.columns:
        ts = ts.rename(columns={"gameid": "game_id"})
    if "team" not in ts.columns:
        if "school" in ts.columns:
            ts = ts.rename(columns={"school": "team"})
        else:
            ts["team"] = np.nan

    # Ensure string game_id for merge compatibility
    if "game_id" in ts.columns:
        ts["game_id"] = ts["game_id"].astype(str)

    # pivot: one row per (game_id, team), columns = stat names
    if "stat" in ts.columns and "value" in ts.columns:
        team_long = ts.pivot_table(index=["game_id", "team"], columns="stat", values="value").reset_index()
    else:
        keep = [c for c in ts.columns if c in ("game_id", "team")]
        team_long = ts[keep].copy()

    # Make numeric
    for c in team_long.columns:
        if c not in ("game_id", "team"):
            team_long[c] = pd.to_numeric(team_long[c], errors="coerce")

    # Attach home/away flag by joining schedule
    join = schedule[["game_id", "home_team", "away_team"]].drop_duplicates()
    # Merge (both sides have string game_id)
    out = team_long.merge(join, on="game_id", how="left")

    # SAFE assignment using object dtype (avoids NumPy dtype promotion with NaN)
    out["home_away"] = pd.Series(index=out.index, dtype="object")
    out.loc[out["team"].eq(out["home_team"]), "home_away"] = "home"
    out.loc[out["team"].eq(out["away_team"]), "home_away"] = "away"

    out = out.dropna(subset=["home_away"]).drop(columns=["home_team", "away_team"])
    return out


def _team_rollup_features(
    schedule: pd.DataFrame,
    team_stats_sided: pd.DataFrame,
    predict_df: pd.DataFrame | None,
    last_n: int = 5
) -> pd.DataFrame:
    """
    Use rolling.py helpers to compute rolling means of advanced stats with prior-season padding.
    Returns a game-level table with home/away-prefixed rolling features.
    """
    if team_stats_sided.empty:
        return pd.DataFrame()

    # Convert sided long stats to one row per game with _home/_away suffixes
    wide_stats = rolling_lib.long_stats_to_wide(team_stats_sided)

    # Build team-side rolling windows, with padding using previous-season averages
    home_roll, away_roll = rolling_lib.build_sidewise_rollups(
        schedule=schedule,
        wide_stats=wide_stats,
        last_n=last_n,
        predict_df=predict_df
    )
    # Join to schedule to label home/away rows correctly
    home_join = schedule[["game_id", "home_team"]].rename(columns={"home_team": "team"})
    away_join = schedule[["game_id", "away_team"]].rename(columns={"away_team": "team"})

    home_feats = home_roll.merge(home_join, on=["game_id", "team"], how="inner").drop(columns=["team"])
    away_feats = away_roll.merge(away_join, on=["game_id", "team"], how="inner").drop(columns=["team"])

    # Add prefixes to distinguish home vs away
    def _pref(df: pd.DataFrame, pfx: str) -> pd.DataFrame:
        df = df.copy()
        ren = {c: f"{pfx}{c}" for c in df.columns if c.startswith("R")}
        return df.rename(columns=ren)

    home_feats = _pref(home_feats, "home_")
    away_feats = _pref(away_feats, "away_")

    out = home_feats.merge(away_feats, on="game_id", how="outer")
    return out


def _market_features(lines_df: pd.DataFrame, manual_lines_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build market features (median spread & total). If manual lines exist, let them override.
    """
    lines = pd.concat([lines_df, manual_lines_df], ignore_index=True) if not manual_lines_df.empty else lines_df.copy()
    if lines.empty:
        return pd.DataFrame(columns=["game_id", "spread_home", "over_under"])
    # Normalize game_id to string before any downstream joins
    if "game_id" in lines.columns:
        lines["game_id"] = lines["game_id"].astype(str)
    med = market_lib.median_lines(lines)
    if "game_id" in med.columns:
        med["game_id"] = med["game_id"].astype(str)
    return med[["game_id", "spread_home", "over_under"]]


def _talent_features(schedule: pd.DataFrame, talent_df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple recruiting/talent differential: home - away.
    """
    if talent_df.empty or "school" not in talent_df.columns or "talent" not in talent_df.columns:
        return pd.DataFrame(columns=["game_id", "talent_diff"])

    t = talent_df.copy()
    t["school"] = t["school"].astype(str).str.strip()

    home = schedule[["game_id", "home_team"]].merge(
        t[["school", "talent"]].rename(columns={"school": "home_team", "talent": "home_talent"}),
        on="home_team", how="left"
    )
    away = schedule[["game_id", "away_team"]].merge(
        t[["school", "talent"]].rename(columns={"school": "away_team", "talent": "away_talent"}),
        on="away_team", how="left"
    )

    out = home.merge(away, on="game_id", how="left")
    out["talent_diff"] = pd.to_numeric(out["home_talent"], errors="coerce") - pd.to_numeric(out["away_talent"], errors="coerce")
    return out[["game_id", "talent_diff"]]


def create_feature_set(
    schedule: pd.DataFrame | None = None,
    team_stats: pd.DataFrame | None = None,
    venues_df: pd.DataFrame | None = None,
    teams_df: pd.DataFrame | None = None,
    talent_df: pd.DataFrame | None = None,
    lines_df: pd.DataFrame | None = None,
    manual_lines_df: pd.DataFrame | None = None,
    games_to_predict_df: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """
    Build the feature matrix and list of feature names for both training and prediction.

    Returns:
        (X, feature_cols):
          X: DataFrame with id columns (game_id, season, week, home_team, away_team,
             neutral_site, home_points, away_points) and engineered numeric features.
          feature_cols: list of the feature column names used for modeling.
    """
    # Load or use provided raw tables
    schedule, stats_raw, lines_raw, teams_raw, venues_raw, talent_raw, manual_lines_raw = _load_raw(
        schedule, team_stats, lines_df, teams_df, venues_df, talent_df, manual_lines_df
    )

    schedule = _prep_schedule(schedule)

    # If we have predict games, append them into the working schedule (ids must be unique)
    if games_to_predict_df is not None and not games_to_predict_df.empty:
        pred = games_to_predict_df.copy()
        for c in ["season", "week", "neutral_site", "home_points", "away_points"]:
            if c not in pred.columns:
                pred[c] = 0
        pred["date"] = pd.to_datetime(pred.get("date"), errors="coerce", utc=True)
        # Ensure string ids for predicted games too
        if "game_id" in pred.columns:
            pred["game_id"] = pred["game_id"].astype(str)
        schedule = pd.concat([schedule, pred], ignore_index=True, sort=False)

    # --------- Build per-module features ---------
    # 1) Rolling advanced team stats (last 5), with previous-season padding for early weeks
    sided = _prep_team_stats(schedule, stats_raw)
    roll_feats = _team_rollup_features(schedule, sided, games_to_predict_df, last_n=5)

    # 2) Market (median spread/OU)
    market_feats = _market_features(lines_raw, manual_lines_raw)

    # 3) Elo pre-game win prob (handles prior-season vs current-season logic)
    elo_probs = elo_lib.pregame_probs(schedule=schedule, talent_df=talent_raw, predict_df=games_to_predict_df)

    # 4) Context: rest, travel, bye, neutral/postseason
    ctx = context_lib.rest_and_travel(schedule=schedule, teams_df=teams_raw, venues_df=venues_raw, predict_df=games_to_predict_df)

    # 5) Talent differential
    talent_feats = _talent_features(schedule, talent_raw)

    # --------- Assemble master table ---------
    id_cols = [
        "game_id", "season", "week",
        "home_team", "away_team", "neutral_site",
        "home_points", "away_points"
    ]
    base = schedule[id_cols].drop_duplicates("game_id")
    base["game_id"] = base["game_id"].astype(str)

    X = base
    for part in (roll_feats, market_feats, elo_probs, ctx, talent_feats):
        if not part.empty:
            if "game_id" in part.columns:
                part = part.copy()
                part["game_id"] = part["game_id"].astype(str)
            X = X.merge(part, on="game_id", how="left")

    # Clean and select feature columns
    feature_cols = [c for c in X.columns if c not in id_cols]
    for c in feature_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # Per-season mean imputation then global mean
    for c in feature_cols:
        X[c] = X.groupby("season")[c].transform(lambda s: s.fillna(s.mean()))
        X[c] = X[c].fillna(X[c].mean())

    kept = [c for c in feature_cols if X[c].notna().any()]
    return X[id_cols + kept].copy(), kept
