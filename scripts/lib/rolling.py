# scripts/lib/rolling.py

import os
import datetime as dt
import numpy as np
import pandas as pd

DERIVED = "data/derived"
SEASON_AVG_PARQUET = f"{DERIVED}/season_averages.parquet"


def long_stats_to_wide(team_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Convert per-team (home/away) long stats into one wide row per game.

    Requires columns:
      - game_id
      - home_away ('home'/'away' or equivalents)
      - team
      - numeric stat columns to aggregate

    If there are no numeric stats, a clear error is raised (so you know
    upstream melt/merge is wrong).
    """
    if team_stats is None or team_stats.empty:
        return pd.DataFrame(columns=["game_id"])

    df = team_stats.copy()

    # Required columns
    required = {"game_id", "home_away"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"long_stats_to_wide(): missing required columns: {sorted(missing)}. "
            f"Have: {list(df.columns)}"
        )

    # Normalize home_away
    df["home_away"] = df["home_away"].astype(str).str.lower().str.strip()
    df["home_away"] = df["home_away"].replace(
        {"h": "home", "home_team": "home", "host": "home",
         "a": "away", "away_team": "away", "visitor": "away"}
    )
    df = df[df["home_away"].isin(["home", "away"])]

    # Coerce object columns to numeric where possible (exclude meta/id fields)
    id_like = {"game_id", "home_away", "team", "season", "week", "date"}
    for c in df.columns:
        if c in id_like:
            continue
        if df[c].dtype == "object":
            tmp = pd.to_numeric(df[c], errors="coerce")
            if tmp.notna().any():
                df[c] = tmp

    # Numeric stat columns (never average ids)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in {"game_id", "season", "week"}]
    if not numeric_cols:
        sample = df.head(25)[list(set(df.columns) - {"date"})].to_dict(orient="records")
        raise ValueError(
            "long_stats_to_wide(): no numeric stat columns to aggregate. "
            "Upstream melt/merge likely included text columns as stats or no stats were loaded. "
            f"Sample rows (trimmed): {sample}"
        )

    # Group and unstack (robustly preserves game_id)
    grp = df.groupby(["game_id", "home_away"], dropna=False)[numeric_cols].mean()
    wide = grp.unstack("home_away")
    if isinstance(wide.columns, pd.MultiIndex):
        wide.columns = [f"{stat}_{side}" for stat, side in wide.columns.to_list()]
    else:
        wide.columns = [f"{c}_home" for c in wide.columns]

    wide = wide.reset_index()  # ensures 'game_id' is a column

    if "game_id" not in wide.columns:
        raise ValueError("long_stats_to_wide(): 'game_id' still missing after unstack/reset_index()")

    return wide


def _get_rollups(df: pd.DataFrame, last_n: int, season_averages_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-team rolling means with previous-season padding.
    Expects: ['game_id','date','season','team', <numeric stat columns>]
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.sort_values(by=["date"]).reset_index(drop=True)

    stat_cols = [
        c for c in df.columns
        if c not in {"game_id", "date", "season", "team"} and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not stat_cols:
        return pd.DataFrame(columns=["game_id", "team"])

    out_all = []
    for team_name, team_df in df.groupby("team"):
        seasons = team_df["season"].dropna().unique()
        for season in seasons:
            s_df = team_df[team_df["season"] == season]
            prior = season - 1
            pad_rows = []
            if not season_averages_df.empty:
                prior_row = season_averages_df[
                    (season_averages_df["team"] == team_name) &
                    (season_averages_df["season"] == prior)
                ]
                if not prior_row.empty:
                    base = prior_row[stat_cols].iloc[0]
                    pad_rows = [base] * last_n

            if pad_rows:
                padded = pd.concat([pd.DataFrame(pad_rows), s_df[stat_cols]], ignore_index=True)
            else:
                padded = s_df[stat_cols]

            rolling = padded.rolling(window=last_n, min_periods=1).mean()
            final = rolling.iloc[-len(s_df):].shift(1)
            final.columns = [f"R{last_n}_{c}" for c in stat_cols]

            counts = pd.Series(range(1, len(s_df) + 1), index=s_df.index)
            counts[counts > last_n] = last_n
            final[f"R{last_n}_count"] = counts

            season_roll = pd.concat(
                [s_df[["game_id"]].reset_index(drop=True), final.reset_index(drop=True)],
                axis=1
            )
            season_roll["team"] = team_name
            out_all.append(season_roll)

    if not out_all:
        return pd.DataFrame()

    return pd.concat(out_all, ignore_index=True)


def build_sidewise_rollups(
    schedule: pd.DataFrame,
    wide_stats: pd.DataFrame,
    last_n: int,
    predict_df: pd.DataFrame | None = None
):
    """
    Create separate home/away rolling frames aligned by game_id
    from a joined schedule+wide_stats table.

    Now tolerant of empty/malformed schedule: returns empty rollups instead
    of crashing.
    """
    # If schedule is missing or lacks required keys, bail gracefully
    if schedule is None or schedule.empty or "game_id" not in schedule.columns:
        return pd.DataFrame(), pd.DataFrame()

    season_averages_df = (
        pd.read_parquet(SEASON_AVG_PARQUET) if os.path.exists(SEASON_AVG_PARQUET) else pd.DataFrame()
    )

    schedule = schedule.copy()
    if "date" in schedule.columns:
        schedule["date"] = pd.to_datetime(schedule["date"], errors="coerce")
        try:
            # localize if naive; convert if tz-aware
            if getattr(schedule["date"].dt, "tz", None) is None:
                schedule["date"] = schedule["date"].dt.tz_localize("UTC")
            else:
                schedule["date"] = schedule["date"].dt.tz_convert("UTC")
        except Exception:
            # Worst case: keep as naive; rolling still works
            pass

    # Ensure 'game_id' is a column on wide_stats
    if "game_id" not in wide_stats.columns:
        wide_stats = wide_stats.reset_index(drop=False)

    # Merge schedule + wide_stats
    full_df = schedule.merge(wide_stats, on="game_id", how="left")

    # Per-team frames
    home_df = full_df[["game_id", "date", "season", "home_team"]].rename(columns={"home_team": "team"})
    home_stats_cols = [c for c in full_df.columns if c.endswith("_home")]
    home_stats = full_df[home_stats_cols].copy()
    home_stats.columns = [c.replace("_home", "") for c in home_stats.columns]
    home_df = pd.concat([home_df, home_stats], axis=1)

    away_df = full_df[["game_id", "date", "season", "away_team"]].rename(columns={"away_team": "team"})
    away_stats_cols = [c for c in full_df.columns if c.endswith("_away")]
    away_stats = full_df[away_stats_cols].copy()
    away_stats.columns = [c.replace("_away", "") for c in away_stats.columns]
    away_df = pd.concat([away_df, away_stats], axis=1)

    # Predict-mode support
    if predict_df is not None and not predict_df.empty:
        pred = predict_df.copy()
        pred["date"] = pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=365)
        if "season" not in pred.columns and "season" in schedule.columns:
            pred["season"] = schedule["season"].max()

        home_pred = pred[["game_id", "date", "season", "home_team"]].rename(columns={"home_team": "team"})
        away_pred = pred[["game_id", "date", "season", "away_team"]].rename(columns={"away_team": "team"})

        home_df = pd.concat([home_df, home_pred], ignore_index=True)
        away_df = pd.concat([away_df, away_pred], ignore_index=True)

    home_rollups = _get_rollups(home_df.dropna(subset=["team"]), last_n, season_averages_df)
    away_rollups = _get_rollups(away_df.dropna(subset=["team"]), last_n, season_averages_df)

    if predict_df is not None and not predict_df.empty:
        to_pred = predict_df["game_id"].unique()
        home_rollups = home_rollups[home_rollups["game_id"].isin(to_pred)]
        away_rollups = away_rollups[away_rollups["game_id"].isin(to_pred)]

    return home_rollups, away_rollups
