# scripts/lib/rolling.py

import os
import datetime as dt
import numpy as np
import pandas as pd

DERIVED = "data/derived"
SEASON_AVG_PARQUET = f"{DERIVED}/season_averages.parquet"


def long_stats_to_wide(team_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot per-team (home/away) long stats into a single wide row per game.

    Expects team_stats to contain:
      - 'game_id' (identifier)
      - 'home_away' in {'home','away'}
      - 'team'
      - one or more numeric stat columns to aggregate

    This function:
      - coerces convertible object columns to numeric
      - selects numeric-only columns for aggregation
      - pivots to home/away columns and RETURNS 'game_id' AS A COLUMN
    """
    if team_stats is None or team_stats.empty:
        # Return a well-shaped empty frame with game_id so downstream logic stays sane
        return pd.DataFrame(columns=["game_id"])

    df = team_stats.copy()

    # Basic schema sanity
    required = {"game_id", "home_away"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"long_stats_to_wide(): missing required columns: {sorted(missing)}. "
            f"Available columns: {list(df.columns)}"
        )

    # Try to coerce object columns (besides ids) to numeric where possible
    id_like = {"game_id", "home_away", "team", "season", "week", "date"}
    for c in df.columns:
        if c not in id_like and df[c].dtype == "object":
            tmp = pd.to_numeric(df[c], errors="coerce")
            # Only replace if any conversions succeed; otherwise leave the column as-is
            if tmp.notna().any():
                df[c] = tmp

    # Use strictly numeric columns for agg; never average ids/meta
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in {"game_id", "season", "week"}]

    if not numeric_cols:
        # Fail FAST with a helpful message instead of letting a later merge blow up
        sample = df.head(20).to_dict(orient="records")
        raise ValueError(
            "long_stats_to_wide(): no numeric stat columns found to pivot. "
            "Upstream melt/merge likely leaked strings into the stat set. "
            f"Sample rows: {sample}"
        )

    # Robust pivot that ignores non-existent category combos
    pivoted = (
        df.pivot_table(
            index="game_id",
            columns="home_away",
            values=numeric_cols,
            aggfunc="mean",
            observed=True,
        )
        .sort_index()
        .reset_index()  # <- CRITICAL: ensure 'game_id' is a COLUMN, not index
    )

    # Flatten MultiIndex columns -> "<stat>_<home|away>"
    pivoted.columns = [
        f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col
        for col in pivoted.columns
    ]

    # At this point, we MUST have 'game_id' as a column
    if "game_id" not in pivoted.columns:
        # As an absolute last resort, pull it from an index if somehow present
        pivoted = pivoted.reset_index()
        if "game_id" not in pivoted.columns:
            raise ValueError(
                "long_stats_to_wide(): internal error — 'game_id' missing after pivot."
            )

    return pivoted


def _get_rollups(df: pd.DataFrame, last_n: int, season_averages_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-team rolling means with previous-season padding.
    Expects df to have: ['game_id', 'date', 'season', 'team', <numeric stat columns>]
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.sort_values(by=["date"]).reset_index(drop=True)

    # Only roll over numeric stats
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
            # previous-season padding (carry some baseline into early weeks)
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

            # Rolling mean over last_n, shifted so we don't peek at the current game
            rolling = padded.rolling(window=last_n, min_periods=1).mean()
            final = rolling.iloc[-len(s_df):].shift(1)
            final.columns = [f"R{last_n}_{c}" for c in stat_cols]

            # How many real games we had backing the rolling window (1..last_n)
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
    """
    season_averages_df = (
        pd.read_parquet(SEASON_AVG_PARQUET) if os.path.exists(SEASON_AVG_PARQUET) else pd.DataFrame()
    )

    schedule = schedule.copy()
    if "date" in schedule.columns:
        # Keep times consistent in UTC; if your input is tz-aware already, use tz_convert
        schedule["date"] = pd.to_datetime(schedule["date"], errors="coerce")
        if schedule["date"].dt.tz is None:
            schedule["date"] = schedule["date"].dt.tz_localize("UTC")
        else:
            schedule["date"] = schedule["date"].dt.tz_convert("UTC")

    # --- HARD GUARANTEE: 'game_id' must be a column on wide_stats
    if "game_id" not in wide_stats.columns:
        wide_stats = wide_stats.reset_index()
    if "game_id" not in wide_stats.columns:
        # If still not present, then wide_stats is malformed/empty. Fail FAST with context.
        raise ValueError(
            f"build_sidewise_rollups(): 'game_id' missing on wide_stats. "
            f"wide_stats columns={list(wide_stats.columns)} shape={wide_stats.shape}"
        )

    # If wide_stats is somehow empty, create a shell so merges don’t KeyError
    if wide_stats.empty:
        wide_stats = pd.DataFrame({"game_id": schedule["game_id"].unique()})

    full_df = schedule.merge(wide_stats, on="game_id", how="left")

    # --- Build per-team (home) frame
    home_df = full_df[["game_id", "date", "season", "home_team"]].rename(columns={"home_team": "team"})
    home_stats_cols = [c for c in full_df.columns if c.endswith("_home")]
    home_stats = full_df[home_stats_cols].copy()
    home_stats.columns = [c.replace("_home", "") for c in home_stats.columns]
    home_df = pd.concat([home_df, home_stats], axis=1)

    # --- Build per-team (away) frame
    away_df = full_df[["game_id", "date", "season", "away_team"]].rename(columns={"away_team": "team"})
    away_stats_cols = [c for c in full_df.columns if c.endswith("_away")]
    away_stats = full_df[away_stats_cols].copy()
    away_stats.columns = [c.replace("_away", "") for c in away_stats.columns]
    away_df = pd.concat([away_df, away_stats], axis=1)

    # If predicting, append the games to the per-team frames with future dates
    if predict_df is not None and not predict_df.empty:
        pred = predict_df.copy()
        pred["date"] = pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=365)
        if "season" not in pred.columns:
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
