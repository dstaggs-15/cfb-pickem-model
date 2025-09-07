# scripts/lib/features.py

import pandas as pd
import numpy as np

from .rolling import long_stats_to_wide, build_sidewise_rollups
# import your other helpers as you already do:
# from .parsing import ensure_schedule_columns
# from .market import median_lines, fit_market_mapping
# from .elo import pregame_probs
# from .context import rest_and_travel
# etc.

# Put the numeric stat names you actually expect to melt/pivot here.
# Keep this list in sync with whatever your raw schema provides.
STAT_FEATURES = [
    # examples — replace/extend with your true numeric metrics
    "ppa",
    "success_rate",
    "explosiveness",
    "rushing_ppa",
    "passing_ppa",
    "defense_ppa",
    "points_per_play",
    "yards_per_play",
    # add the rest of your numeric stat fields
]

ID_VARS = ["season", "week", "game_id", "team", "home_away", "date"]

def _is_numeric_series(s: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(s):
        return True
    coerced = pd.to_numeric(s, errors="coerce")
    return coerced.notna().any()

def _log_small(name, df, rows=5):
    try:
        print(f"[FEATURES] {name}: shape={df.shape} cols={list(df.columns)[:20]}")
        if not df.empty:
            print(df.head(rows).to_string(index=False))
    except Exception as e:
        print(f"[FEATURES] {name}: <failed to log: {e}>")

def _melt_numeric_only(df: pd.DataFrame) -> pd.DataFrame:
    """
    Melt only the numeric stat columns we expect. Never melt team names, etc.
    """
    # Only keep columns we know are numeric features and required ids
    value_vars = [c for c in df.columns if c in STAT_FEATURES]
    missing_stats = [c for c in STAT_FEATURES if c not in df.columns]
    if missing_stats:
        print(f"[FEATURES] WARNING: Missing expected stat columns: {missing_stats[:20]}{' ...' if len(missing_stats)>20 else ''}")

    long_df = df.melt(
        id_vars=[c for c in ID_VARS if c in df.columns],
        value_vars=value_vars,
        var_name="stat",
        value_name="value",
    )

    # Coerce values to numeric and drop non-numeric rows
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    bad = long_df["value"].isna()
    if bad.any():
        # show a quick sample so you can locate the leak
        sample = long_df.loc[bad, ["game_id", "team", "home_away", "stat", "value"]].head(25)
        print("[FEATURES] Dropping non-numeric long rows (sample):")
        print(sample.to_string(index=False))
        long_df = long_df.loc[~bad].copy()

    return long_df

def create_feature_set(use_cache: bool = True, predict_only: bool = False):
    """
    Your existing entry point. Read your raw/cached inputs here,
    build long stats safely, pivot, build sidewise rollups, etc.
    """
    # ----------------------------------------------------------------------
    # 1) Load your preprocessed team-level wide stats table here
    #    (Adjust these loads to your actual source; the point is to
    #     show/validate the frame we’re about to melt.)
    # ----------------------------------------------------------------------
    # Example skeleton — replace with your actual upstream loads:
    # team_wide = pd.read_parquet("data/derived/team_wide.parquet")
    # schedule  = pd.read_parquet("data/derived/schedule.parquet")

    # The following are placeholders — replace with your real source
    try:
        team_wide = pd.read_parquet("data/derived/team_wide.parquet")
    except Exception:
        # Fallback so code runs; replace with your true source integration
        team_wide = pd.DataFrame()

    try:
        schedule = pd.read_parquet("data/derived/schedule.parquet")
    except Exception:
        schedule = pd.DataFrame()

    _log_small("team_wide (pre-melt)", team_wide)
    _log_small("schedule (pre-rollups)", schedule)

    # ----------------------------------------------------------------------
    # 2) Create long stats safely (numeric-only melt)
    # ----------------------------------------------------------------------
    if team_wide.empty:
        print("[FEATURES] WARNING: team_wide is empty — downstream frames will be empty.")
        team_stats_long = pd.DataFrame(columns=["season","week","game_id","team","home_away","date","stat","value"])
    else:
        # Ensure required id columns exist; adapt to your schema as needed.
        for col in ["season","week","game_id","team","home_away","date"]:
            if col not in team_wide.columns:
                team_wide[col] = pd.NA

        team_stats_long = _melt_numeric_only(team_wide)

    _log_small("team_stats_long (post-melt)", team_stats_long)

    # ----------------------------------------------------------------------
    # 3) Convert long → wide per game: home/away columns
    # ----------------------------------------------------------------------
    team_stats_sided = team_stats_long.pivot_table(
        index=["season","week","game_id","team","home_away","date"],
        columns="stat",
        values="value",
        aggfunc="mean",
        observed=True,
    ).reset_index()

    _log_small("team_stats_sided (pre-wide)", team_stats_sided)

    # Now call the robust wide pivot (ensures game_id ends up as a column)
    wide_stats = long_stats_to_wide(team_stats_sided)
    _log_small("wide_stats (post-wide)", wide_stats)

    # ----------------------------------------------------------------------
    # 4) Sidewise rollups (last_n could be your configured windows, e.g., 5)
    # ----------------------------------------------------------------------
    LAST_N = 5
    home_roll, away_roll = build_sidewise_rollups(
        schedule=schedule,
        wide_stats=wide_stats,
        last_n=LAST_N,
        predict_df=None  # or pass your predict-only frame if you're in predict mode
    )
    _log_small("home_roll", home_roll)
    _log_small("away_roll", away_roll)

    # ----------------------------------------------------------------------
    # 5) Build your final feature matrix X and feature_list
    #     (This is just a scaffold; plug in your actual logic here.)
    # ----------------------------------------------------------------------
    # Example: join home/away rollups and any other signals you use
    if home_roll.empty and away_roll.empty:
        X = pd.DataFrame()
        feature_list = []
    else:
        # naive example: align on game_id and suffix columns
        X = home_roll.merge(away_roll, on=["game_id","team"], how="outer", suffixes=("_home","_away"))
        feature_cols = [c for c in X.columns if c not in {"game_id","team"}]
        feature_list = feature_cols

    return X, feature_list
