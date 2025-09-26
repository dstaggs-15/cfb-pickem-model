# scripts/build_dataset.py

from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd

from .lib.features import create_feature_set
from .lib.rolling import STAT_FEATURES, SEASON_AVG_PARQUET
from .lib.parsing import ensure_schedule_columns

DERIVED = "data/derived"
RAW_DIR = "data/raw/cfbd"

SCHED_CSV = f"{RAW_DIR}/cfb_schedule.csv"
STATS_CSV = f"{RAW_DIR}/cfb_game_team_stats.csv"
LINES_CSV = f"{RAW_DIR}/cfb_lines.csv"
TEAMS_CSV = f"{RAW_DIR}/cfbd_teams.csv"
VENUES_CSV = f"{RAW_DIR}/cfbd_venues.csv"
TALENT_CSV = f"{RAW_DIR}/cfbd_talent.csv"

TRAIN_PARQUET = f"{DERIVED}/training.parquet"
META_JSON = "docs/data/train_meta.json"

# Train window: last 20 years (2005–2024) for 2025 predictions
MIN_SEASON = 2005
MAX_SEASON = 2024


def _load_csv(path, usecols=None):
    if not os.path.exists(path):
        return pd.DataFrame()
    if usecols is None:
        return pd.read_csv(path, low_memory=False)
    # Only keep columns that exist
    cols = pd.read_csv(path, nrows=0).columns
    keep = [c for c in cols if c in set(usecols)]
    return pd.read_csv(path, usecols=keep, low_memory=False)


def _prep_schedule_basic(df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_schedule_columns(df.copy())
    df["game_id"] = df["game_id"].astype(str)
    for c in ("home_team", "away_team"):
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    df["date"] = pd.to_datetime(df.get("date"), errors="coerce", utc=True)
    for c in ("season", "week", "home_points", "away_points"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _build_season_averages(schedule_csv: str, team_stats_csv: str, out_parquet: str):
    """
    Build per-(team, season) averages of STAT_FEATURES using the long-form team game stats
    joined to schedule for the season label. This is used by rolling.py to pad early weeks.
    """
    if os.path.exists(out_parquet):
        # Already computed; skip to save time
        return

    sched = _load_csv(schedule_csv, usecols=["game_id", "season"])
    if sched.empty:
        raise FileNotFoundError(f"{schedule_csv} not found or empty")

    sched = _prep_schedule_basic(sched)
    # keep only training seasons to reduce size
    sched = sched[(sched["season"] >= MIN_SEASON) & (sched["season"] <= MAX_SEASON)]

    ts = _load_csv(team_stats_csv)
    if ts.empty:
        # create empty file to avoid repeated rebuild attempts
        os.makedirs(os.path.dirname(out_parquet), exist_ok=True)
        pd.DataFrame(columns=["team", "season", *STAT_FEATURES]).to_parquet(out_parquet, index=False)
        return

    ts = ts.copy()
    ts.columns = [c.lower() for c in ts.columns]
    if "game_id" not in ts.columns and "gameid" in ts.columns:
        ts = ts.rename(columns={"gameid": "game_id"})
    if "team" not in ts.columns:
        if "school" in ts.columns:
            ts = ts.rename(columns={"school": "team"})
        else:
            ts["team"] = np.nan

    # Keep only features we need
    if "stat" in ts.columns and "value" in ts.columns:
        ts = ts[ts["stat"].isin(set(STAT_FEATURES))]
        # downcast value early
        ts["value"] = pd.to_numeric(ts["value"], errors="coerce").astype("float32")
        # Merge season
        ts["game_id"] = ts["game_id"].astype(str)
        sched_small = sched[["game_id", "season"]]
        long = ts.merge(sched_small, on="game_id", how="inner")
        # pivot to columns, then groupby team-season
        wide = long.pivot_table(index=["team", "season"], columns="stat", values="value", aggfunc="mean").reset_index()
        for c in STAT_FEATURES:
            if c in wide.columns:
                wide[c] = pd.to_numeric(wide[c], errors="coerce").astype("float32")
        out = wide[["team", "season", *[c for c in STAT_FEATURES if c in wide.columns]]].copy()
    else:
        # Already wide? Try to keep only known cols + season
        keep = {"game_id", "team", *STAT_FEATURES}
        ts = ts[[c for c in ts.columns if c in keep]].copy()
        ts["game_id"] = ts["game_id"].astype(str)
        sched_small = sched[["game_id", "season"]]
        long = ts.merge(sched_small, on="game_id", how="inner")
        for c in STAT_FEATURES:
            if c in long.columns:
                long[c] = pd.to_numeric(long[c], errors="coerce").astype("float32")
        out = long.groupby(["team", "season"], as_index=False).mean(numeric_only=True)

    os.makedirs(os.path.dirname(out_parquet), exist_ok=True)
    out.to_parquet(out_parquet, index=False)


def main():
    print("Building training dataset...")

    # Make sure season averages exist for rolling backfill
    print("  Computing season averages (for prior-season padding)...")
    _build_season_averages(SCHED_CSV, STATS_CSV, SEASON_AVG_PARQUET)

    print("  Creating feature set for all historical games...")
    X, feature_list = create_feature_set()

    # Keep only training seasons and rows with actual results
    X = X[(X["season"] >= MIN_SEASON) & (X["season"] <= MAX_SEASON)].copy()
    X = X[X["home_points"].notna() & X["away_points"].notna()].copy()

    # Training target: 1 if home won
    y = (pd.to_numeric(X["home_points"], errors="coerce") >
         pd.to_numeric(X["away_points"], errors="coerce")).astype("int8")

    # Persist training table (features + label)
    os.makedirs(DERIVED, exist_ok=True)
    train_df = X.drop(columns=["home_points", "away_points"]).copy()
    train_df["home_win"] = y
    train_df.to_parquet(TRAIN_PARQUET, index=False)

    # Save metadata
    meta = {
        "features": [c for c in train_df.columns if c not in (
            "game_id", "season", "week", "home_team", "away_team",
            "neutral_site", "home_win"
        )],
        "min_season": MIN_SEASON,
        "max_season": MAX_SEASON,
    }
    os.makedirs(os.path.dirname(META_JSON), exist_ok=True)
    with open(META_JSON, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"✅ Wrote {TRAIN_PARQUET} and {META_JSON}")
    print(f"  Rows: {len(train_df)} | Features: {len(meta['features'])}")


if __name__ == "__main__":
    main()
