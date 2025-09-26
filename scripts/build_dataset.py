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

SCHED_CSV  = f"{RAW_DIR}/cfb_schedule.csv"
STATS_CSV  = f"{RAW_DIR}/cfb_game_team_stats.csv"
LINES_CSV  = f"{RAW_DIR}/cfb_lines.csv"
TEAMS_CSV  = f"{RAW_DIR}/cfbd_teams.csv"
VENUES_CSV = f"{RAW_DIR}/cfbd_venues.csv"
TALENT_CSV = f"{RAW_DIR}/cfbd_talent.csv"

TRAIN_PARQUET = f"{DERIVED}/training.parquet"
META_JSON     = "docs/data/train_meta.json"

# Train window: last 20 years (2005–2024) for 2025 predictions
MIN_SEASON  = 2005
MAX_SEASON  = 2024
CHUNK_YEARS = 3       # keep small to stay under CI memory/time
CHUNKSIZE   = 300_000 # streaming rows per read_csv chunk (tune if needed)


def _load_csv(path, usecols=None):
    if not os.path.exists(path):
        return pd.DataFrame()
    if usecols is None:
        return pd.read_csv(path, low_memory=False)
    cols = pd.read_csv(path, nrows=0).columns
    keep = [c for c in cols if c in set(usecols)]
    return pd.read_csv(path, usecols=keep, low_memory=False)


def _prep_schedule(df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_schedule_columns(df.copy())
    df["game_id"] = df["game_id"].astype(str)
    for c in ("home_team","away_team","venue"):
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    df["date"] = pd.to_datetime(df.get("date"), errors="coerce", utc=True)
    for c in ("season","week","home_points","away_points","venue_id"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _build_season_averages(schedule_csv: str, team_stats_csv: str, out_parquet: str):
    """Precompute per-(team,season) averages for prior-season padding, streamed to avoid RAM spikes."""
    if os.path.exists(out_parquet):
        return

    sched = _load_csv(schedule_csv, usecols=["game_id","season"])
    if sched.empty:
        raise FileNotFoundError(f"{schedule_csv} not found or empty")
    sched = _prep_schedule(sched)
    sched = sched[(sched["season"] >= MIN_SEASON) & (sched["season"] <= MAX_SEASON)]
    sched_small = sched[["game_id","season"]].drop_duplicates()
    gid_to_season = dict(zip(sched_small["game_id"], sched_small["season"]))

    if not os.path.exists(team_stats_csv):
        os.makedirs(os.path.dirname(out_parquet), exist_ok=True)
        pd.DataFrame(columns=["team","season", *STAT_FEATURES]).to_parquet(out_parquet, index=False)
        return

    # Stream the stats file, filter on STAT_FEATURES and sched GIDs, then aggregate
    acc = []
    # Try to determine columns first
    header_cols = pd.read_csv(team_stats_csv, nrows=0).columns.str.lower().tolist()
    # typical columns: game_id/gameid, team/school, stat, value
    usecols = []
    for c in header_cols:
        if c in ("game_id","gameid","team","school","stat","value"):
            usecols.append(c)
    if not usecols:
        usecols = None  # fall back to all cols

    reader = pd.read_csv(team_stats_csv, chunksize=CHUNKSIZE, usecols=usecols, low_memory=False)
    want_stats = set(STAT_FEATURES)

    for chunk in reader:
        chunk.columns = [c.lower() for c in chunk.columns]
        if "game_id" not in chunk.columns and "gameid" in chunk.columns:
            chunk = chunk.rename(columns={"gameid":"game_id"})
        if "team" not in chunk.columns and "school" in chunk.columns:
            chunk = chunk.rename(columns={"school":"team"})
        if "game_id" not in chunk.columns or "team" not in chunk.columns:
            continue
        chunk["game_id"] = chunk["game_id"].astype(str)

        # Keep only rows whose game_id is in our training seasons
        chunk = chunk[chunk["game_id"].isin(gid_to_season.keys())]
        if chunk.empty:
            continue

        if "stat" in chunk.columns and "value" in chunk.columns:
            chunk = chunk[chunk["stat"].isin(want_stats)]
            if chunk.empty:
                continue
            chunk["value"] = pd.to_numeric(chunk["value"], errors="coerce").astype("float32")
            # attach season via map
            chunk["season"] = chunk["game_id"].map(gid_to_season)
            # pivot inside the chunk
            wide = chunk.pivot_table(index=["team","season"], columns="stat", values="value", aggfunc="mean").reset_index()
            acc.append(wide)
        else:
            # already wide? just keep columns if they exist
            keep = ["game_id","team"] + [c for c in STAT_FEATURES if c in chunk.columns]
            chunk = chunk[[c for c in keep if c in chunk.columns]].copy()
            chunk["season"] = chunk["game_id"].map(gid_to_season)
            acc.append(chunk.groupby(["team","season"], as_index=False).mean(numeric_only=True))

    if acc:
        combined = pd.concat(acc, ignore_index=True)
        # ensure float32
        for c in STAT_FEATURES:
            if c in combined.columns:
                combined[c] = pd.to_numeric(combined[c], errors="coerce").astype("float32")
        out = combined.groupby(["team","season"], as_index=False).mean(numeric_only=True)
    else:
        out = pd.DataFrame(columns=["team","season", *STAT_FEATURES])

    os.makedirs(os.path.dirname(out_parquet), exist_ok=True)
    out.to_parquet(out_parquet, index=False)


def _stream_filter_by_gids(csv_path: str, gids: set[str], candidate_cols: tuple[str, ...]) -> pd.DataFrame:
    """
    Stream a large CSV and return only rows whose game_id is in 'gids'.
    candidate_cols are possible names for the game_id column.
    """
    if not os.path.exists(csv_path) or not gids:
        return pd.DataFrame()

    # Pre-read header to choose usecols that include the game_id column + columns we may need
    header = pd.read_csv(csv_path, nrows=0)
    cols = {c.lower(): c for c in header.columns}
    gid_col = None
    for cand in candidate_cols:
        lc = cand.lower()
        if lc in cols:
            gid_col = cols[lc]
            break

    usecols = list(header.columns)  # default
    # For lines, we don't need every column; but keeping header-selected is fine and robust.

    keep_chunks = []
    for chunk in pd.read_csv(csv_path, chunksize=CHUNKSIZE, usecols=usecols, low_memory=False):
        # normalize a game_id col
        if gid_col is None:
            # can't filter w/o a gid
            keep_chunks.append(chunk)
            continue
        chunk[gid_col] = chunk[gid_col].astype(str)
        piece = chunk[chunk[gid_col].isin(gids)]
        if not piece.empty:
            keep_chunks.append(piece)

    return pd.concat(keep_chunks, ignore_index=True) if keep_chunks else pd.DataFrame()


def main():
    print("Building training dataset...")

    # 0) Season averages for rolling backfill
    print("  Computing season averages (for prior-season padding)...")
    _build_season_averages(SCHED_CSV, STATS_CSV, SEASON_AVG_PARQUET)

    # 1) Load schedule & small tables once
    sched_all = _load_csv(SCHED_CSV)
    if sched_all.empty:
        raise FileNotFoundError(f"{SCHED_CSV} is missing or empty")
    sched_all = _prep_schedule(sched_all)

    teams_df  = _load_csv(TEAMS_CSV)
    venues_df = _load_csv(VENUES_CSV)
    talent_df = _load_csv(TALENT_CSV)

    # 2) Process in small season chunks
    chunk_frames: list[pd.DataFrame] = []
    for start in range(MIN_SEASON, MAX_SEASON + 1, CHUNK_YEARS):
        end = min(start + CHUNK_YEARS - 1, MAX_SEASON)
        print(f"  → Building features for seasons {start}-{end}...")

        sched_chunk = sched_all[(sched_all["season"] >= start) & (sched_all["season"] <= end)].copy()
        if sched_chunk.empty:
            continue

        gids = set(sched_chunk["game_id"].astype(str).unique())

        # Stream-filter big CSVs by gids (no huge in-memory load)
        stats_chunk = _stream_filter_by_gids(STATS_CSV, gids, candidate_cols=("game_id","gameid"))
        lines_chunk = _stream_filter_by_gids(LINES_CSV,  gids, candidate_cols=("game_id","gameid"))

        # Build features for this slice only
        X_chunk, feat_cols = create_feature_set(
            schedule=sched_chunk,
            team_stats=stats_chunk,
            venues_df=venues_df,
            teams_df=teams_df,
            talent_df=talent_df,
            lines_df=lines_chunk,
            manual_lines_df=None,
            games_to_predict_df=None
        )

        # Keep only completed games
        X_chunk = X_chunk[X_chunk["home_points"].notna() & X_chunk["away_points"].notna()]
        if not X_chunk.empty:
            chunk_frames.append(X_chunk)

    if not chunk_frames:
        raise RuntimeError("No training rows built; check raw data inputs.")

    X = pd.concat(chunk_frames, ignore_index=True)

    # Target
    y = (pd.to_numeric(X["home_points"], errors="coerce") >
         pd.to_numeric(X["away_points"], errors="coerce")).astype("int8")

    # Save
    os.makedirs(DERIVED, exist_ok=True)
    train_df = X.drop(columns=["home_points","away_points"]).copy()
    train_df["home_win"] = y
    train_df.to_parquet(TRAIN_PARQUET, index=False)

    meta = {
        "features": [c for c in train_df.columns if c not in (
            "game_id","season","week","home_team","away_team","neutral_site","home_win"
        )],
        "min_season": MIN_SEASON,
        "max_season": MAX_SEASON,
        "chunk_years": CHUNK_YEARS,
        "chunksize": CHUNKSIZE
    }
    os.makedirs(os.path.dirname(META_JSON), exist_ok=True)
    with open(META_JSON, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"✅ Wrote {TRAIN_PARQUET} and {META_JSON}")
    print(f"  Rows: {len(train_df)} | Features: {len(meta['features'])}")


if __name__ == "__main__":
    main()
