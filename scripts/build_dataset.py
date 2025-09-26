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
CHUNK_YEARS = 5  # <= reduce if CI is still tight (e.g., 4 or 3)

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
    for c in ("home_team", "away_team", "venue"):
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    df["date"] = pd.to_datetime(df.get("date"), errors="coerce", utc=True)
    for c in ("season","week","home_points","away_points","venue_id"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _build_season_averages(schedule_csv: str, team_stats_csv: str, out_parquet: str):
    """Precompute per-(team,season) averages for prior-season padding."""
    if os.path.exists(out_parquet):
        return

    sched = _load_csv(schedule_csv, usecols=["game_id","season"])
    if sched.empty:
        raise FileNotFoundError(f"{schedule_csv} not found or empty")
    sched = _prep_schedule(sched)
    sched = sched[(sched["season"] >= MIN_SEASON) & (sched["season"] <= MAX_SEASON)]
    sched_small = sched[["game_id","season"]].drop_duplicates()

    ts = _load_csv(team_stats_csv)
    if ts.empty:
        os.makedirs(os.path.dirname(out_parquet), exist_ok=True)
        pd.DataFrame(columns=["team","season", *STAT_FEATURES]).to_parquet(out_parquet, index=False)
        return

    ts = ts.copy()
    ts.columns = [c.lower() for c in ts.columns]
    if "game_id" not in ts.columns and "gameid" in ts.columns:
        ts = ts.rename(columns={"gameid":"game_id"})
    if "team" not in ts.columns:
        if "school" in ts.columns:
            ts = ts.rename(columns={"school":"team"})
        else:
            ts["team"] = np.nan

    if "stat" in ts.columns and "value" in ts.columns:
        ts = ts[ts["stat"].isin(set(STAT_FEATURES))]
        ts["value"] = pd.to_numeric(ts["value"], errors="coerce").astype("float32")
        ts["game_id"] = ts["game_id"].astype(str)
        long = ts.merge(sched_small, on="game_id", how="inner")
        wide = long.pivot_table(index=["team","season"], columns="stat", values="value", aggfunc="mean").reset_index()
        for c in STAT_FEATURES:
            if c in wide.columns:
                wide[c] = pd.to_numeric(wide[c], errors="coerce").astype("float32")
        out = wide[["team","season", *[c for c in STAT_FEATURES if c in wide.columns]]].copy()
    else:
        keep = {"game_id","team", *STAT_FEATURES}
        ts = ts[[c for c in ts.columns if c in keep]].copy()
        ts["game_id"] = ts["game_id"].astype(str)
        long = ts.merge(sched_small, on="game_id", how="inner")
        for c in STAT_FEATURES:
            if c in long.columns:
                long[c] = pd.to_numeric(long[c], errors="coerce").astype("float32")
        out = long.groupby(["team","season"], as_index=False).mean(numeric_only=True)

    os.makedirs(os.path.dirname(out_parquet), exist_ok=True)
    out.to_parquet(out_parquet, index=False)

def _filter_by_game_ids(df: pd.DataFrame, game_ids: set[str]) -> pd.DataFrame:
    """Filter a long/wide table to a set of game_ids, handling naming quirks."""
    if df is None or df.empty:
        return df
    x = df.copy()
    cols = {c.lower(): c for c in x.columns}
    gid_col = None
    for cand in ("game_id","gameid","GameID","GAME_ID"):
        if cand in x.columns:
            gid_col = cand
            break
        # tolerate case fold
        lc = cand.lower()
        if lc in cols:
            gid_col = cols[lc]
            break
    if gid_col is None:
        return x  # nothing to filter
    x[gid_col] = x[gid_col].astype(str)
    return x[x[gid_col].isin(game_ids)]

def main():
    print("Building training dataset...")

    # 0) Season averages for rolling backfill (once)
    print("  Computing season averages (for prior-season padding)...")
    _build_season_averages(SCHED_CSV, STATS_CSV, SEASON_AVG_PARQUET)

    # 1) Load ALL raw tables once (avoid repeated disk IO)
    sched_all = _load_csv(SCHED_CSV)
    if sched_all.empty:
        raise FileNotFoundError(f"{SCHED_CSV} is missing or empty")
    sched_all = _prep_schedule(sched_all)

    stats_all = _load_csv(STATS_CSV)              # filtered per-chunk by game_id
    lines_all = _load_csv(LINES_CSV)              # filtered per-chunk by game_id
    teams_df  = _load_csv(TEAMS_CSV)
    venues_df = _load_csv(VENUES_CSV)
    talent_df = _load_csv(TALENT_CSV)

    # 2) Iterate in season chunks to keep memory/time bounded
    chunk_frames: list[pd.DataFrame] = []
    for start in range(MIN_SEASON, MAX_SEASON + 1, CHUNK_YEARS):
        end = min(start + CHUNK_YEARS - 1, MAX_SEASON)
        print(f"  → Building features for seasons {start}-{end}...")

        sched_chunk = sched_all[(sched_all["season"] >= start) & (sched_all["season"] <= end)].copy()
        if sched_chunk.empty:
            continue

        # Completed games only for training target (still pass full chunk to features;
        # we’ll filter after feature creation to keep padding & context consistent)
        completed_mask = sched_chunk["home_points"].notna() & sched_chunk["away_points"].notna()
        # Game IDs present in this chunk (use to prune big tables before heavy pivots)
        gids = set(sched_chunk["game_id"].astype(str).unique())

        stats_chunk = _filter_by_game_ids(stats_all, gids)
        lines_chunk = _filter_by_game_ids(lines_all, gids)

        # 3) Build features for this chunk only
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

        # 4) Keep only rows with results inside the chunk
        X_chunk = X_chunk[completed_mask.reindex(X_chunk.index, fill_value=False).values] \
            if len(X_chunk) == len(sched_chunk) else \
            X_chunk[X_chunk["game_id"].isin(gids) & X_chunk["home_points"].notna() & X_chunk["away_points"].notna()]

        chunk_frames.append(X_chunk)

    if not chunk_frames:
        raise RuntimeError("No training rows built; check raw data inputs.")

    # 5) Concatenate all chunks
    X = pd.concat(chunk_frames, ignore_index=True)

    # 6) Training target: 1 if home won
    y = (pd.to_numeric(X["home_points"], errors="coerce") >
         pd.to_numeric(X["away_points"], errors="coerce")).astype("int8")

    # 7) Persist
    os.makedirs(DERIVED, exist_ok=True)
    train_df = X.drop(columns=["home_points", "away_points"]).copy()
    train_df["home_win"] = y
    train_df.to_parquet(TRAIN_PARQUET, index=False)

    # 8) Save metadata
    meta = {
        "features": [c for c in train_df.columns if c not in (
            "game_id", "season", "week", "home_team", "away_team",
            "neutral_site", "home_win"
        )],
        "min_season": MIN_SEASON,
        "max_season": MAX_SEASON,
        "chunk_years": CHUNK_YEARS
    }
    os.makedirs(os.path.dirname(META_JSON), exist_ok=True)
    with open(META_JSON, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"✅ Wrote {TRAIN_PARQUET} and {META_JSON}")
    print(f"  Rows: {len(train_df)} | Features: {len(meta['features'])}")

if __name__ == "__main__":
    main()
