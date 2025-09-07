# scripts/lib/features.py
# Creates usable numeric features so training doesn't choke.
#
# Data sources (CSV paths are what your repo uses already):
#   - data/raw/cfbd/cfb_schedule.csv         (REQUIRED)
#   - data/raw/cfbd/cfb_lines.csv            (OPTIONAL)
#
# Feature families:
#   A) Rolling team form from schedule (no API stats required):
#       - margin, points for/against rolling means (L3, L5)
#       - win rate (L5)
#   B) Vegas lines (optional — if cfb_lines.csv exists):
#       - home_closing_spread, home_total (numeric best-effort)
#
# Output: X (with labels kept for train), feature_list (only predictors)

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
    # Normalize expected columns
    need = ["game_id", "season", "week", "home_team", "away_team", "date", "home_points", "away_points"]
    for col in need:
        if col not in df.columns:
            df[col] = pd.NA
    # Types
    df["game_id"] = pd.to_numeric(df["game_id"], errors="coerce")
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df["week"] = pd.to_numeric(df["week"], errors="coerce")
    df["home_points"] = pd.to_numeric(df["home_points"], errors="coerce")
    df["away_points"] = pd.to_numeric(df["away_points"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    # Sort deterministically
    df = df.sort_values(["season", "week", "date", "game_id"], na_position="last")
    # Drop rows without ids
    df = df[df["game_id"].notna()].copy()
    df["game_id"] = df["game_id"].astype("int64")
    return df


def _melt_schedule_to_team_games(sched: pd.DataFrame) -> pd.DataFrame:
    # Create a per-team/per-game view to compute rolling features.
    # Two rows per game: one for home, one for away.
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

    cols = ["game_id", "season", "week", "date", "team", "opponent", "is_home", "points_for", "points_against"]
    long = pd.concat([home[cols], away[cols]], ignore_index=True)
    long["margin"] = pd.to_numeric(long["points_for"], errors="coerce") - pd.to_numeric(long["points_against"], errors="coerce")
    long["win"] = (long["margin"] > 0).astype("float")
    return long


def _add_rolling_form(long: pd.DataFrame) -> pd.DataFrame:
    # Rolling L3/L5 over *prior* games only (shift before rolling)
    long = long.sort_values(["team", "date", "game_id"])
    grp = long.groupby("team", group_keys=False)

    def _roll(s, w):
        return s.shift(1).rolling(w, min_periods=1).mean()

    long["pf_l3"] = grp["points_for"].apply(lambda s: _roll(s, 3))
    long["pa_l3"] = grp["points_against"].apply(lambda s: _roll(s, 3))
    long["margin_l3"] = grp["margin"].apply(lambda s: _roll(s, 3))

    long["pf_l5"] = grp["points_for"].apply(lambda s: _roll(s, 5))
    long["pa_l5"] = grp["points_against"].apply(lambda s: _roll(s, 5))
    long["margin_l5"] = grp["margin"].apply(lambda s: _roll(s, 5))

    # Win rate last 5
    long["winrate_l5"] = grp["win"].apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean())

    # Home/away-specific rolling margin (small signal)
    long["home_margin_l3"] = grp.apply(
        lambda g: g.assign(_hm=g["margin"].where(g["is_home"] == 1).shift(1).rolling(3, min_periods=1).mean())
    )["_hm"].values
    long["away_margin_l3"] = grp.apply(
        lambda g: g.assign(_am=g["margin"].where(g["is_home"] == 0).shift(1).rolling(3, min_periods=1).mean())
    )["_am"].values

    return long


def _pivot_back_to_game(long_with_rolls: pd.DataFrame) -> pd.DataFrame:
    # Split back to home/away rows, then merge on game_id
    home_side = long_with_rolls[long_with_rolls["is_home"] == 1].copy()
    away_side = long_with_rolls[long_with_rolls["is_home"] == 0].copy()

    keep_feats = [
        "pf_l3", "pa_l3", "margin_l3",
        "pf_l5", "pa_l5", "margin_l5",
        "winrate_l5",
        "home_margin_l3", "away_margin_l3",
    ]
    home_feats = home_side[["game_id", "team"] + keep_feats].add_prefix("home_")
    home_feats = home_feats.rename(columns={"home_game_id": "game_id", "home_team": "home_team_name"})

    away_feats = away_side[["game_id", "team"] + keep_feats].add_prefix("away_")
    away_feats = away_feats.rename(columns={"away_game_id": "game_id", "away_team": "away_team_name"})

    roll = pd.merge(home_feats, away_feats, on="game_id", how="inner")
    return roll


def _load_lines_features(schedule: pd.DataFrame) -> pd.DataFrame:
    """Best-effort parse of lines to get a numeric home spread and total.
       If file missing or malformed, returns empty DF to be ignored."""
    if not os.path.exists(LINES_CSV):
        return pd.DataFrame()

    df = pd.read_csv(LINES_CSV)
    if df.empty:
        return pd.DataFrame()

    # Try to identify game_id
    if "game_id" not in df.columns:
        # CFBD "id" can sometimes be the game id; try it
        if "id" in df.columns:
            df = df.rename(columns={"id": "game_id"})
        else:
            return pd.DataFrame()

    # Attempt to find provider and closing numbers
    # Common columns seen: "spread", "overUnder", sometimes nested by provider.
    # We'll pick the last row per game_id (assuming later = closer to close).
    df["game_id"] = pd.to_numeric(df["game_id"], errors="coerce")
    df = df[df["game_id"].notna()].copy()
    df["game_id"] = df["game_id"].astype("int64")

    # Normalize possible spread columns
    spread_cols = [c for c in df.columns if c.lower() in {"spread", "spread_close", "closing_spread", "close_spread"}]
    total_cols = [c for c in df.columns if c.lower() in {"overunder", "total", "closing_total", "total_close"}]

    def _pick_numeric(series):
        s = pd.to_numeric(series, errors="coerce")
        return s

    # If we have multiple rows per game_id (different books/updates), keep the last
    df = df.sort_values(["game_id"]).groupby("game_id", as_index=False).tail(1)

    out = df[["game_id"]].copy()
    if spread_cols:
        out["home_closing_spread"] = _pick_numeric(df[spread_cols[0]])
    if total_cols:
        out["home_total"] = _pick_numeric(df[total_cols[0]])

    # Some feeds give spreads from the perspective of the home team already (home negative if favored).
    # If not, this will still be a weak signal; we won't try to flip by favorite because formats vary widely.
    return out.drop_duplicates("game_id")


def create_feature_set(
    # Keep signature stable for build_dataset.py
    team_stats_sided: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Return (X, feature_list). X includes identifiers + labels so downstream stays unchanged,
       but feature_list only contains predictor names."""
    schedule = _load_schedule()
    print(f"[FEATURES] schedule (raw normalized): shape={schedule.shape} cols={list(schedule.columns)}")
    print(schedule.head().to_string(index=False))

    # Build rolling features from schedule only
    team_long = _melt_schedule_to_team_games(schedule)
    team_long = _add_rolling_form(team_long)

    roll = _pivot_back_to_game(team_long)

    # Merge identifiers and labels back in
    id_cols = ["game_id", "season", "week", "home_points", "away_points"]
    base = schedule[id_cols].drop_duplicates("game_id")

    X = base.merge(roll, on="game_id", how="left")

    # Optional: add Vegas features if present
    lines_feats = _load_lines_features(schedule)
    if not lines_feats.empty:
        X = X.merge(lines_feats, on="game_id", how="left")

    # Final cleanup: all features numeric, fill small NaNs with group means
    feature_cols = [c for c in X.columns if c not in ["game_id", "season", "week", "home_points", "away_points"]]
    # Coerce to numeric
    for c in feature_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # Simple imputation: fill per-season means, then global
    for c in feature_cols:
        X[c] = X.groupby("season")[c].transform(lambda s: s.fillna(s.mean()))
        X[c] = X[c].fillna(X[c].mean())

    # Remove all-NaN columns (if any)
    kept_feats = [c for c in feature_cols if X[c].notna().any()]

    print(f"[FEATURES] engineered features: count={len(kept_feats)} sample={kept_feats[:10]}")
    return X[["game_id", "season", "week", "home_points", "away_points"] + kept_feats].copy(), kept_feats
