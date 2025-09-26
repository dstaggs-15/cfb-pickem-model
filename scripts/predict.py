# scripts/predict.py

from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from joblib import load as joblib_load

from .lib.features import create_feature_set
from .lib.parsing import ensure_schedule_columns

DERIVED_DIR = Path("data/derived")
RAW_DIR     = Path("data/raw/cfbd")
DOCS_DATA   = Path("docs/data")
INPUT_DIR   = Path("docs/input")

# Raw CSVs
SCHED_CSV   = RAW_DIR / "cfb_schedule.csv"
STATS_CSV   = RAW_DIR / "cfb_game_team_stats.csv"
LINES_CSV   = RAW_DIR / "cfb_lines.csv"
TEAMS_CSV   = RAW_DIR / "cfbd_teams.csv"
VENUES_CSV  = RAW_DIR / "cfbd_venues.csv"
TALENT_CSV  = RAW_DIR / "cfbd_talent.csv"

# Outputs
PRED_JSON   = DOCS_DATA / "predictions.json"
META_JSON   = DOCS_DATA / "train_meta.json"

# Inputs
GAMES_TXT   = INPUT_DIR / "games.txt"
ALIASES_JSON= INPUT_DIR / "aliases.json"
MANUAL_LINES= INPUT_DIR / "manual_lines.csv"

MODEL_FILE  = DERIVED_DIR / "model.joblib"
CHUNKSIZE   = 200_000


def _read_csv(path: Path, usecols: Iterable[str] | None = None) -> pd.DataFrame:
    if not path.exists():
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
    for c in ("season", "week", "home_points", "away_points", "venue_id"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _load_games_list(path: Path) -> list[str]:
    """
    Accepts formats:
      - 'A @ B'  (B is home)
      - 'A vs B'
      - 'A,B'
    Normalizes to 'Home vs Away' order for matching logic (but we still match both orientations).
    """
    if not path.exists():
        return []
    lines = [l.strip() for l in path.read_text().splitlines() if l.strip()]
    out: list[str] = []
    for l in lines:
        if " @ " in l:
            a, b = [p.strip() for p in l.split(" @ ", 1)]
            # '@' means right side is HOME
            out.append(f"{b} vs {a}")
        elif " vs " in l:
            out.append(l)
        elif "," in l:
            parts = [p.strip() for p in l.split(",", 1)]
            if len(parts) == 2:
                out.append(f"{parts[0]} vs {parts[1]}")
    return out


def _apply_alias(name: str, aliases: dict) -> str:
    if not isinstance(name, str):
        return name
    n = name.strip()
    return aliases.get(n, n)


def _make_predict_rows(sched: pd.DataFrame, games_list: list[str], season: int, week: int, aliases: dict) -> pd.DataFrame:
    """
    Build a DataFrame of the games we want to predict (one row each),
    aligned to schedule so we pick up game_id/neutral_site/etc.
    """
    if not games_list:
        df = sched[(sched["season"] == season) & (sched["week"] == week)][
            ["game_id", "season", "week", "date", "home_team", "away_team",
             "neutral_site", "home_points", "away_points", "venue_id", "venue"]
        ].drop_duplicates()
        return df

    want = []
    for gl in games_list:
        try:
            a, b = [x.strip() for x in gl.split(" vs ", 1)]
        except ValueError:
            continue
        a = _apply_alias(a, aliases)
        b = _apply_alias(b, aliases)

        rows = sched[
            (sched["season"] == season) &
            (sched["week"] == week) &
            (
                ((sched["home_team"] == a) & (sched["away_team"] == b)) |
                ((sched["home_team"] == b) & (sched["away_team"] == a))
            )
        ][["game_id", "season", "week", "date", "home_team", "away_team",
           "neutral_site", "home_points", "away_points", "venue_id", "venue"]]

        if not rows.empty:
            want.append(rows)

    if not want:
        return pd.DataFrame(columns=[
            "game_id","season","week","date","home_team","away_team",
            "neutral_site","home_points","away_points","venue_id","venue"
        ])
    return pd.concat(want, ignore_index=True).drop_duplicates()


def _stream_filter_by_gids(csv_path: Path, gids: set[str], candidate_cols=("game_id","gameid")) -> pd.DataFrame:
    """
    Stream a large CSV and return only rows whose game_id is in 'gids'.
    """
    if not csv_path.exists() or not gids:
        return pd.DataFrame()

    header = pd.read_csv(csv_path, nrows=0)
    columns = list(header.columns)
    # Find the actual name of the game_id column
    gid_col = None
    for cand in candidate_cols:
        for c in columns:
            if c.lower() == cand.lower():
                gid_col = c
                break
        if gid_col:
            break

    keep_chunks = []
    for chunk in pd.read_csv(csv_path, chunksize=CHUNKSIZE, low_memory=False):
        if gid_col is None:
            # can't filter w/o a game id; keep nothing (defensive)
            continue
        chunk[gid_col] = chunk[gid_col].astype(str)
        piece = chunk[chunk[gid_col].isin(gids)]
        if not piece.empty:
            keep_chunks.append(piece)

    return pd.concat(keep_chunks, ignore_index=True) if keep_chunks else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--week", type=int, required=True, help="CFB week to predict (e.g., 5)")
    ap.add_argument("--season", type=int, default=None, help="Override season (default: latest 'season' in schedule)")
    args = ap.parse_args()

    # Load artifacts
    if not MODEL_FILE.exists():
        print(f"ERROR: {MODEL_FILE} not found. Run scripts.train_model first.", file=sys.stderr)
        sys.exit(2)
    model = joblib_load(MODEL_FILE)

    if not META_JSON.exists():
        print(f"ERROR: {META_JSON} not found. Run scripts.build_dataset first.", file=sys.stderr)
        sys.exit(2)
    meta = json.loads(META_JSON.read_text())
    feature_names: list[str] = meta.get("features", [])

    # Load schedule + determine season
    sched_all = _read_csv(SCHED_CSV)
    if sched_all.empty:
        print("ERROR: schedule CSV not found or empty.", file=sys.stderr)
        sys.exit(2)
    sched_all = _prep_schedule(sched_all)

    season = args.season or int(sched_all["season"].max())
    week   = int(args.week)
    print(f"Predicting season={season}, week={week}")

    # Inputs
    games_list = _load_games_list(GAMES_TXT)
    if not games_list:
        print(f"NOTE: {GAMES_TXT} is empty or missing; predicting all games in season={season} week={week}")
    aliases = {}
    if ALIASES_JSON.exists():
        try:
            aliases = json.loads(ALIASES_JSON.read_text())
        except Exception:
            aliases = {}
    # Helpful defaults (harmless if already present)
    for k, v in {
        "USC": "Southern California",
        "Ole Miss": "Mississippi",
        "BYU": "Brigham Young"
    }.items():
        aliases.setdefault(k, v)

    # Build the predict rows by matching to schedule
    pred_rows = _make_predict_rows(sched_all, games_list, season=season, week=week, aliases=aliases)
    print(f"Matched {len(pred_rows)} games from input list.")
    if pred_rows.empty:
        # Write an empty file so the frontend doesn't show old games
        DOCS_DATA.mkdir(parents=True, exist_ok=True)
        with open(PRED_JSON, "w") as f:
            json.dump({"games": []}, f, indent=2)
        print(f"Wrote 0 predictions to {PRED_JSON}")
        return

    gids = set(pred_rows["game_id"].astype(str).unique())

    # Load small reference tables and stream-filter the big ones by game_id
    teams_df  = _read_csv(TEAMS_CSV)
    venues_df = _read_csv(VENUES_CSV)
    talent_df = _read_csv(TALENT_CSV)

    stats_chunk = _stream_filter_by_gids(STATS_CSV, gids)
    lines_chunk = _stream_filter_by_gids(LINES_CSV,  gids)

    # Create features only for the selected games
    X_all, feat_list = create_feature_set(
        schedule=sched_all[sched_all["game_id"].isin(gids)].copy(),
        team_stats=stats_chunk,
        venues_df=venues_df,
        teams_df=teams_df,
        talent_df=talent_df,
        lines_df=lines_chunk,
        manual_lines_df=_read_csv(MANUAL_LINES) if MANUAL_LINES.exists() else None,
        games_to_predict_df=pred_rows
    )

    # Keep only our target games (defensive)
    Xp = X_all[X_all["game_id"].isin(gids)].copy()

    # Order columns as training features; missing => fill with 0.0
    for c in feature_names:
        if c not in Xp.columns:
            Xp[c] = 0.0
    Xp = Xp[["game_id","home_team","away_team"] + feature_names].copy()

    # Model probabilities for home team
    probs = model.predict_proba(Xp[feature_names])[:, 1]  # assuming binary classifier with class 1 = home win
    preds = []
    for (gid, home, away, p_home) in zip(Xp["game_id"], Xp["home_team"], Xp["away_team"], probs):
        pick = home if p_home >= 0.5 else away
        preds.append({
            "home_team": str(home),
            "away_team": str(away),
            "neutral_site": bool(False),  # could be added from schedule if desired
            "model_prob_home": float(round(p_home, 4)),
            "pick": pick,
            "explanation": []
        })

    DOCS_DATA.mkdir(parents=True, exist_ok=True)
    with open(PRED_JSON, "w") as f:
        json.dump({"games": preds}, f, indent=2)
    print(f"Wrote {len(preds)} predictions to {PRED_JSON}")


if __name__ == "__main__":
    main()
