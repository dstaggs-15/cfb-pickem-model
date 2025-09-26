# scripts/predict.py

from __future__ import annotations
import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

from .lib.features import create_feature_set
from .lib.parsing import ensure_schedule_columns
from .lib import market as market_lib

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

CHUNKSIZE   = 200_000


def _read_csv(path: Path, usecols=None) -> pd.DataFrame:
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
    if not path.exists():
        return []
    lines = [l.strip() for l in path.read_text().splitlines() if l.strip()]
    out = []
    for l in lines:
        if " vs " in l:
            out.append(l)
        elif "," in l:
            parts = [p.strip() for p in l.split(",")]
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
        return sched[(sched["season"] == season) & (sched["week"] == week)][
            ["game_id", "season", "week", "date", "home_team", "away_team",
             "neutral_site", "home_points", "away_points", "venue_id", "venue"]
        ].drop_duplicates()

    want = []
    for gl in games_list:
        try:
            a, b = [x.strip() for x in gl.split(" vs ", 1)]
        except ValueError:
            continue
        a = _apply_alias(a, aliases)
        b = _apply_alias(b, aliases)

        # both orientations
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
        return pd.DataFrame(columns=["game_id","season","week","date","home_team","away_team",
                                     "neutral_site","home_points","away_points","venue_id","venue"])
    return pd.concat(want, ignore_index=True).drop_duplicates()
