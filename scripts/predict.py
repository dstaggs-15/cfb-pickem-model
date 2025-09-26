# scripts/predict.py

from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable, Dict, List, Tuple

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
DEBUG_JSON  = DOCS_DATA / "debug_predict.json"

# Inputs
GAMES_TXT   = INPUT_DIR / "games.txt"
ALIASES_JSON= INPUT_DIR / "aliases.json"
MANUAL_LINES= INPUT_DIR / "manual_lines.csv"

MODEL_FILE  = DERIVED_DIR / "model.joblib"
CHUNKSIZE   = 200_000

DESIRED_SCHED_COLS = [
    "game_id","season","week","date","home_team","away_team",
    "neutral_site","home_points","away_points","venue_id","venue"
]

def _canon(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return re.sub(r"[^A-Z0-9]", "", s.upper())

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
    for c in ("season","week","home_points","away_points","venue_id"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "neutral_site" not in df.columns: df["neutral_site"] = False
    if "venue_id" not in df.columns: df["venue_id"] = pd.NA
    if "venue" not in df.columns: df["venue"] = pd.NA
    return df

def _select_cols(df: pd.DataFrame, desired: List[str]) -> pd.DataFrame:
    present = [c for c in desired if c in df.columns]
    return df[present]

def _extract_pairs_from_text(txt: str) -> List[Tuple[str,str]]:
    """Return list of (away, home) extracted from ANYWHERE in the text."""
    pairs: List[Tuple[str,str]] = []
    # A @ B  (B is home)
    for m in re.finditer(r"([A-Za-z0-9&.\' \-]+?)\s*@\s*([A-Za-z0-9&.\' \-]+)", txt):
        away = m.group(1).strip()
        home = m.group(2).strip()
        pairs.append((away, home))
    # A vs B (A is home)  -> convert to (away, home)
    for m in re.finditer(r"([A-Za-z0-9&.\' \-]+?)\s*vs\s*([A-Za-z0-9&.\' \-]+)", txt, flags=re.IGNORECASE):
        home = m.group(1).strip()
        away = m.group(2).strip()
        pairs.append((away, home))
    # A,B (A is home) -> convert to (away, home)
    for m in re.finditer(r"([A-Za-z0-9&.\' \-]+?)\s*,\s*([A-Za-z0-9&.\' \-]+)", txt):
        home = m.group(1).strip()
        away = m.group(2).strip()
        pairs.append((away, home))
    # Deduplicate while preserving order
    seen = set()
    dedup: List[Tuple[str,str]] = []
    for a,h in pairs:
        key = (a.lower(), h.lower())
        if key not in seen:
            seen.add(key)
            dedup.append((a,h))
    return dedup

def _load_games_list(path: Path) -> List[Tuple[str, str]]:
    """Parse games from file regardless of line breaks or spacing."""
    if not path.exists():
        return []
    txt = path.read_text()
    return _extract_pairs_from_text(txt)

def _load_aliases() -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    try:
        if ALIASES_JSON.exists():
            aliases = json.loads(ALIASES_JSON.read_text())
    except Exception:
        aliases = {}
    defaults = {
        "USC": "Southern California",
        "Ole Miss": "Mississippi",
        "BYU": "Brigham Young",
        "Pitt": "Pittsburgh",
        "Texas A&M": "Texas A and M",
        "UCF": "Central Florida",
        "LSU": "Louisiana State"
    }
    for k, v in defaults.items():
        aliases.setdefault(k, v)
    return aliases

def _map_alias(name: str, aliases: Dict[str, str]) -> str:
    if not isinstance(name, str):
        return name
    name = name.strip()
    return aliases.get(name, name)

def _match_games_with_week_fallback(
    sched_all: pd.DataFrame,
    season: int,
    desired_week: int,
    requested_pairs: List[Tuple[str, str]],
    aliases: Dict[str, str],
) -> tuple[pd.DataFrame, int, dict]:
    debug = {"season": season, "requested_week": desired_week, "tried": []}

    pool = sched_all[sched_all["season"] == season].copy()
    pool["home_c"] = pool["home_team"].map(_canon)
    pool["away_c"] = pool["away_team"].map(_canon)

    req = []
    for away, home in requested_pairs:
        away_m = _canon(_map_alias(away, aliases))
        home_m = _canon(_map_alias(home, aliases))
        req.append((away_m, home_m))
    debug["requested_pairs_canon"] = req

    def find_for_week(wk: int) -> pd.DataFrame:
        cand = pool[pool["week"] == wk].copy()
        if cand.empty: return cand
        hits = []
        for (away_c, home_c) in req:
            r = cand[
                ((cand["home_c"] == home_c) & (cand["away_c"] == away_c)) |
                ((cand["home_c"] == away_c) & (cand["away_c"] == home_c))
            ]
            if not r.empty:
                hits.append(r)
        return pd.concat(hits, ignore_index=True).drop_duplicates() if hits else cand.iloc[0:0].copy()

    # exact
    rows = find_for_week(desired_week)
    debug["tried"].append({"week": desired_week, "matched": int(len(rows))})
    if not rows.empty:
        return _select_cols(rows, DESIRED_SCHED_COLS), desired_week, debug

    # ±1
    for wk in (desired_week - 1, desired_week + 1):
        if wk >= 0:
            r = find_for_week(wk)
            debug["tried"].append({"week": wk, "matched": int(len(r))})
            if not r.empty:
                return _select_cols(r, DESIRED_SCHED_COLS), wk, debug

    # any week
    out_rows = []
    weeks_hit = set()
    for wk in sorted(pool["week"].dropna().unique()):
        r = find_for_week(int(wk))
        if not r.empty:
            out_rows.append(r)
            weeks_hit.add(int(wk))
    rows_any = pd.concat(out_rows, ignore_index=True).drop_duplicates() if out_rows else pool.iloc[0:0].copy()
    debug["tried"].append({"week": "ANY", "matched": int(len(rows_any)), "weeks_hit": sorted(list(weeks_hit))})
    if not rows_any.empty:
        best_wk = None
        best_count = -1
        for wk in sorted(weeks_hit):
            cnt = len(rows_any[rows_any["week"] == wk])
            if cnt > best_count:
                best_count, best_wk = cnt, wk
        return _select_cols(rows_any[rows_any["week"] == best_wk], DESIRED_SCHED_COLS), int(best_wk), debug

    return rows_any, desired_week, debug  # empty

def _stream_filter_by_gids(csv_path: Path, gids: set[str], candidate_cols=("game_id","gameid")) -> pd.DataFrame:
    if not csv_path.exists() or not gids:
        return pd.DataFrame()
    header = pd.read_csv(csv_path, nrows=0)
    columns = list(header.columns)
    gid_col = None
    for cand in candidate_cols:
        for c in columns:
            if c.lower() == cand.lower():
                gid_col = c
                break
        if gid_col: break
    keep = []
    for chunk in pd.read_csv(csv_path, chunksize=CHUNKSIZE, low_memory=False):
        if gid_col is None: continue
        chunk[gid_col] = chunk[gid_col].astype(str)
        piece = chunk[chunk[gid_col].isin(gids)]
        if not piece.empty: keep.append(piece)
    return pd.concat(keep, ignore_index=True) if keep else pd.DataFrame()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--season", type=int, default=None)
    args = ap.parse_args()

    if not MODEL_FILE.exists():
        print(f"ERROR: {MODEL_FILE} not found.", file=sys.stderr); sys.exit(2)
    model = joblib_load(MODEL_FILE)

    if not META_JSON.exists():
        print(f"ERROR: {META_JSON} not found.", file=sys.stderr); sys.exit(2)
    meta = json.loads(META_JSON.read_text())
    feature_names: list[str] = meta.get("features", [])

    sched_all = _read_csv(SCHED_CSV)
    if sched_all.empty:
        print("ERROR: schedule CSV not found or empty.", file=sys.stderr); sys.exit(2)
    sched_all = _prep_schedule(sched_all)

    season = args.season or int(sched_all["season"].max())
    week   = int(args.week)
    print(f"Predicting season={season}, week={week}")

    requested_pairs = _load_games_list(GAMES_TXT)   # list of (away, home)
    if not requested_pairs:
        print(f"NOTE: {GAMES_TXT} empty or unparsable.")
    aliases = _load_aliases()

    pred_rows, matched_week, dbg = _match_games_with_week_fallback(
        sched_all, season, week, requested_pairs, aliases
    )
    dbg["matched_week"] = matched_week
    dbg["requested_pairs"] = requested_pairs
    dbg["matched_games_preview"] = pred_rows[["home_team","away_team","season","week"]].to_dict(orient="records") if not pred_rows.empty else []
    print(f"Matched {len(pred_rows)} games (using week {matched_week}).")

    if pred_rows.empty:
        DOCS_DATA.mkdir(parents=True, exist_ok=True)
        with open(PRED_JSON, "w") as f: json.dump({"games": []}, f, indent=2)
        with open(DEBUG_JSON, "w") as f: json.dump(dbg, f, indent=2)
        print(f"Wrote 0 predictions to {PRED_JSON}. Debug in {DEBUG_JSON}.")
        return

    gids = set(pred_rows["game_id"].astype(str).unique())
    teams_df  = _read_csv(TEAMS_CSV)
    venues_df = _read_csv(VENUES_CSV)
    talent_df = _read_csv(TALENT_CSV)
    stats_chunk = _stream_filter_by_gids(STATS_CSV, gids)
    lines_chunk = _stream_filter_by_gids(LINES_CSV,  gids)

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
    Xp = X_all[X_all["game_id"].isin(gids)].copy()

    for c in feature_names:
        if c not in Xp.columns: Xp[c] = 0.0
    Xp = Xp[["game_id","home_team","away_team","neutral_site"] + feature_names].copy()

    probs = model.predict_proba(Xp[feature_names])[:, 1]  # P(home win)

    out = []
    for (gid, home, away, ns, p_home) in zip(
        Xp["game_id"], Xp["home_team"], Xp["away_team"], Xp.get("neutral_site", False), probs
    ):
        pick = home if p_home >= 0.5 else away
        out.append({
            "home_team": str(home),
            "away_team": str(away),
            "neutral_site": bool(ns),
            "model_prob_home": float(round(p_home, 4)),
            "pick": pick,
            "explanation": []
        })

    DOCS_DATA.mkdir(parents=True, exist_ok=True)
    with open(PRED_JSON, "w") as f: json.dump({"games": out}, f, indent=2)
    with open(DEBUG_JSON, "w") as f: json.dump(dbg, f, indent=2)

    print(f"Wrote {len(out)} predictions to {PRED_JSON} (matched on week {matched_week}). Debug in {DEBUG_JSON}.")

if __name__ == "__main__":
    main()
