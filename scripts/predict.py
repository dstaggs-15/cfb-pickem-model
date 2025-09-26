# scripts/predict.py

from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable, Dict, List, Tuple, Optional

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
MANUAL_LINES= INPUT_DIR / "manual_lines.csv"

MODEL_FILE  = DERIVED_DIR / "model.joblib"
CHUNKSIZE   = 200_000

DESIRED_SCHED_COLS = [
    "game_id","season","week","date","home_team","away_team",
    "neutral_site","home_points","away_points","venue_id","venue"
]

# -----------------------
# String normalization
# -----------------------
STOP = {
    "THE","OF","UNIVERSITY","UNIV","U","STATE","ST","&","AND","AT"
}

def _canon(s: str) -> str:
    """Aggressive canonical form used for equality checks."""
    if not isinstance(s, str): return ""
    s = s.upper()
    s = s.replace("&", " AND ")
    s = s.replace("A&M", "A AND M").replace("A & M", "A AND M")
    s = s.replace(".", " ").replace("'", " ")
    # expand common abbreviations
    s = re.sub(r"\bST\b", "STATE", s)
    s = re.sub(r"\bPENN ST\b", "PENN STATE", s)
    s = re.sub(r"\bKANSAS ST\b", "KANSAS STATE", s)
    s = re.sub(r"\bOLE MISS\b", "MISSISSIPPI", s)
    s = re.sub(r"\bUSC\b", "SOUTHERN CALIFORNIA", s)
    s = re.sub(r"\bBYU\b", "BRIGHAM YOUNG", s)
    s = re.sub(r"\bUCF\b", "CENTRAL FLORIDA", s)
    s = re.sub(r"\bLSU\b", "LOUISIANA STATE", s)
    s = re.sub(r"\bPITT\b", "PITTSBURGH", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _tokens(s: str) -> List[str]:
    """Token set for fuzzy overlap scoring."""
    if not isinstance(s, str): return []
    s = _canon(s)
    toks = re.split(r"[^A-Z0-9]+", s)
    return [t for t in toks if t and t not in STOP]

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
    for c in ("home_team","away_team","venue"):
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    df["date"] = pd.to_datetime(df.get("date"), errors="coerce", utc=True)
    for c in ("season","week","home_points","away_points","venue_id"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "neutral_site" not in df.columns: df["neutral_site"] = False
    if "venue_id" not in df.columns: df["venue_id"] = pd.NA
    if "venue" not in df.columns: df["venue"] = pd.NA
    # canonical + tokens for fuzzy match
    df["home_c"] = df["home_team"].map(_canon)
    df["away_c"] = df["away_team"].map(_canon)
    df["home_toks"] = df["home_team"].map(_tokens)
    df["away_toks"] = df["away_team"].map(_tokens)
    return df

def _select_cols(df: pd.DataFrame, desired: List[str]) -> pd.DataFrame:
    present = [c for c in desired if c in df.columns]
    return df[present]

def _extract_pairs_from_text(txt: str) -> List[Tuple[str,str]]:
    """Return list of (away, home) extracted from ANYWHERE in the text."""
    pairs: List[Tuple[str,str]] = []
    # A @ B  (B is home)
    for m in re.finditer(r"([A-Za-z0-9&.\' \-]+?)\s*@\s*([A-Za-z0-9&.\' \-]+)", txt):
        away = m.group(1).strip(); home = m.group(2).strip()
        pairs.append((away, home))
    # A vs B (A is home)  -> convert to (away, home)
    for m in re.finditer(r"([A-Za-z0-9&.\' \-]+?)\s*vs\s*([A-Za-z0-9&.\' \-]+)", txt, flags=re.IGNORECASE):
        home = m.group(1).strip(); away = m.group(2).strip()
        pairs.append((away, home))
    # A,B (A is home) -> convert to (away, home)
    for m in re.finditer(r"([A-Za-z0-9&.\' \-]+?)\s*,\s*([A-Za-z0-9&.\' \-]+)", txt):
        home = m.group(1).strip(); away = m.group(2).strip()
        pairs.append((away, home))
    # Dedup preserving order
    seen = set(); out = []
    for a,h in pairs:
        key = (a.lower(), h.lower())
        if key not in seen:
            seen.add(key); out.append((a,h))
    return out

def _load_games_list(path: Path) -> List[Tuple[str, str]]:
    if not path.exists():
        return []
    return _extract_pairs_from_text(path.read_text())

def _score_match(away_req: List[str], home_req: List[str], away_row: List[str], home_row: List[str]) -> float:
    """
    Simple token overlap score for (away,home) orientation.
    Score is average Jaccard(sim_away, sim_home).
    """
    def jacc(a,b):
        A,B = set(a), set(b)
        return 0.0 if not A or not B else len(A&B)/len(A|B)
    return 0.5 * (jacc(away_req, away_row) + jacc(home_req, home_row))

def _best_row_for_pair(pool: pd.DataFrame, away: str, home: str) -> Optional[pd.Series]:
    """Pick best schedule row for a requested (away,home) by token overlap; consider both orientations."""
    away_req = _tokens(away)
    home_req = _tokens(home)
    if pool.empty or not away_req or not home_req:
        return None

    # Pre-filter: rows where each side shares at least one token with one side of request
    candidates = pool[
        pool["home_toks"].apply(lambda t: bool(set(t)&set(home_req)) or bool(set(t)&set(away_req))) &
        pool["away_toks"].apply(lambda t: bool(set(t)&set(home_req)) or bool(set(t)&set(away_req)))
    ]
    if candidates.empty:
        return None

    best = None
    best_score = -1.0
    for _, row in candidates.iterrows():
        sc1 = _score_match(away_req, home_req, row["away_toks"], row["home_toks"])  # as listed
        sc2 = _score_match(away_req, home_req, row["home_toks"], row["away_toks"])  # swapped
        sc  = max(sc1, sc2)
        if sc > best_score:
            best_score = sc
            best = row

    # require a minimum confidence to avoid bad picks
    return best if best_score >= 0.40 else None  # tweakable threshold

def _match_games_with_week_fallback(sched_all: pd.DataFrame, season: int, desired_week: int,
                                    requested_pairs: List[Tuple[str,str]]) -> Tuple[pd.DataFrame, int, dict]:
    """
    Try requested week first (token overlap), then +/-1, then ANY week in season with best-per-pair.
    Returns (rows, matched_week, debug).
    """
    debug = {"season": season, "requested_week": desired_week, "tried": [], "pairs": requested_pairs}

    pool_season = sched_all[sched_all["season"] == season].copy()
    tried_weeks = [desired_week, desired_week-1, desired_week+1]

    def try_week(wk: int) -> pd.DataFrame:
        cand = pool_season[pool_season["week"] == wk].copy()
        picked = []
        for away, home in requested_pairs:
            row = _best_row_for_pair(cand, away, home)
            if row is not None:
                picked.append(row)
        return pd.concat(picked, axis=1).T if picked else cand.iloc[0:0].copy()

    # 1) exact, then +-1
    for wk in tried_weeks:
        if wk < 0: continue
        rows = try_week(wk)
        debug["tried"].append({"week": wk, "matched": int(len(rows))})
        if not rows.empty and len(rows) >= max(1, len(requested_pairs)//2):
            return _select_cols(rows, DESIRED_SCHED_COLS).drop_duplicates(), wk, debug

    # 2) ANY week: pick best per pair across the whole season
    picked = []
    for away, home in requested_pairs:
        row = _best_row_for_pair(pool_season, away, home)
        if row is not None:
            picked.append(row)
    rows_any = pd.concat(picked, axis=1).T if picked else pool_season.iloc[0:0].copy()
    debug["tried"].append({"week": "ANY", "matched": int(len(rows_any))})

    # Choose a representative week (most frequent among found rows), else desired_week
    if not rows_any.empty:
        wk = int(rows_any["week"].mode().iloc[0]) if "week" in rows_any.columns and not rows_any["week"].isna().all() else desired_week
        return _select_cols(rows_any, DESIRED_SCHED_COLS).drop_duplicates(), wk, debug

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
                gid_col = c; break
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
    ap.add_argument("--week", type=int, required=True, help="CFB week to predict (e.g., 5)")
    ap.add_argument("--season", type=int, default=None, help="Override season (default: latest season in schedule)")
    args = ap.parse_args()

    # model + meta
    if not MODEL_FILE.exists():
        print(f"ERROR: {MODEL_FILE} not found. Run scripts.train_model first.", file=sys.stderr); sys.exit(2)
    model = joblib_load(MODEL_FILE)
    if not META_JSON.exists():
        print(f"ERROR: {META_JSON} not found. Run scripts.build_dataset first.", file=sys.stderr); sys.exit(2)
    meta = json.loads(META_JSON.read_text())
    feature_names: list[str] = meta.get("features", [])

    # schedule
    sched_all = _read_csv(SCHED_CSV)
    if sched_all.empty:
        print("ERROR: schedule CSV not found or empty.", file=sys.stderr); sys.exit(2)
    sched_all = _prep_schedule(sched_all)

    season = args.season or int(sched_all["season"].max())
    week   = int(args.week)
    print(f"Predicting season={season}, week={week}")

    # input pairs (away, home)
    requested_pairs = _load_games_list(GAMES_TXT)
    if not requested_pairs:
        print(f"NOTE: {GAMES_TXT} empty/unparsable.")
        # If truly empty, try predicting all games of that week:
        requested_pairs = [(a,b) for _,a,b in sched_all[(sched_all["season"]==season)&(sched_all["week"]==week)][["away_team","home_team"]].itertuples(index=False)]
    # robust match
    pred_rows, matched_week, dbg = _match_games_with_week_fallback(sched_all, season, week, requested_pairs)
    dbg["matched_week"] = matched_week
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
    # align features
    for c in feature_names:
        if c not in Xp.columns: Xp[c] = 0.0
    base_cols = ["game_id","home_team","away_team"]
    if "neutral_site" in Xp.columns:
        base_cols.append("neutral_site")
    else:
        Xp["neutral_site"] = False
        base_cols.append("neutral_site")
    Xp = Xp[base_cols + feature_names].copy()

    # predict
    probs = model.predict_proba(Xp[feature_names])[:, 1]  # P(home)
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
