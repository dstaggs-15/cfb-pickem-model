# scripts/predict.py

from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
from typing import Iterable, List, Tuple, Optional

import numpy as np
import pandas as pd
from joblib import load as joblib_load

from .lib.features import create_feature_set
from .lib.parsing import ensure_schedule_columns
from .lib import hypo  # used for fallback "hypothetical" mode

# --- Paths ---
DERIVED_DIR = Path("data/derived")
RAW_DIR     = Path("data/raw/cfbd")
DOCS_DATA   = Path("docs/data")
INPUT_DIR   = Path("docs/input")

SCHED_CSV   = RAW_DIR / "cfb_schedule.csv"
STATS_CSV   = RAW_DIR / "cfb_game_team_stats.csv"
LINES_CSV   = RAW_DIR / "cfb_lines.csv"
TEAMS_CSV   = RAW_DIR / "cfbd_teams.csv"
VENUES_CSV  = RAW_DIR / "cfbd_venues.csv"
TALENT_CSV  = RAW_DIR / "cfbd_talent.csv"

PRED_JSON   = DOCS_DATA / "predictions.json"
META_JSON   = DOCS_DATA / "train_meta.json"
DEBUG_JSON  = DOCS_DATA / "debug_predict.json"

GAMES_TXT   = INPUT_DIR / "games.txt"
MANUAL_LINES= INPUT_DIR / "manual_lines.csv"
MODEL_FILE  = DERIVED_DIR / "model.joblib"

CHUNKSIZE   = 200_000
DESIRED_SCHED_COLS = [
    "game_id","season","week","date","home_team","away_team","neutral_site",
    "home_points","away_points","venue_id","venue"
]

STOP = {"THE","OF","UNIVERSITY","UNIV","U","STATE","ST","&","AND","AT"}


# -------------------- Helpers --------------------

def _canon(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.upper()
    s = s.replace("&", " AND ").replace("A&M", "A AND M").replace("A & M", "A AND M")
    s = s.replace(".", " ").replace("'", " ")
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
    df["home_toks"] = df["home_team"].map(_tokens)
    df["away_toks"] = df["away_team"].map(_tokens)
    return df


def _extract_pairs_from_text(txt: str) -> List[Tuple[str,str]]:
    pairs: List[Tuple[str,str]] = []
    # "Away @ Home"
    for m in re.finditer(r"([A-Za-z0-9&.\' \-]+?)\s*@\s*([A-Za-z0-9&.\' \-]+)", txt):
        pairs.append((m.group(1).strip(), m.group(2).strip()))  # (away, home)
    # "TeamA vs TeamB" -> treat as (away, home) = (TeamB, TeamA)
    for m in re.finditer(r"([A-Za-z0-9&.\' \-]+?)\s*vs\s*([A-Za-z0-9&.\' \-]+)", txt, flags=re.IGNORECASE):
        pairs.append((m.group(2).strip(), m.group(1).strip()))
    # "TeamA, TeamB" -> (away, home) = (TeamB, TeamA)
    for m in re.finditer(r"([A-Za-z0-9&.\' \-]+?)\s*,\s*([A-Za-z0-9&.\' \-]+)", txt):
        pairs.append((m.group(2).strip(), m.group(1).strip()))
    # dedupe (case-insensitive)
    seen=set(); out=[]
    for a,h in pairs:
        key=(a.lower(),h.lower())
        if key not in seen:
            seen.add(key); out.append((a,h))
    return out


def _load_games_list(path: Path) -> List[Tuple[str, str]]:
    if not path.exists(): return []
    return _extract_pairs_from_text(path.read_text())


def _score_match(away_req: List[str], home_req: List[str],
                 away_row: List[str], home_row: List[str]) -> float:
    def jacc(a,b):
        A,B=set(a),set(b)
        return 0.0 if not A or not B else len(A&B)/len(A|B)
    return 0.5*(jacc(away_req,away_row)+jacc(home_req,home_row))


# >>>>>>>>>>>>>>>>>> FIX APPLIED HERE <<<<<<<<<<<<<<<<<<
def _best_row_for_pair(pool: pd.DataFrame, away: str, home: str) -> Optional[pd.Series]:
    """
    Find the best schedule row for (away, home). If the best score hits only when
    the row is interpreted with swapped teams, return a copy of the row with
    home/away (and their tokens/points) flipped so downstream logic is correct.
    """
    away_req = _tokens(away)
    home_req = _tokens(home)
    if pool.empty or not away_req or not home_req:
        return None

    cand = pool[
        pool["home_toks"].apply(lambda t: bool(set(t)&set(home_req)) or bool(set(t)&set(away_req))) &
        pool["away_toks"].apply(lambda t: bool(set(t)&set(home_req)) or bool(set(t)&set(away_req)))
    ]
    if cand.empty:
        return None

    best = None
    best_score = -1.0
    best_flip = False

    for _, row in cand.iterrows():
        sc1 = _score_match(away_req, home_req, row["away_toks"], row["home_toks"])  # (away->away, home->home)
        sc2 = _score_match(away_req, home_req, row["home_toks"], row["away_toks"])  # swapped
        if sc1 >= sc2:
            sc = sc1
            flip = False
        else:
            sc = sc2
            flip = True
        if sc > best_score:
            best_score = sc
            best = row
            best_flip = flip

    if best is None or best_score < 0.40:
        return None

    if best_flip:
        # Return a flipped copy so "away @ home" matches input orientation
        r = best.copy()
        if "home_team" in r and "away_team" in r:
            r["home_team"], r["away_team"] = r["away_team"], r["home_team"]
        if "home_points" in r and "away_points" in r:
            r["home_points"], r["away_points"] = r["away_points"], r["home_points"]
        if "home_toks" in r and "away_toks" in r:
            r["home_toks"], r["away_toks"] = r["away_toks"], r["home_toks"]
        return r

    return best
# >>>>>>>>>>>>>>>>> END FIX <<<<<<<<<<<<<<<<<<<<<<<<<<<<


def _stream_filter_by_gids(csv_path: Path, gids: set[str]) -> pd.DataFrame:
    if not csv_path.exists() or not gids:
        return pd.DataFrame()
    header = pd.read_csv(csv_path, nrows=0)
    cols = list(header.columns)
    gid_col = None
    for c in cols:
        if c.lower() in ("game_id","gameid"):
            gid_col = c; break
    if gid_col is None:
        return pd.DataFrame()

    keep=[]
    for chunk in pd.read_csv(csv_path, chunksize=CHUNKSIZE, low_memory=False):
        chunk[gid_col]=chunk[gid_col].astype(str)
        piece = chunk[chunk[gid_col].isin(gids)]
        if not piece.empty:
            keep.append(piece)
    return pd.concat(keep, ignore_index=True) if keep else pd.DataFrame()


# -------------------- PREDICT --------------------

def _normal_mode_predict(pairs: List[tuple[str,str]], season: int, week: int) -> tuple[list, dict]:
    """Use the trained model with full features IF schedule rows can be matched."""
    if not MODEL_FILE.exists() or not META_JSON.exists():
        return [], {"mode":"NORMAL","error":"missing_model_or_meta"}
    if not SCHED_CSV.exists():
        return [], {"mode":"NORMAL","error":"missing_schedule_csv"}

    model = joblib_load(MODEL_FILE)
    meta = json.loads(META_JSON.read_text())
    features: list[str] = meta.get("features", [])

    sched_all = _read_csv(SCHED_CSV)
    if sched_all.empty:
        return [], {"mode":"NORMAL","error":"empty_schedule"}
    sched_all = _prep_schedule(sched_all)

    pool = sched_all[sched_all["season"]==season].copy()

    def try_week(wk: int) -> pd.DataFrame:
        cand = pool[pool["week"]==wk].copy()
        picked=[]
        for away,home in pairs:
            r=_best_row_for_pair(cand,away,home)
            if r is not None:
                picked.append(r)
        return pd.concat(picked,axis=1).T if picked else cand.iloc[0:0].copy()

    tried=[]
    pred_rows = try_week(week); tried.append({"week":week,"matched":int(len(pred_rows))})
    if pred_rows.empty:
        for wk in (week-1, week+1):
            if wk>=0:
                r=try_week(wk); tried.append({"week":wk,"matched":int(len(r))})
                if not r.empty:
                    pred_rows=r; week=wk; break
    if pred_rows.empty:
        picked=[]
        for away,home in pairs:
            r=_best_row_for_pair(pool,away,home)
            if r is not None:
                picked.append(r)
        pred_rows = pd.concat(picked,axis=1).T if picked else pool.iloc[0:0].copy()
        tried.append({"week":"ANY","matched":int(len(pred_rows))})

    if pred_rows.empty:
        return [], {"mode":"NORMAL","tried":tried,"matched":0}

    gids=set(pred_rows["game_id"].astype(str).unique())

    teams_df  = _read_csv(TEAMS_CSV)
    venues_df = _read_csv(VENUES_CSV)
    talent_df = _read_csv(TALENT_CSV)
    stats_chunk = _stream_filter_by_gids(STATS_CSV, gids)
    lines_chunk = _stream_filter_by_gids(LINES_CSV, gids)

    X_all, feat_list = create_feature_set(
        schedule=sched_all[sched_all["game_id"].isin(gids)].copy(),
        team_stats=stats_chunk,
        venues_df=venues_df,
        teams_df=teams_df,
        talent_df=talent_df,
        lines_df=lines_chunk,
        manual_lines_df=_read_csv(MANUAL_LINES) if MANUAL_LINES.exists() else None,
        games_to_predict_df=pred_rows[DESIRED_SCHED_COLS]
    )

    Xp = X_all[X_all["game_id"].isin(gids)].copy()
    for c in features:
        if c not in Xp.columns:
            Xp[c]=0.0
    if "neutral_site" not in Xp.columns:
        Xp["neutral_site"]=False

    base=["game_id","home_team","away_team","neutral_site"]
    Xp = Xp[[c for c in base+features if c in Xp.columns]].copy()

    probs = model.predict_proba(Xp[features])[:,1]

    out=[]
    for (gid,home,away,ns,p_home) in zip(
        Xp["game_id"],Xp["home_team"],Xp["away_team"],Xp["neutral_site"],probs
    ):
        out.append({
            "id": str(gid),               # compat for UI
            "season": int(season),
            "week": int(week),
            "home_team": str(home),
            "away_team": str(away),
            "neutral_site": bool(ns),
            "model_prob_home": float(round(p_home,4)),
            "prob_home": float(round(p_home,4)),    # compat alias
            "prob_away": float(round(1.0-p_home,4)),# compat alias
            "pick": str(home if p_home>=0.5 else away),
            "explanation": []
        })

    dbg={
        "mode":"NORMAL",
        "season":season,
        "matched_week":week,
        "tried":tried,
        "matched_preview": pred_rows[["home_team","away_team","season","week"]].to_dict(orient="records")
    }
    return out, dbg


def _hypothetical_stats_mode(pairs: List[tuple[str,str]], season: int) -> tuple[list, dict]:
    """
    Stats-first hypothetical predictions:
    - Build team strength from current season blended with last ~20 seasons
    - Logistic on rating diff + small home edge -> P(home)
    """
    strength = hypo.team_strength_table(STATS_CSV, season=season, years_back=20)
    if strength.empty:
        # No stats at all? hard fail with empty output
         # No stats available; produce fallback predictions using default home edge
    out = []
    for away, home in pairs:
        diff = 0.15  # default home edge
        p_home = float(1 / (1 + np.exp(-diff)))
        out.append({
            "id": f"NA_{home}_{away}",
            "season": int(season),
            "week": -1,
            "home_team": str(home),
            "away_team": str(away),
            "neutral_site": False,
            "model_prob_home": round(p_home, 4),
            "prob_home": round(p_home, 4),
            "prob_away": round(1.0 - p_home, 4),
            "pick": str(home if p_home >= 0.5 else away),
            "explanation": []
        })
       return out, {"mode": "HYPOTHETICAL", "error": "no_stats_loaded", "fallback": True}
 # Resolve names
    resolved = hypo.resolve_pairs_against_strength(pairs, strength)
    rating_map = strength.set_index("team")["rating"].to_dict()

    out=[]
    for entry in resolved:
        home = entry["resolved"]["home"]
        away = entry["resolved"]["away"]
        rh = float(rating_map.get(home, 0.0))
        ra = float(rating_map.get(away, 0.0))
        home_edge = 0.15
        diff = (rh - ra) + home_edge
       
