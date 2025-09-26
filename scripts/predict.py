# scripts/predict.py
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable, List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from joblib import load as joblib_load

from .lib.features import create_feature_set
from .lib.parsing import ensure_schedule_columns

# Paths
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

STOP = {"THE","OF","UNIVERSITY","UNIV","U","STATE","ST","&","AND","AT"}

def _canon(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.upper()
    s = s.replace("&", " AND ")
    s = s.replace("A&M", "A AND M").replace("A & M", "A AND M")
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
    # canonical tokens for fuzzy
    df["home_toks"] = df["home_team"].map(_tokens)
    df["away_toks"] = df["away_team"].map(_tokens)
    return df

def _select_cols(df: pd.DataFrame, desired: List[str]) -> pd.DataFrame:
    present = [c for c in desired if c in df.columns]
    return df[present]

def _extract_pairs_from_text(txt: str) -> List[Tuple[str,str]]:
    pairs: List[Tuple[str,str]] = []
    # A @ B (B home)
    for m in re.finditer(r"([A-Za-z0-9&.\' \-]+?)\s*@\s*([A-Za-z0-9&.\' \-]+)", txt):
        away = m.group(1).strip(); home = m.group(2).strip()
        pairs.append((away, home))
    # A vs B (A home) -> (away, home)
    for m in re.finditer(r"([A-Za-z0-9&.\' \-]+?)\s*vs\s*([A-Za-z0-9&.\' \-]+)", txt, flags=re.IGNORECASE):
        home = m.group(1).strip(); away = m.group(2).strip()
        pairs.append((away, home))
    # A,B (A home) -> (away, home)
    for m in re.finditer(r"([A-Za-z0-9&.\' \-]+?)\s*,\s*([A-Za-z0-9&.\' \-]+)", txt):
        home = m.group(1).strip(); away = m.group(2).strip()
        pairs.append((away, home))
    # Dedup preserving order
    seen = set(); out=[]
    for a,h in pairs:
        key=(a.lower(),h.lower())
        if key not in seen:
            seen.add(key); out.append((a,h))
    return out

def _load_games_list(path: Path) -> List[Tuple[str, str]]:
    if not path.exists():
        return []
    return _extract_pairs_from_text(path.read_text())

def _score_match(away_req: List[str], home_req: List[str], away_row: List[str], home_row: List[str]) -> float:
    def jacc(a,b):
        A,B = set(a), set(b)
        return 0.0 if not A or not B else len(A&B)/len(A|B)
    return 0.5 * (jacc(away_req, away_row) + jacc(home_req, home_row))

def _best_row_for_pair(pool: pd.DataFrame, away: str, home: str) -> Optional[pd.Series]:
    away_req = _tokens(away); home_req = _tokens(home)
    if pool.empty or not away_req or not home_req:
        return None
    candidates = pool[
        pool["home_toks"].apply(lambda t: bool(set(t)&set(home_req)) or bool(set(t)&set(away_req))) &
        pool["away_toks"].apply(lambda t: bool(set(t)&set(home_req)) or bool(set(t)&set(away_req)))
    ]
    if candidates.empty:
        return None
    best = None; best_score = -1.0
    for _, row in candidates.iterrows():
        sc1 = _score_match(away_req, home_req, row["away_toks"], row["home_toks"])
        sc2 = _score_match(away_req, home_req, row["home_toks"], row["away_toks"])
        sc  = max(sc1, sc2)
        if sc > best_score:
            best_score = sc; best = row
    return best if best_score >= 0.40 else None

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
        piece = chunk[gid_col].isin(gids)
        piece = chunk[piece]
        if not piece.empty: keep.append(piece)
    return pd.concat(keep, ignore_index=True) if keep else pd.DataFrame()

# ---------- HYPOTHETICAL MODE (no schedule match) ----------
def _build_team_index(teams_df: pd.DataFrame, talent_df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Build a lookup map of team display name -> data, and a token index for fuzzy matching.
    """
    names = set()
    if not teams_df.empty:
        for col in ("school","team","name"):
            if col in teams_df.columns:
                names.update(teams_df[col].dropna().astype(str).tolist())
    if not talent_df.empty:
        col = "team" if "team" in talent_df.columns else ("school" if "school" in talent_df.columns else None)
        if col:
            names.update(talent_df[col].dropna().astype(str).tolist())
    names = sorted({n.strip() for n in names if n and n.strip()})
    # map to talent
    tmap = {}
    if not talent_df.empty:
        tcol = "team" if "team" in talent_df.columns else ("school" if "school" in talent_df.columns else None)
        if tcol and "talent" in talent_df.columns:
            for _, r in talent_df[[tcol,"talent"]].dropna().iterrows():
                tmap[str(r[tcol]).strip()] = float(r["talent"])
    # index
    idx = {}
    for n in names:
        idx[n] = {
            "tokens": set(_tokens(n)),
            "talent": tmap.get(n, np.nan)
        }
    return idx

def _resolve_team(name: str, index: Dict[str, Dict]) -> Tuple[str, float]:
    """
    Resolve a user-supplied team name to the closest known team using token Jaccard.
    Returns (resolved_name, score). Score in [0,1].
    """
    rq = set(_tokens(name))
    if not rq:
        return name, 0.0
    best = ("", 0.0)
    for n, meta in index.items():
        cand = meta["tokens"]
        if not cand: continue
        j = len(rq & cand) / len(rq | cand)
        if j > best[1]:
            best = (n, j)
    return best

def _hypo_predict(pairs: List[Tuple[str,str]], teams_df: pd.DataFrame, talent_df: pd.DataFrame) -> Tuple[list, dict]:
    """
    Produce predictions without schedule:
      - Resolve team names
      - Build a simple rating from talent (z-score)
      - Add small home edge
      - Logistic to get P(home)
    """
    dbg = {"mode": "HYPOTHETICAL", "resolved": []}

    index = _build_team_index(teams_df, talent_df)

    # Talent series
    tcol = "team" if "team" in talent_df.columns else ("school" if "school" in talent_df.columns else None)
    talents = None
    if tcol and "talent" in talent_df.columns and not talent_df.empty:
        s = talent_df[[tcol, "talent"]].dropna()
        s = s.groupby(tcol, as_index=True)["talent"].mean()
        talents = s

    # z-score
    def talent_rating(team_name: str) -> float:
        if talents is None: return 0.0
        if team_name not in talents.index: return 0.0
        val = float(talents.loc[team_name])
        # zscore over available talents
        mu = float(talents.mean())
        sd = float(talents.std(ddof=0)) or 1.0
        return (val - mu) / sd

    out = []
    for away_raw, home_raw in pairs:
        home_name, sc_h = _resolve_team(home_raw, index)
        away_name, sc_a = _resolve_team(away_raw, index)

        r_home = talent_rating(home_name)
        r_away = talent_rating(away_name)
        home_edge = 0.15  # small edge; tune if you want

        diff = (r_home - r_away) + home_edge
        p_home = 1.0 / (1.0 + np.exp(-1.50 * diff))  # steeper logistic

        pick = home_name if p_home >= 0.5 else away_name

        out.append({
            "home_team": str(home_name),
            "away_team": str(away_name),
            "neutral_site": False,
            "model_prob_home": float(round(p_home, 4)),
            "pick": pick,
            "explanation": [
                "hypothetical_mode",
                f"talent_z_home={round(r_home,3)}",
                f"talent_z_away={round(r_away,3)}",
                f"home_edge={home_edge}"
            ]
        })
        dbg["resolved"].append({
            "input": {"away": away_raw, "home": home_raw},
            "resolved": {"away": away_name, "home": home_name},
            "resolve_score": {"away": round(sc_a,3), "home": round(sc_h,3)},
            "talent_z": {"home": round(r_home,3), "away": round(r_away,3)}
        })

    return out, dbg

# ---------- Main flow ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--season", type=int, default=None)
    args = ap.parse_args()

    # Load inputs file
    pairs = _load_games_list(GAMES_TXT)

    # Normal artifacts for the standard path
    sched_all = _read_csv(SCHED_CSV)
    sched_all = _prep_schedule(sched_all) if not sched_all.empty else sched_all

    # Try normal model path only if we have schedule and model+meta
    can_do_full = (
        (not sched_all.empty) and
        MODEL_FILE.exists() and
        META_JSON.exists()
    )

    # If we CAN run the normal path, try it first
    if can_do_full:
        try:
            model = joblib_load(MODEL_FILE)
            meta = json.loads(META_JSON.read_text())
            feature_names: list[str] = meta.get("features", [])

            season = args.season or int(sched_all["season"].max())
            week   = int(args.week)
            print(f"Predicting season={season}, week={week}")

            # Build candidate rows from schedule using token matcher
            pool_season = sched_all[sched_all["season"] == season].copy()

            def try_week(wk: int) -> pd.DataFrame:
                cand = pool_season[pool_season["week"] == wk].copy()
                picked = []
                for away, home in pairs:
                    row = _best_row_for_pair(cand, away, home)
                    if row is not None:
                        picked.append(row)
                return pd.concat(picked, axis=1).T if picked else cand.iloc[0:0].copy()

            tried = []
            pred_rows = try_week(week)
            tried.append({"week": week, "matched": int(len(pred_rows))})
            if pred_rows.empty:
                for wk in (week-1, week+1):
                    if wk >= 0:
                        r = try_week(wk)
                        tried.append({"week": wk, "matched": int(len(r))})
                        if not r.empty:
                            pred_rows = r
                            week = wk
                            break
            if pred_rows.empty:
                # any week
                picked = []
                for away, home in pairs:
                    row = _best_row_for_pair(pool_season, away, home)
                    if row is not None:
                        picked.append(row)
                pred_rows = pd.concat(picked, axis=1).T if picked else pool_season.iloc[0:0].copy()
                tried.append({"week": "ANY", "matched": int(len(pred_rows))})

            if not pred_rows.empty:
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
                    games_to_predict_df=_select_cols(pred_rows, DESIRED_SCHED_COLS)
                )

                Xp = X_all[X_all["game_id"].isin(gids)].copy()
                for c in feature_names:
                    if c not in Xp.columns: Xp[c] = 0.0
                if "neutral_site" not in Xp.columns: Xp["neutral_site"] = False
                base = ["game_id","home_team","away_team","neutral_site"]
                usecols = [c for c in base + feature_names if c in Xp.columns]
                Xp = Xp[usecols].copy()

                probs = model.predict_proba(Xp[feature_names])[:, 1]
                out = []
                for (home, away, ns, p_home) in zip(Xp["home_team"], Xp["away_team"], Xp["neutral_site"], probs):
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
                with open(DEBUG_JSON, "w") as f: json.dump({
                    "mode": "NORMAL",
                    "season": season,
                    "matched_week": week,
                    "tried": tried,
                    "matched_preview": _select_cols(pred_rows, ["home_team","away_team","season","week"]).to_dict(orient="records")
                }, f, indent=2)
                print(f"Wrote {len(out)} predictions to {PRED_JSON} (normal mode).")
                return
            else:
                print("Normal mode found 0 games; falling back to hypothetical mode.")

        except Exception as e:
            print(f"Normal mode failed with error: {e}. Falling back to hypothetical mode.", file=sys.stderr)

    # ---------- HYPOTHETICAL MODE ----------
    teams_df  = _read_csv(TEAMS_CSV)
    talent_df = _read_csv(TALENT_CSV)

    if not pairs:
        print(f"NOTE: {GAMES_TXT} empty/unparsable and no schedule; nothing to predict.")
        DOCS_DATA.mkdir(parents=True, exist_ok=True)
        with open(PRED_JSON, "w") as f: json.dump({"games": []}, f, indent=2)
        with open(DEBUG_JSON, "w") as f: json.dump({
            "mode": "HYPOTHETICAL",
            "error": "no_input_pairs"
        }, f, indent=2)
        return

    preds, dbg = _hypo_predict(pairs, teams_df, talent_df)
    DOCS_DATA.mkdir(parents=True, exist_ok=True)
    with open(PRED_JSON, "w") as f: json.dump({"games": preds}, f, indent=2)
    with open(DEBUG_JSON, "w") as f: json.dump(dbg, f, indent=2)
    print(f"Wrote {len(preds)} predictions to {PRED_JSON} (hypothetical mode).")

if __name__ == "__main__":
    main()
