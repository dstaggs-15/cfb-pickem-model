# scripts/lib/hypo.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import re

CHUNKSIZE = 200_000

# ---------- helpers ----------
def _ci_lookup(cols: List[str], candidates: List[str]) -> Optional[str]:
    """Case-insensitive column resolver, ignoring underscores/spaces."""
    def norm(s: str) -> str:
        return re.sub(r"[_\s]+", "", s.strip().lower())
    want = {norm(c) for c in candidates}
    for c in cols:
        if norm(c) in want:
            return c
    return None

def _z(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    mu = series.mean()
    sd = series.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return series * 0.0
    return (series - mu) / sd

def _canon_team(s: str) -> str:
    if not isinstance(s, str): return ""
    return re.sub(r"\s+", " ", s.strip())

# ---------- load & pivot team-game stats ----------
def load_team_game_stats(stats_csv: Path) -> tuple[pd.DataFrame, dict] | tuple[(), dict]:
    """
    Load CFBD team-game stats flexibly; pivot to wide per game per team.
    Looks for columns like: game_id, season, team, stat, value.
    Returns (wide_df, meta) or ((), {}) if not loadable.
    """
    if not stats_csv.exists():
        return (), {}

    # Try to map columns from header
    header = pd.read_csv(stats_csv, nrows=0)
    cols = list(header.columns)

    c_game   = _ci_lookup(cols, ["game_id", "gameid"])
    c_season = _ci_lookup(cols, ["season", "year"])
    c_team   = _ci_lookup(cols, ["team", "school"])
    c_stat   = _ci_lookup(cols, ["stat_name", "statname", "category", "stat"])
    c_value  = _ci_lookup(cols, ["stat_value", "statvalue", "value"])

    usecols = [x for x in [c_game, c_season, c_team, c_stat, c_value] if x]
    if not usecols or (c_game is None or c_season is None or c_team is None or c_stat is None or c_value is None):
        # fallback: load full and pray
        df = pd.read_csv(stats_csv, low_memory=False)
    else:
        df = pd.read_csv(stats_csv, usecols=usecols, low_memory=False)

    # Canonical names
    ren = {}
    if c_game:   ren[c_game] = "game_id"
    if c_season: ren[c_season] = "season"
    if c_team:   ren[c_team] = "team"
    if c_stat:   ren[c_stat] = "stat"
    if c_value:  ren[c_value] = "value"
    df = df.rename(columns=ren)

    for c in ["game_id", "team", "stat"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    if "season" in df.columns:
        df["season"] = pd.to_numeric(df["season"], errors="coerce")
    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

    need = {"game_id","season","team","stat","value"}
    if not need.issubset(df.columns):
        return (), {}

    # keep frequent stats only (cuts noise/misc categories)
    frequent = df["stat"].value_counts()
    frequent = frequent[frequent > 100].index.tolist()
    df = df[df["stat"].isin(frequent)].copy()

    wide = df.pivot_table(index=["game_id","season","team"], columns="stat", values="value", aggfunc="first")
    wide = wide.reset_index()
    for c in ["team"]:
        if c in wide.columns:
            wide[c] = wide[c].map(_canon_team)

    # Try to locate some common offense metrics (names vary across dumps)
    def col_like(options: List[str]) -> Optional[str]:
        for opt in options:
            pat = re.sub(r"[^a-z]", "", opt.lower())
            for c in wide.columns:
                if re.sub(r"[^a-z]", "", c.lower()) == pat:
                    return c
        return None

    meta = {
        "pts_off":  col_like(["points", "points_for", "score", "pointsfor"]),
        "ypp_off":  col_like(["yards_per_play", "ypp", "off_ypp", "offense_ypp"]),
        "sr_off":   col_like(["success_rate", "off_success_rate", "succ_rate_off"]),
        # You can extend with defensive columns when present (e.g., opp_points, ypp_def, sr_def)
    }
    return wide, meta

# ---------- aggregate to season level ----------
def season_aggregates(wide: pd.DataFrame, meta: dict,
                      lo: int, hi: int) -> pd.DataFrame:
    """
    Average per-team per-season stats for seasons in [lo, hi].
    Returns one row per (season, team) with 'games' count.
    """
    df = wide[(wide["season"] >= lo) & (wide["season"] <= hi)].copy()
    df["team"] = df["team"].map(_canon_team)

    stat_cols = [c for c in [meta.get("pts_off"), meta.get("ypp_off"), meta.get("sr_off")] if c and c in df.columns]
    if not stat_cols:
        # At least return games played for weighting if columns missing
        gp = df.groupby(["season","team"], as_index=False).size().rename(columns={"size":"games"})
        return gp

    agg = df.groupby(["season","team"], as_index=False)[stat_cols].mean()
    gp  = df.groupby(["season","team"], as_index=False).size().rename(columns={"size":"games"})
    out = agg.merge(gp, on=["season","team"], how="left")
    return out

def fuse_current_with_baseline(cur: pd.DataFrame, base: pd.DataFrame, season: int) -> pd.DataFrame:
    """
    Blend current-season means with multi-year baseline per team.
    Weight current by games/(games+3) to stabilize early season.
    """
    cur = cur[cur["season"] == season].copy()
    base = base[base["season"] < season].copy()

    if not base.empty:
        base_team = base.groupby("team", as_index=False).mean(numeric_only=True)
        base_team = base_team.rename(columns={c: f"base_{c}" for c in base_team.columns if c != "team"})
    else:
        base_team = pd.DataFrame(columns=["team"])

    cur = cur.set_index("team")
    base_team = base_team.set_index("team") if "team" in base_team.columns else base_team

    teams = sorted(set(cur.index) | set(base_team.index))
    rows = []
    for t in teams:
        c = cur.loc[t] if t in cur.index else None
        b = base_team.loc[t] if (not base_team.empty and t in base_team.index) else None

        rec: Dict[str, float | str] = {"team": t}
        gp = float(c["games"]) if (c is not None and "games" in c) else 0.0
        w  = gp / (gp + 3.0) if (gp + 3.0) > 0 else 0.0

        stat_cols = [col for col in (list(c.index) if c is not None else []) if col not in ["games"]]
        for col in stat_cols:
            cur_val  = float(c[col]) if (c is not None and pd.notna(c[col])) else np.nan
            base_val = float(b.get(f"base_{col}", np.nan)) if b is not None else np.nan
            if pd.notna(cur_val) and pd.notna(base_val):
                val = w * cur_val + (1 - w) * base_val
            elif pd.notna(cur_val):
                val = cur_val
            elif pd.notna(base_val):
                val = base_val
            else:
                val = np.nan
            rec[col] = val

        rec["games"] = gp
        rows.append(rec)

    out = pd.DataFrame(rows).fillna(0.0)
    return out

def team_strength_table(stats_csv: Path, season: int, years_back: int = 20) -> pd.DataFrame:
    """
    Build a per-team strength table for `season` by:
      1) loading team-game stats (wide)
      2) aggregating current season and a multi-year baseline (last 20 seasons)
      3) fusing the two
      4) standardizing features and computing a composite 'rating'
    """
    loaded = load_team_game_stats(stats_csv)
    if not loaded:
        return pd.DataFrame()
    wide, meta = loaded

    cur  = season_aggregates(wide, meta, lo=season, hi=season)
    base = season_aggregates(wide, meta, lo=max(int(wide["season"].min()), season - years_back), hi=season - 1)

    fused = fuse_current_with_baseline(cur, base, season=season)

    # Standardize numeric cols and compute a composite rating
    num_cols = [c for c in fused.columns if c not in ["team","games","season"]]
    Z = fused.copy()
    for c in num_cols:
        Z[c] = _z(Z[c])

    # Offense-only proxy available universally; average the standardized features
    if num_cols:
        Z["rating"] = Z[num_cols].mean(axis=1)
    else:
        Z["rating"] = 0.0

    return Z[["team","rating","games"]].copy()

# ---------- name resolution for user inputs ----------
_STOP = {"THE","OF","UNIVERSITY","UNIV","U","STATE","ST","&","AND","AT"}
def _canon_for_match(s: str) -> List[str]:
    s = s.upper()
    s = s.replace("&", " AND ").replace("A&M", "A AND M")
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
    toks = re.split(r"[^A-Z0-9]+", s)
    return [t for t in toks if t and t not in _STOP]

def _resolve_name(name: str, candidates: List[str]) -> tuple[str, float]:
    rq = set(_canon_for_match(name))
    best = ("", 0.0)
    for cand in candidates:
        cj = set(_canon_for_match(cand))
        if not cj and not rq:
            continue
        j = (len(rq & cj) / len(rq | cj)) if (rq or cj) else 0.0
        if j > best[1]:
            best = (cand, j)
    return best

def resolve_pairs_against_strength(pairs: List[tuple[str,str]], strength: pd.DataFrame) -> list[dict]:
    """
    For each (away, home) input, fuzzy-resolve names to the strength table 'team' values.
    """
    teams = strength["team"].astype(str).tolist()
    out = []
    for away_in, home_in in pairs:
        home_name, sc_h = _resolve_name(home_in, teams)
        away_name, sc_a = _resolve_name(away_in, teams)
        out.append({
            "input": {"away": away_in, "home": home_in},
            "resolved": {"away": away_name, "home": home_name},
            "scores": {"away": round(sc_a,3), "home": round(sc_h,3)}
        })
    return out
