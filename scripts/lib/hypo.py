# scripts/lib/hypo.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import re
import ast
import json

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
def _maybe_literal_eval(val):
    if isinstance(val, str):
        val = val.strip()
        if not val:
            return {}
        try:
            return ast.literal_eval(val)
        except Exception:
            try:
                return json.loads(val)
            except Exception:
                return {}
    if isinstance(val, dict):
        return val
    return {}

def _flatten_off_def(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Handle the newer CFBD format where offense/defense columns contain nested dicts."""
    need_cols = {"game_id", "season", "team", "offense", "defense"}
    if not need_cols.issubset(df.columns):
        return pd.DataFrame(), {}

    rows: list[dict] = []
    for _, row in df.iterrows():
        try:
            off = _maybe_literal_eval(row["offense"])
            deff = _maybe_literal_eval(row["defense"])
        except Exception:
            continue

        rec: dict = {
            "game_id": str(row.get("game_id", "")),
            "season": row.get("season"),
            "team": _canon_team(row.get("team", "")),
        }

        def grab(src: dict, key: str, dest: str):
            if isinstance(src, dict) and key in src:
                rec[dest] = src[key]

        grab(off, "ppa", "off_ppa")
        grab(off, "successRate", "off_sr")
        grab(off, "explosiveness", "off_explosiveness")

        rushing = off.get("rushingPlays") if isinstance(off, dict) else {}
        passing = off.get("passingPlays") if isinstance(off, dict) else {}
        grab(rushing or {}, "ppa", "rush_ppa")
        grab(passing or {}, "ppa", "pass_ppa")

        grab(deff, "ppa", "def_ppa")
        grab(deff, "successRate", "def_sr")
        grab(deff, "explosiveness", "def_explosiveness")

        rows.append(rec)

    if not rows:
        return pd.DataFrame(), {}

    wide = pd.DataFrame(rows)
    for col in ["season"]:
        if col in wide.columns:
            wide[col] = pd.to_numeric(wide[col], errors="coerce")
    for col in wide.columns:
        if col not in {"game_id", "season", "team"}:
            wide[col] = pd.to_numeric(wide[col], errors="coerce")

    meta = {
        "pts_off": "off_ppa",
        "ypp_off": "rush_ppa",
        "sr_off": "off_sr",
    }
    return wide, meta

def load_team_game_stats(stats_csv: Path) -> tuple[pd.DataFrame, dict]:
    """
    Load CFBD team-game stats flexibly; pivot to wide per game per team.
    Looks for columns like: game_id, season, team, stat, value.
    RETURNS: (wide_df, meta) — NEVER returns tuples of tuples.
    """
    if not stats_csv.exists():
        return pd.DataFrame(), {}

    try:
        header = pd.read_csv(stats_csv, nrows=0)
    except Exception:
        return pd.DataFrame(), {}

    cols = list(header.columns)
    c_game   = _ci_lookup(cols, ["game_id", "gameid"])
    c_season = _ci_lookup(cols, ["season", "year"])
    c_team   = _ci_lookup(cols, ["team", "school"])
    c_stat   = _ci_lookup(cols, ["stat_name", "statname", "category", "stat"])
    c_value  = _ci_lookup(cols, ["stat_value", "statvalue", "value"])

    try:
        if c_game and c_season and c_team and c_stat and c_value:
            usecols = [c_game, c_season, c_team, c_stat, c_value]
            df = pd.read_csv(stats_csv, usecols=usecols, low_memory=False)
        else:
            # Fallback: read everything and hope common names exist
            df = pd.read_csv(stats_csv, low_memory=False)
    except Exception:
        return pd.DataFrame(), {}

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
        wide_alt, meta_alt = _flatten_off_def(df)
        if not wide_alt.empty:
            return wide_alt, meta_alt
        return pd.DataFrame(), {}

    # Keep frequent stats to reduce noise
    vc = df["stat"].value_counts()
    keep_stats = vc[vc > 100].index.tolist()
    if keep_stats:
        df = df[df["stat"].isin(keep_stats)].copy()

    if df.empty:
        return pd.DataFrame(), {}

    try:
        wide = df.pivot_table(index=["game_id","season","team"], columns="stat", values="value", aggfunc="first")
        wide = wide.reset_index()
    except Exception:
        return pd.DataFrame(), {}

    for c in ["team"]:
        if c in wide.columns:
            wide[c] = wide[c].map(_canon_team)

    # Locate common offense metrics (names vary)
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
        # Extend with defensive metrics when available in your dump
    }
    return wide, meta

# ---------- aggregate to season level ----------
def season_aggregates(wide: pd.DataFrame, meta: dict,
                      lo: int, hi: int) -> pd.DataFrame:
    """
    Average per-team per-season stats for seasons in [lo, hi].
    Returns one row per (season, team) with 'games' count.
    """
    if not isinstance(wide, pd.DataFrame) or wide.empty:
        return pd.DataFrame()

    if "season" not in wide.columns or "team" not in wide.columns:
        return pd.DataFrame()

    df = wide[(pd.to_numeric(wide["season"], errors="coerce") >= lo) &
              (pd.to_numeric(wide["season"], errors="coerce") <= hi)].copy()
    if df.empty:
        return pd.DataFrame()

    df["team"] = df["team"].map(_canon_team)

    stat_cols = [c for c in [meta.get("pts_off"), meta.get("ypp_off"), meta.get("sr_off")] if c and c in df.columns]
    gp = df.groupby(["season","team"], as_index=False).size().rename(columns={"size":"games"})

    if not stat_cols:
        return gp  # at least games played

    agg = df.groupby(["season","team"], as_index=False)[stat_cols].mean()
    out = agg.merge(gp, on=["season","team"], how="left")
    return out

def fuse_current_with_baseline(cur: pd.DataFrame, base: pd.DataFrame, season: int) -> pd.DataFrame:
    """
    Blend current-season means with multi-year baseline per team.
    Weight current by games/(games+3) to stabilize early season.
    """
    if cur.empty and base.empty:
        return pd.DataFrame()

    cur = cur[cur.get("season", season) == season].copy() if "season" in cur.columns else pd.DataFrame()
    base = base[base.get("season", season-1) < season].copy() if "season" in base.columns else pd.DataFrame()

    if not base.empty:
        base_team = base.groupby("team", as_index=False).mean(numeric_only=True)
        base_team = base_team.rename(columns={c: f"base_{c}" for c in base_team.columns if c != "team"})
    else:
        base_team = pd.DataFrame(columns=["team"])

    cur = cur.set_index("team") if "team" in cur.columns else pd.DataFrame()
    base_team = base_team.set_index("team") if "team" in base_team.columns else pd.DataFrame()

    teams = sorted(set(cur.index) | set(base_team.index))
    rows = []
    for t in teams:
        c = cur.loc[t] if (not cur.empty and t in cur.index) else None
        b = base_team.loc[t] if (not base_team.empty and t in base_team.index) else None

        rec: Dict[str, float | str] = {"team": t}
        gp = float(c["games"]) if (c is not None and "games" in c) else 0.0
        w  = gp / (gp + 3.0) if (gp + 3.0) > 0 else 0.0

        cur_cols = [col for col in (list(c.index) if c is not None else []) if col not in ["games"]]
        base_cols = []
        if b is not None:
            base_cols = [col.removeprefix("base_") for col in b.index if col.startswith("base_")]
        stat_cols = sorted(set(cur_cols) | set(base_cols))

        for col in stat_cols:
            cur_val  = float(c[col]) if (c is not None and col in c.index and pd.notna(c[col])) else np.nan
            base_key = f"base_{col}"
            base_val = float(b.get(base_key, np.nan)) if (b is not None and base_key in b.index) else np.nan
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
    wide, meta = load_team_game_stats(stats_csv)
    if not isinstance(wide, pd.DataFrame) or wide.empty:
        return pd.DataFrame()

    # Current season aggregates
    cur  = season_aggregates(wide, meta, lo=season, hi=season)
    # Baseline aggregates (as much history as we have, capped to 20 years back)
    try:
        min_season = int(pd.to_numeric(wide["season"], errors="coerce").min())
    except Exception:
        min_season = season - years_back
    base_lo = max(min_season, season - years_back)
    base_hi = season - 1
    base = season_aggregates(wide, meta, lo=base_lo, hi=base_hi)

    fused = fuse_current_with_baseline(cur, base, season=season)
    if fused.empty():
        return pd.DataFrame()

    # Standardize numeric cols and compute a composite rating
    num_cols = [c for c in fused.columns if c not in ["team","games","season"]]
    Z = fused.copy()
    for c in num_cols:
        Z[c] = _z(Z[c])

    if num_cols:
        Z["rating"] = Z[num_cols].mean(axis=1)
    else:
        Z["rating"] = 0.0

    return Z[["team","rating","games"]].copy()

# ---------- name resolution ----------
_STOP = {"THE","OF","UNIVERSITY","UNIV","U","ST","&","AND","AT"}
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
    if strength.empty or "team" not in strength.columns:
        return []
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
