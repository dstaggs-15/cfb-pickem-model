#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CFB Pick'em model — v2

Key upgrades:
- Train/predict feature parity (true last-5, split by home/away)
- Neutral-site handling (HFA=0; neutral flag into model)
- Smart imputation (season means) + *_count features
- Robust stat parsing (5-12, 5/12, 5 of 12, 5 for 12)
- Non-linear model: HistGradientBoostingClassifier (+ calibration)
- Elo pregame win-prob & Market-implied win-prob as FEATURES
"""

import os
import re
import json
import math
import datetime as dt
from io import StringIO
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import requests

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss

# ----------------------------
# Files & inputs
# ----------------------------
LOCAL_DIR = "data/raw/cfbd"
LOCAL_SCHEDULE = f"{LOCAL_DIR}/cfb_schedule.csv"
LOCAL_TEAM_STATS = f"{LOCAL_DIR}/cfb_game_team_stats.csv"
LOCAL_LINES = f"{LOCAL_DIR}/cfb_lines.csv"
LOCAL_VENUES = f"{LOCAL_DIR}/cfbd_venues.csv"
LOCAL_TEAMS = f"{LOCAL_DIR}/cfbd_teams.csv"
LOCAL_TALENT = f"{LOCAL_DIR}/cfbd_talent.csv"

RAW_BASE = "https://raw.githubusercontent.com/moneyball-ab/cfb-data/master/csv"
FALLBACK_SCHEDULE_URL = f"{RAW_BASE}/cfb_schedule.csv"
FALLBACK_TEAM_STATS_URL = f"{RAW_BASE}/cfb_game_team_stats.csv"

INPUT_GAMES_TXT = "docs/input/games.txt"
INPUT_ALIASES_JSON = "docs/input/aliases.json"
INPUT_LINES_CSV = "docs/input/lines.csv"

PRED_OUT_JSON = "docs/data/predictions.json"

# ----------------------------
# Settings
# ----------------------------
LAST_N = 5                 # rolling window
RECENT_YEARS_ONLY = None   # e.g., 8 to limit history
USE_SEASON_AHEAD_CV = True

# Features from cfbd game team stats
STAT_FEATURES = [
    "totalYards", "netPassingYards", "rushingYards", "firstDowns",
    "turnovers", "sacks", "tacklesForLoss",
    "thirdDownEff", "fourthDownEff", "kickingPoints",
]

ENG_FEATURES_BASE = [
    "rest_diff", "shortweek_diff", "bye_diff", "travel_diff_km",
    "neutral_site", "is_postseason",
]
LINE_FEATURES = ["spread_home", "over_under"]
PROB_FEATURES = ["elo_home_prob", "market_home_prob"]

# Elo params (can be tuned later)
ELO_START = 1500.0
ELO_K_BASE = 20.0
ELO_K_EARLY = 32.0
ELO_HFA = 55.0
MEAN_REVERT = 0.30

# Built-in alias seeds (extend via docs/input/aliases.json)
BUILTIN_ALIASES = {
    "hawaii": "Hawai'i Rainbow Warriors",
    "hawai'i": "Hawai'i Rainbow Warriors",
    "miami": "Miami (FL) Hurricanes",
    "miami (fl)": "Miami (FL) Hurricanes",
    "utep": "UTEP Miners",
    "utsa": "UTSA Roadrunners",
    "ole miss": "Ole Miss Rebels",
}

# =========================================================
# Utilities
# =========================================================
def load_csv_local_or_url(local_path: str, fallback_url: str) -> pd.DataFrame:
    if os.path.exists(local_path):
        return pd.read_csv(local_path)
    r = requests.get(fallback_url, timeout=60)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text))

def to_int(x, default=0):
    try:
        return int(float(x))
    except Exception:
        return default

def to_dt(x):
    try:
        return pd.to_datetime(x, utc=True)
    except Exception:
        return pd.NaT

def ensure_schedule_columns(schedule: pd.DataFrame) -> pd.DataFrame:
    df = schedule.rename(columns=lambda c: c.strip())
    # Normalize
    for c in ["season","week"]:
        if c in df.columns:
            df[c] = df[c].apply(to_int)
        else:
            df[c] = 0
    # Date
    date_col = None
    for cand in ["date","startDate","start_date","game_date","startTime","start_time"]:
        if cand in df.columns:
            date_col = cand; break
    if date_col is not None:
        df["date"] = df[date_col].apply(to_dt)
    else:
        df["date"] = pd.NaT
    # Boolean-ish flags
    if "season_type" not in df.columns: df["season_type"] = "regular"
    if "neutral_site" not in df.columns: df["neutral_site"] = False
    if "venue_id" not in df.columns: df["venue_id"] = np.nan
    if "home_points" not in df.columns: df["home_points"] = np.nan
    if "away_points" not in df.columns: df["away_points"] = np.nan
    # GameId
    if "game_id" not in df.columns:
        df["game_id"] = pd.util.hash_pandas_object(
            df[["season","week","home_team","away_team"]].fillna(""),
            index=False
        ).astype(np.int64)
    return df

def parse_ratio_val(val: str) -> float:
    """
    Handle 'a-b', 'a/b', 'a of b', 'a for b'. Return a/b.
    """
    if not isinstance(val, str):
        return float(val) if pd.notna(val) else np.nan
    s = val.strip().lower().replace("–", "-").replace("—", "-")
    m = re.match(r"^\s*(\d+)\s*(?:[-/]|of|for)\s*(\d+)\s*$", s)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        return a / b if b else np.nan
    # fallback numeric
    try:
        return float(val)
    except Exception:
        return np.nan

def numericize_stat(cat: str, val):
    if cat in ("thirdDownEff","fourthDownEff"):
        return parse_ratio_val(val)
    return pd.to_numeric(val, errors="coerce")

def long_stats_to_wide(df_stats: pd.DataFrame) -> pd.DataFrame:
    keep = df_stats[df_stats["category"].isin(STAT_FEATURES)].copy()
    keep["stat_value_num"] = [numericize_stat(c, v) for c, v in zip(keep["category"], keep["stat_value"])]
    wide = keep.pivot_table(index=["game_id","team","homeAway"],
                            columns="category", values="stat_value_num", aggfunc="mean").reset_index()
    # ensure columns
    for c in STAT_FEATURES:
        if c not in wide.columns:
            wide[c] = np.nan
    return wide

def load_alias_map() -> Dict[str, str]:
    alias = dict(BUILTIN_ALIASES)
    if os.path.exists(INPUT_ALIASES_JSON):
        try:
            with open(INPUT_ALIASES_JSON, "r") as f:
                extra = json.load(f)
            for k, v in extra.items():
                alias[str(k).strip().lower()] = str(v).strip()
        except Exception as e:
            print(f"[WARN] aliases.json read failed: {e}")
    return alias

def normalize_name(name: str, alias_map: Dict[str, str]) -> str:
    if not name: return ""
    key = name.strip().lower()
    return alias_map.get(key, name.strip())

# =========================================================
# Rolling features (home/away) + counts
# =========================================================
def build_sidewise_rollups(schedule: pd.DataFrame, wide: pd.DataFrame, n: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      home_roll: columns = ['game_id','team', f'home_R{n}_<stat>', f'home_R{n}_count']
      away_roll: columns = ['game_id','team', f'away_R{n}_<stat>', f'away_R{n}_count']
    For each game_id, values represent the team's average over its last-N *prior* games at that side.
    """
    # Join dates
    sw = schedule[["game_id","date"]].copy()
    w = wide.merge(sw, on="game_id", how="left")
    w = w.sort_values(["team","date","game_id"]).reset_index(drop=True)

    # Rolling per (team, homeAway), shifted by 1
    out = []
    for side in ["home","away"]:
        side_df = w[w["homeAway"]==side].copy()
        side_df = side_df.sort_values(["team","date","game_id"])
        grp = side_df.groupby("team", group_keys=False)

        # counts need min_periods=1 then shift
        counts = grp.cumcount()
        side_df[f"{side}_games_so_far"] = counts
        # rolling mean with min_periods=1, then shift so we don't peek
        for c in STAT_FEATURES:
            side_df[f"{side}_R{n}_{c}"] = grp[c].apply(lambda s: s.rolling(window=n, min_periods=1).mean()).shift(1)
        # count contributing samples for this rolling window (cap at n)
        side_df[f"{side}_R{n}_count"] = grp.apply(lambda g: g[f"{side}_games_so_far"].shift(1).clip(lower=0)).values
        side_df[f"{side}_R{n}_count"] = side_df[f"{side}_R{n}_count"].fillna(0).clip(upper=n)

        cols = ["game_id","team", f"{side}_R{n}_count"] + [f"{side}_R{n}_{c}" for c in STAT_FEATURES]
        out.append(side_df[cols])

    home_roll, away_roll = out
    return home_roll, away_roll

# =========================================================
# Rest / travel / context
# =========================================================
def haversine_km(lat1, lon1, lat2, lon2):
    if any(pd.isna(v) for v in [lat1,lon1,lat2,lon2]): return np.nan
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2-lat1)
    dlmb = math.radians(lon2-lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlmb/2)**2
    return 2*R*math.asin(math.sqrt(a))

def rest_and_travel(schedule: pd.DataFrame, teams_df: pd.DataFrame, venues_df: pd.DataFrame) -> pd.DataFrame:
    df = schedule[["game_id","season","week","date","home_team","away_team","neutral_site","venue_id","season_type"]].copy()

    # rest days
    both = pd.concat([
        df[["game_id","date","home_team"]].rename(columns={"home_team":"team"}),
        df[["game_id","date","away_team"]].rename(columns={"away_team":"team"})
    ], ignore_index=True).sort_values(["team","date"])
    both["prev_date"] = both.groupby("team")["date"].shift(1)
    both["rest_days"] = (both["date"] - both["prev_date"]).dt.days
    rest_map = both[["game_id","team","rest_days"]]

    m = df.merge(rest_map.rename(columns={"team":"home_team","rest_days":"home_rest_days"}),
                 on=["game_id","home_team"], how="left")
    m = m.merge(rest_map.rename(columns={"team":"away_team","rest_days":"away_rest_days"}),
                 on=["game_id","away_team"], how="left")
    m["home_rest_days"] = m["home_rest_days"].fillna(14)
    m["away_rest_days"] = m["away_rest_days"].fillna(14)
    m["home_short_week"] = (m["home_rest_days"] <= 6).astype(int)
    m["away_short_week"] = (m["away_rest_days"] <= 6).astype(int)
    m["home_bye"] = (m["home_rest_days"] >= 13).astype(int)
    m["away_bye"] = (m["away_rest_days"] >= 13).astype(int)

    # travel (campus -> venue; at neutral use venue coords; else away travels to home)
    def team_latlon(school: str):
        if teams_df.empty: return (np.nan, np.nan)
        r = teams_df[teams_df["school"]==school]
        if r.empty: return (np.nan, np.nan)
        return (r.iloc[0].get("latitude"), r.iloc[0].get("longitude"))

    def venue_latlon(vid):
        if venues_df.empty: return (np.nan, np.nan)
        r = venues_df[venues_df["venue_id"]==vid]
        if r.empty: return (np.nan, np.nan)
        return (r.iloc[0].get("latitude"), r.iloc[0].get("longitude"))

    m["home_lat"], m["home_lon"] = zip(*m["home_team"].map(team_latlon))
    m["away_lat"], m["away_lon"] = zip(*m["away_team"].map(team_latlon))
    m["ven_lat"], m["ven_lon"] = zip(*m["venue_id"].map(venue_latlon))

    def travel(row):
        if bool(row["neutral_site"]) and pd.notna(row["ven_lat"]) and pd.notna(row["ven_lon"]):
            hd = haversine_km(row["home_lat"], row["home_lon"], row["ven_lat"], row["ven_lon"])
            ad = haversine_km(row["away_lat"], row["away_lon"], row["ven_lat"], row["ven_lon"])
        else:
            hd = 0.0
            ad = haversine_km(row["away_lat"], row["away_lon"], row["home_lat"], row["home_lon"])
        return hd, ad

    m["home_travel_km"], m["away_travel_km"] = zip(*m.apply(travel, axis=1))

    m["rest_diff"] = m["home_rest_days"] - m["away_rest_days"]
    m["shortweek_diff"] = m["home_short_week"] - m["away_short_week"]
    m["bye_diff"] = m["home_bye"] - m["away_bye"]
    m["travel_diff_km"] = m["home_travel_km"] - m["away_travel_km"]
    m["is_postseason"] = (m["season_type"].astype(str) != "regular").astype(int)

    keep = m[["game_id","rest_diff","shortweek_diff","bye_diff","travel_diff_km","neutral_site","is_postseason"]]
    return keep

# =========================================================
# Lines → median & market-implied win prob
# =========================================================
def median_lines(lines: pd.DataFrame) -> pd.DataFrame:
    if lines is None or lines.empty:
        return pd.DataFrame(columns=["game_id","spread_home","over_under"])
    df = lines.copy()
    for old, new in [("spread","spread"),("overUnder","over_under"),("overunder","over_under")]:
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
    df["spread"] = pd.to_numeric(df.get("spread"), errors="coerce")
    df["over_under"] = pd.to_numeric(df.get("over_under"), errors="coerce")
    if "game_id" not in df.columns:
        return pd.DataFrame(columns=["game_id","spread_home","over_under"])
    med = df.groupby("game_id")[["spread","over_under"]].median().reset_index()
    med = med.rename(columns={"spread":"spread_home"})
    return med

def fit_market_mapping(spread: np.ndarray, y: np.ndarray) -> Tuple[float,float]:
    """
    Fit logistic: P(home win) = sigmoid(a + b * (-spread_home))
    Returns (a,b). If not enough data, fall back to simple prior.
    """
    ok = ~np.isnan(spread) & ~np.isnan(y)
    spread = spread[ok]; y = y[ok]
    if len(spread) < 200:
        # weak prior: roughly converts -7 => ~70%
        return (0.0, 0.17)
    X = (-spread).reshape(-1,1)
    lr = LogisticRegression(max_iter=200)
    lr.fit(X, y)
    a = float(lr.intercept_[0]); b = float(lr.coef_[0][0])
    return (a,b)

def market_prob(spread_home: float, a: float, b: float) -> Optional[float]:
    if spread_home is None or pd.isna(spread_home):
        return None
    z = a + b * (-float(spread_home))
    return 1.0 / (1.0 + math.exp(-z))

# =========================================================
# Elo — pregame probability (no leakage)
# =========================================================
def elo_expect(a: float, b: float) -> float:
    return 1.0 / (1.0 + 10 ** (-(a - b) / 400.0))

def mov_multiplier(point_diff: float, rating_diff: float) -> float:
    d = abs(point_diff)
    if d <= 0: return 1.0
    return math.log(d + 1.0) * (2.2 / ((rating_diff * 0.001) + 2.2))

def preseason_seed_from_talent(talent_df: pd.DataFrame) -> Dict[str, float]:
    seed = {}
    if talent_df is None or talent_df.empty: return seed
    t = talent_df.copy()
    t.columns = [c.strip() for c in t.columns]
    if "school" in t.columns and "talent" in t.columns:
        t["talent"] = pd.to_numeric(t["talent"], errors="coerce")
        m = float(t["talent"].mean())
        s = float(t["talent"].std()) if pd.notna(t["talent"].std()) else 1.0
        for _, r in t.iterrows():
            sc = str(r["school"]).strip()
            z = 0.0 if s == 0 else (float(r["talent"]) - m) / s
            seed[sc] = ELO_START + 60.0 * z
    return seed

def pregame_elo_probs(schedule: pd.DataFrame, talent_df: pd.DataFrame) -> pd.DataFrame:
    """
    Walk the schedule in chronological order. For each game, compute Elo P(home win)
    BEFORE updating ratings with the game result. HFA=0 when neutral.
    Return: DataFrame [game_id, elo_home_prob]
    """
    sched = schedule.sort_values(["season","week","date","game_id"]).copy()
    seed = preseason_seed_from_talent(talent_df)
    ratings: Dict[str, float] = {}

    cur_season = None
    rows = []

    for _, r in sched.iterrows():
        season = int(r["season"])
        week = int(r["week"])
        home = str(r["home_team"])
        away = str(r["away_team"])
        neutral = bool(r.get("neutral_site", False))
        hp = pd.to_numeric(r.get("home_points"), errors="coerce")
        ap = pd.to_numeric(r.get("away_points"), errors="coerce")

        # Offseason mean reversion on first game of a season
        if cur_season is None or season != cur_season:
            for k in list(ratings.keys()):
                ratings[k] = ELO_START + (ratings[k] - ELO_START) * (1.0 - MEAN_REVERT)
            # apply seed (blend 50/50 with existing)
            for sc, val in seed.items():
                ratings[sc] = 0.5*ratings.get(sc, ELO_START) + 0.5*val
            cur_season = season

        ra = ratings.get(home, seed.get(home, ELO_START))
        rb = ratings.get(away, seed.get(away, ELO_START))
        hfa = 0.0 if neutral else ELO_HFA
        p_home = elo_expect(ra + hfa, rb)
        rows.append({"game_id": r["game_id"], "elo_home_prob": p_home})

        # Update only if result known (postgame)
        if pd.notna(hp) and pd.notna(ap):
            score_home = 1.0 if hp > ap else (0.5 if hp == ap else 0.0)
            k = ELO_K_EARLY if week <= 4 else ELO_K_BASE
            movm = mov_multiplier(hp - ap, (ra + hfa) - rb)
            ratings[home] = ra + k * movm * (score_home - p_home)
            ratings[away] = rb + k * movm * ((1.0 - score_home) - (1.0 - p_home))

    return pd.DataFrame(rows)

# =========================================================
# Training set builder
# =========================================================
def build_training_examples(
    schedule: pd.DataFrame, wide: pd.DataFrame, lines_df: pd.DataFrame,
    teams_df: pd.DataFrame, venues_df: pd.DataFrame, talent_df: pd.DataFrame, n: int
) -> Tuple[pd.DataFrame, List[str], Dict[str, float]]:
    # Core
    home_roll, away_roll = build_sidewise_rollups(schedule, wide, n)

    base_cols = ["game_id","season","week","date","home_team","away_team","home_points","away_points","season_type","neutral_site","venue_id"]
    for bc in base_cols:
        if bc not in schedule.columns:
            schedule[bc] = np.nan if bc not in ["season","week","neutral_site","season_type"] else (0 if bc in ["season","week"] else (False if bc=="neutral_site" else "regular"))
    base = schedule[base_cols].copy()

    # Join rolling (home/away)
    X = base.merge(home_roll, left_on=["game_id","home_team"], right_on=["game_id","team"], how="left").drop(columns=["team"])
    X = X.merge(away_roll, left_on=["game_id","away_team"], right_on=["game_id","team"], how="left").drop(columns=["team"])

    # Build diffs & counts
    diff_cols = []
    for c in STAT_FEATURES:
        hc = f"home_R{n}_{c}"
        ac = f"away_R{n}_{c}"
        dc = f"diff_R{n}_{c}"
        X[dc] = X[hc] - X[ac]
        diff_cols.append(dc)

    # counts
    if f"home_R{n}_count" not in X.columns: X[f"home_R{n}_count"] = 0.0
    if f"away_R{n}_count" not in X.columns: X[f"away_R{n}_count"] = 0.0

    eng = rest_and_travel(schedule, teams_df, venues_df)
    X = X.merge(eng, on="game_id", how="left")

    # Lines median + market prob
    med = median_lines(lines_df)
    X = X.merge(med, on="game_id", how="left")

    # Elo pregame prob (no leakage)
    elo_df = pregame_elo_probs(schedule, talent_df)
    X = X.merge(elo_df, on="game_id", how="left")

    # Target
    X["home_win"] = (pd.to_numeric(X["home_points"], errors="coerce") > pd.to_numeric(X["away_points"], errors="coerce")).astype(int)

    # Impute numeric features to season means (no zero fakes)
    feat_cols = diff_cols + [f"home_R{n}_count", f"away_R{n}_count"] + ENG_FEATURES_BASE + LINE_FEATURES + ["elo_home_prob"]
    for c in feat_cols:
        if c not in X.columns:
            X[c] = np.nan
    # season means
    X["_season"] = pd.to_numeric(X["season"], errors="coerce")
    for c in feat_cols:
        if c in ["neutral_site","is_postseason"]:  # already 0/1
            X[c] = X[c].fillna(0.0)
            continue
        # numeric
        X[c] = pd.to_numeric(X[c], errors="coerce")
        m = X.groupby("_season")[c].transform("mean")
        X[c] = X[c].fillna(m)
    X = X.drop(columns=["_season"])

    # Fit market mapping on available rows
    a, b = fit_market_mapping(X["spread_home"].to_numpy(dtype=float), X["home_win"].to_numpy(dtype=float))
    X["market_home_prob"] = [market_prob(s, a, b) if pd.notna(s) else np.nan for s in X["spread_home"]]
    # Impute market prob to season mean too
    mkt_season_mean = X.groupby("season")["market_home_prob"].transform("mean")
    X["market_home_prob"] = X["market_home_prob"].fillna(mkt_season_mean)

    # Final feature list
    feature_cols = diff_cols + [f"home_R{n}_count", f"away_R{n}_count"] + ENG_FEATURES_BASE + LINE_FEATURES + PROB_FEATURES

    # Drop rows without result for training
    train_df = X.dropna(subset=["home_points","away_points"]).copy()
    return train_df, feature_cols, {"market_a": a, "market_b": b}

# =========================================================
# Season-ahead validation & final fit
# =========================================================
def season_ahead_metrics(df: pd.DataFrame, features: List[str]) -> Dict[str, float]:
    seasons = sorted(pd.to_numeric(df["season"], errors="coerce").dropna().unique().tolist())
    res = []
    for i in range(2, len(seasons)):
        test_season = seasons[i]
        calib_season = seasons[i-1]
        train_seasons = seasons[:i-1]
        tr = df[df["season"].isin(train_seasons)]
        ca = df[df["season"]==calib_season]
        te = df[df["season"]==test_season]

        if len(tr) < 200 or len(ca) < 80 or len(te) < 80: 
            continue

        base = HistGradientBoostingClassifier(max_depth=None, learning_rate=0.08, max_iter=400,
                                              min_samples_leaf=20, l2_regularization=0.0)
        base.fit(tr[features], tr["home_win"])
        method = "isotonic" if len(ca) >= 400 else "sigmoid"
        cal = CalibratedClassifierCV(estimator=base, method=method, cv="prefit")
        cal.fit(ca[features], ca["home_win"])

        p = cal.predict_proba(te[features])[:,1]
        res.append({
            "acc": accuracy_score(te["home_win"], (p>=0.5).astype(int)),
            "auc": roc_auc_score(te["home_win"], p),
            "brier": brier_score_loss(te["home_win"], p),
        })

    if not res:
        return {"acc": np.nan, "auc": np.nan, "brier": np.nan}
    d = pd.DataFrame(res)
    return {"acc": float(d["acc"].mean()), "auc": float(d["auc"].mean()), "brier": float(d["brier"].mean())}

def fit_final_model(df: pd.DataFrame, features: List[str]) -> CalibratedClassifierCV:
    seasons = sorted(pd.to_numeric(df["season"], errors="coerce").dropna().unique().tolist())
    if len(seasons) >= 2:
        calib_season = seasons[-1]
        tr = df[df["season"] < calib_season]
        ca = df[df["season"] == calib_season]
        method = "isotonic" if len(ca) >= 400 else "sigmoid"
    else:
        df = df.sort_values("date")
        split = int(len(df)*0.9)
        tr, ca = df.iloc[:split], df.iloc[split:]
        method = "sigmoid"

    base = HistGradientBoostingClassifier(max_depth=None, learning_rate=0.08, max_iter=400,
                                          min_samples_leaf=20, l2_regularization=0.0)
    base.fit(tr[features], tr["home_win"])
    cal = CalibratedClassifierCV(estimator=base, method=method, cv="prefit")
    cal.fit(ca[features], ca["home_win"])
    return cal

# =========================================================
# games.txt parsing (neutral support)
# =========================================================
GAMES_PATTERNS = [
    re.compile(r"^\s*(?P<away>.+?)\s*@\s*(?P<home>.+?)\s*$", re.I),         # Away @ Home
    re.compile(r"^\s*(?P<home>.+?)\s*vs\.?\s*(?P<away>.+?)(?:\s*\(N\))?\s*$", re.I),  # Home vs Away [(N)]
    re.compile(r"^\s*(?P<home>.+?)\s*,\s*(?P<away>.+?)\s*$", re.I),         # Home, Away
]

def parse_games_txt(path: str, alias_map: Dict[str,str]) -> List[Dict[str,object]]:
    out = []
    if not os.path.exists(path):
        return out
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"): continue
            neutral = "(N)" in line or " (n)" in line
            matched = None
            for pat in GAMES_PATTERNS:
                m = pat.match(line)
                if m: matched = m.groupdict(); break
            if not matched:
                print(f"[SKIP] Unrecognized line: {line}"); continue
            home = normalize_name(" ".join(matched["home"].split()), alias_map)
            away = normalize_name(" ".join(matched["away"].split()), alias_map)
            out.append({"home": home, "away": away, "neutral": neutral})
    return out

# =========================================================
# Predict custom games — build same features
# =========================================================
def predict_games(
    model: CalibratedClassifierCV, features_lastn: List[str],
    schedule: pd.DataFrame, wide: pd.DataFrame,
    teams_df: pd.DataFrame, venues_df: pd.DataFrame, talent_df: pd.DataFrame,
    raw_games: List[Dict[str,object]], manual_lines: pd.DataFrame,
    market_params: Dict[str,float]
) -> Tuple[List[Dict], List[str]]:
    alias_unknown = []

    # Build sidewise rollups using entire history up to "now"
    home_roll, away_roll = build_sidewise_rollups(schedule, wide, LAST_N)

    # Precompute per-team last observed rolling rows by side
    def last_row(df: pd.DataFrame, prefix: str):
        cols = [c for c in df.columns if c.startswith(f"{prefix}_R{LAST_N}_")]
        cols_count = [f"{prefix}_R{LAST_N}_count"]
        need = ["team"] + cols_count + cols
        v = df[need].dropna(how="all", subset=cols).sort_values(["team"]).groupby("team").tail(1).set_index("team")
        return v

    last_home = last_row(home_roll, "home")
    last_away = last_row(away_roll, "away")

    # For neutral detection from schedule: map (home,away)->neutral flag if exists this season
    sched_now = schedule.copy()
    season_max = int(pd.to_numeric(sched_now["season"], errors="coerce").max()) if "season" in sched_now.columns else None
    if season_max is not None:
        sched_now = sched_now[sched_now["season"]==season_max]
    pair_neutral = {}
    if {"home_team","away_team","neutral_site"}.issubset(sched_now.columns):
        tmp = sched_now.sort_values(["week","date"]).drop_duplicates(subset=["home_team","away_team"], keep="last")
        for _, r in tmp.iterrows():
            pair_neutral[(str(r["home_team"]), str(r["away_team"]))] = bool(r["neutral_site"])

    # Market mapping params
    a = float(market_params.get("market_a", 0.0))
    b = float(market_params.get("market_b", 0.17))

    # Pregame Elo probs (for all historical games) already computed; for new pairs we approximate with current ratings:
    elo_pregame = pregame_elo_probs(schedule, talent_df)
    # Use latest Elo per team for custom game probability
    # Build ratings from last known game (approx: reverse solve expectation):
    # We can estimate team rating by solving for ra given p and rb ; but simpler:
    # derive per-team rating from last pregame expectation on a known opponent.
    # Instead, recompute end-of-history ratings quickly:
    def end_of_history_ratings():
        sched = schedule.sort_values(["season","week","date","game_id"]).copy()
        seed = preseason_seed_from_talent(talent_df)
        ratings = {}
        cur_season = None
        for _, r in sched.iterrows():
            season = int(r["season"]); week = int(r["week"])
            home, away = str(r["home_team"]), str(r["away_team"])
            neutral = bool(r.get("neutral_site", False))
            hp = pd.to_numeric(r.get("home_points"), errors="coerce")
            ap = pd.to_numeric(r.get("away_points"), errors="coerce")
            if cur_season is None or season != cur_season:
                for k in list(ratings.keys()):
                    ratings[k] = ELO_START + (ratings[k] - ELO_START) * (1.0 - MEAN_REVERT)
                for sc, val in seed.items():
                    ratings[sc] = 0.5*ratings.get(sc, ELO_START) + 0.5*val
                cur_season = season
            ra = ratings.get(home, seed.get(home, ELO_START))
            rb = ratings.get(away, seed.get(away, ELO_START))
            hfa = 0.0 if neutral else ELO_HFA
            p_home = elo_expect(ra + hfa, rb)
            if pd.notna(hp) and pd.notna(ap):
                score_home = 1.0 if hp > ap else (0.5 if hp == ap else 0.0)
                k = ELO_K_EARLY if week <= 4 else ELO_K_BASE
                movm = mov_multiplier(hp - ap, (ra + hfa) - rb)
                ratings[home] = ra + k * movm * (score_home - p_home)
                ratings[away] = rb + k * movm * ((1.0 - score_home) - (1.0 - p_home))
        return ratings

    current_ratings = end_of_history_ratings()

    # Manual lines lookup
    man = pd.DataFrame()
    if isinstance(manual_lines, pd.DataFrame) and not manual_lines.empty:
        man = manual_lines.copy()
        man.columns = [c.strip().lower() for c in man.columns]
        if not {"home","away","spread","over_under"}.issubset(set(man.columns)):
            man = pd.DataFrame()
        else:
            man["spread"] = pd.to_numeric(man["spread"], errors="coerce")
            man["over_under"] = pd.to_numeric(man["over_under"], errors="coerce")

    def lines_for(home, away):
        if man.empty: return (np.nan, np.nan)
        r = man[(man["home"]==home) & (man["away"]==away)]
        if r.empty: return (np.nan, np.nan)
        return float(r.iloc[0]["spread"]), float(r.iloc[0]["over_under"])

    rows = []
    for g in raw_games:
        home = str(g.get("home","")).strip()
        away = str(g.get("away","")).strip()
        neutral_flag = bool(g.get("neutral", False))
        if not home or not away:
            continue

        # Use schedule's neutral flag if present for this pair this season
        neutral = neutral_flag or pair_neutral.get((home, away), False)

        # Rolling diffs + counts
        hrow = last_home.loc[home] if home in last_home.index else None
        arow = last_away.loc[away] if away in last_away.index else None

        feats = {}
        # counts
        feats[f"home_R{LAST_N}_count"] = float(hrow[f"home_R{LAST_N}_count"]) if hrow is not None else 0.0
        feats[f"away_R{LAST_N}_count"] = float(arow[f"away_R{LAST_N}_count"]) if arow is not None else 0.0
        # diffs
        for c in STAT_FEATURES:
            hv = float(hrow[f"home_R{LAST_N}_{c}"]) if (hrow is not None and pd.notna(hrow[f"home_R{LAST_N}_{c}"])) else np.nan
            av = float(arow[f"away_R{LAST_N}_{c}"]) if (arow is not None and pd.notna(arow[f"away_R{LAST_N}_{c}"])) else np.nan
            feats[f"diff_R{LAST_N}_{c}"] = hv - av if (pd.notna(hv) and pd.notna(av)) else np.nan

        # Context: can't know rest/bye future reliably → 0 baseline
        feats["rest_diff"] = 0.0
        feats["shortweek_diff"] = 0.0
        feats["bye_diff"] = 0.0
        feats["neutral_site"] = 1.0 if neutral else 0.0
        feats["is_postseason"] = 0.0

        # Travel: away to home campus (neutral assumed center not known)
        # we approximate travel_diff_km with away->home distance if coords exist
        # (if not, 0)
        if os.path.exists(LOCAL_TEAMS):
            tdf = pd.read_csv(LOCAL_TEAMS)
            tdf_cols = {c.lower(): c for c in tdf.columns}
            for req in ["school","latitude","longitude"]:
                if req not in tdf.columns and req in tdf_cols:
                    tdf.rename(columns={tdf_cols[req]: req}, inplace=True)
            def latlon(school):
                r = tdf[tdf["school"]==school]
                if r.empty: return (np.nan, np.nan)
                return (r.iloc[0].get("latitude"), r.iloc[0].get("longitude"))
            hlat, hlon = latlon(home); alat, alon = latlon(away)
            feats["travel_diff_km"] = 0.0 if neutral else (haversine_km(alat, alon, hlat, hlon) if all(pd.notna(v) for v in [hlat,hlon,alat,alon]) else 0.0)
        else:
            feats["travel_diff_km"] = 0.0

        # Lines & market prob
        sp, ou = lines_for(home, away)
        feats["spread_home"] = sp if pd.notna(sp) else np.nan
        feats["over_under"] = ou if pd.notna(ou) else np.nan
        feats["market_home_prob"] = market_prob(feats["spread_home"], a, b) if pd.notna(feats["spread_home"]) else np.nan

        # Elo pregame prob (use current ratings)
        ra = current_ratings.get(home, ELO_START)
        rb = current_ratings.get(away, ELO_START)
        hfa = 0.0 if neutral else ELO_HFA
        feats["elo_home_prob"] = elo_expect(ra + hfa, rb)

        # Impute to global means for this inference set (fallback)
        # We'll build a 1-row DF and let the model accept NaNs (HGB tolerates),
        # but to be safe we fill NaNs of key engineered features with mid values.
        # Assemble row in feature order:
        feat_order = features_lastn
        X = pd.DataFrame([{k: feats.get(k, np.nan) for k in feat_order}])
        # Simple impute for line/prob if fully missing
        for c in ["market_home_prob","elo_home_prob"]:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        # Predict
        p_home = float(model.predict_proba(X)[0,1])
        rows.append({
            "home": home, "away": away,
            "home_prob": round(p_home, 4),
            "away_prob": round(1.0 - p_home, 4),
            "pick": home if p_home >= 0.5 else away,
            "neutral": bool(neutral),
            "spread_home": None if pd.isna(sp) else float(sp),
            "over_under": None if pd.isna(ou) else float(ou),
            "p_elo": round(float(feats["elo_home_prob"]), 4),
            "p_market": None if pd.isna(feats["market_home_prob"]) else round(float(feats["market_home_prob"]), 4)
        })

    return rows, sorted(set(alias_unknown))

# =========================================================
# Main
# =========================================================
def main():
    print("Loading schedule & team stats ...")

    alias_map = load_alias_map()
    schedule = load_csv_local_or_url(LOCAL_SCHEDULE, FALLBACK_SCHEDULE_URL)
    schedule = ensure_schedule_columns(schedule)

    team_stats = load_csv_local_or_url(LOCAL_TEAM_STATS, FALLBACK_TEAM_STATS_URL)
    wide = long_stats_to_wide(team_stats)

    lines_df = pd.read_csv(LOCAL_LINES) if os.path.exists(LOCAL_LINES) else pd.DataFrame()
    venues_df = pd.read_csv(LOCAL_VENUES) if os.path.exists(LOCAL_VENUES) else pd.DataFrame()
    teams_df = pd.read_csv(LOCAL_TEAMS) if os.path.exists(LOCAL_TEAMS) else pd.DataFrame()
    talent_df = pd.read_csv(LOCAL_TALENT) if os.path.exists(LOCAL_TALENT) else pd.DataFrame()

    # Build training set
    examples, feature_cols, market_params = build_training_examples(
        schedule, wide, lines_df, teams_df, venues_df, talent_df, LAST_N
    )

    # Optionally trim history
    if RECENT_YEARS_ONLY:
        max_season = int(pd.to_numeric(examples["season"], errors="coerce").max())
        examples = examples[pd.to_numeric(examples["season"], errors="coerce") >= max_season - RECENT_YEARS_ONLY + 1]

    # Season-ahead validation
    metric = {}
    if USE_SEASON_AHEAD_CV:
        try:
            m = season_ahead_metrics(examples, feature_cols)
            metric = {
                "season_ahead_acc": round(m["acc"], 4) if pd.notna(m["acc"]) else None,
                "season_ahead_auc": round(m["auc"], 4) if pd.notna(m["auc"]) else None,
                "season_ahead_brier": round(m["brier"], 4) if pd.notna(m["brier"]) else None,
            }
        except Exception as e:
            print(f"[WARN] Season-ahead CV failed: {e}")

    # Fit final calibrated model
    cal_model = fit_final_model(examples, feature_cols)

    # Read user games + manual lines
    raw_games = parse_games_txt(INPUT_GAMES_TXT, alias_map)
    manual_lines = pd.read_csv(INPUT_LINES_CSV) if os.path.exists(INPUT_LINES_CSV) else pd.DataFrame()

    # Predict
    rows, unknown = predict_games(
        cal_model, feature_cols, schedule, wide, teams_df, venues_df, talent_df,
        raw_games, manual_lines, market_params
    )

    out = {
        "generated": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": f"ensemble_last{LAST_N} (HGB + calib; Elo/Market as features)",
        "metric": metric,
        "games": rows,
        "unknown_teams": unknown,
    }
    os.makedirs(os.path.dirname(PRED_OUT_JSON), exist_ok=True)
    with open(PRED_OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {PRED_OUT_JSON}")

if __name__ == "__main__":
    main()
