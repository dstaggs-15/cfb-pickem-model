#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CFB Pick'em model — v2.1

Upgrades vs v2:
- Adds score prediction with two regressors (home & away points)
- Derives expected margin/total and home cover probability vs spread

Still includes from v2:
- Train/predict feature parity (true last-5, split by home/away)
- Neutral-site handling (HFA=0; neutral flag into model)
- Smart imputation (season means) + *_count features
- Robust stat parsing (5-12, 5/12, 5 of 12, 5 for 12)
- Non-linear classifier: HistGradientBoostingClassifier (+ calibration)
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
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
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
    # Flags & ids
    if "season_type" not in df.columns: df["season_type"] = "regular"
    if "neutral_site" not in df.columns: df["neutral_site"] = False
    if "venue_id" not in df.columns: df["venue_id"] = np.nan
    if "home_points" not in df.columns: df["home_points"] = np.nan
    if "away_points" not in df.columns: df["away_points"] = np.nan
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
      home_roll: ['game_id','team', f'home_R{n}_count', f'home_R{n}_<stat>'...]
      away_roll: ['game_id','team', f'away_R{n}_count', f'away_R{n}_<stat>'...]
    Values are team's mean over last-N prior games at that side.
    """
    sw = schedule[["game_id","date"]].copy()
    w = wide.merge(sw, on="game_id", how="left")
    w = w.sort_values(["team","date","game_id"]).reset_index(drop=True)

    out = []
    for side in ["home","away"]:
        side_df = w[w["homeAway"]==side].copy()
        side_df = side_df.sort_values(["team","date","game_id"])
        grp = side_df.groupby("team", group_keys=False)

        side_df[f"{side}_games_so_far"] = grp.cumcount()
        for c in STAT_FEATURES:
            side_df[f"{side}_R{n}_{c}"] = grp[c].apply(lambda s: s.rolling(window=n, min_periods=1).mean()).shift(1)
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
    Returns (a,b). If not enough data, use a mild prior.
    """
    ok = ~np.isnan(spread) & ~np.isnan(y)
    spread = spread[ok]; y = y[ok]
    if len(spread) < 200:
        return (0.0, 0.17)  # decent default
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
        rows.append({"game_id": r["game_id"], "elo_home_prob": p_home})

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
    home_roll, away_roll = build_sidewise_rollups(schedule, wide, n)

    base_cols = ["game_id","season","week","date","home_team","away_team","home_points","away_points","season_type","neutral_site","venue_id"]
    for bc in base_cols:
        if bc not in schedule.columns:
            schedule[bc] = np.nan if bc not in ["season","week","neutral_site","season_type"] else (0 if bc in ["season","week"] else (False if bc=="neutral_site" else "regular"))
    base = schedule[base_cols].copy()

    X = base.merge(home_roll, left_on=["game_id","home_team"], right_on=["game_id","team"], how="left").drop(columns=["team"])
    X = X.merge(away_roll, left_on=["game_id","away_team"], right_on=["game_id","team"], how="left").drop(columns=["team"])

    diff_cols = []
    for c in STAT_FEATURES:
        hc = f"home_R{n}_{c}"
        ac = f"away_R{n}_{c}"
        dc = f"diff_R{n}_{c}"
        X[dc] = X[hc] - X[ac]
        diff_cols.append(dc)

    if f"home_R{n}_count" not in X.columns: X[f"home_R{n}_count"] = 0.0
    if f"away_R{n}_count" not in X.columns: X[f"away_R{n}_count"] = 0.0

    eng = rest_and_travel(schedule, teams_df, venues_df)
    X = X.merge(eng, on="game_id", how="left")

    med = median_lines(lines_df)
    X = X.merge(med, on="game_id", how="left")

    elo_df = pregame_elo_probs(schedule, talent_df)
    X = X.merge(elo_df, on="game_id", how="left")

    X["home_win"] = (pd.to_numeric(X["home_points"], errors="coerce") > pd.to_numeric(X["away_points"], errors="coerce")).astype(int)

    feat_cols = diff_cols + [f"home_R{n}_count", f"away_R{n}_count"] + ENG_FEATURES_BASE + LINE_FEATURES + ["elo_home_prob"]
    for c in feat_cols:
        if c not in X.columns:
            X[c] = np.nan
    X["_season"] = pd.to_numeric(X["season"], errors="coerce")
    for c in feat_cols:
        if c in ["neutral_site","is_postseason"]:
            X[c] = X[c].fillna(0.0)
            continue
        X[c] = pd.to_numeric(X[c], errors="coerce")
        m = X.groupby("_season")[c].transform("mean")
        X[c] = X[c].fillna(m)
    X = X.drop(columns=["_season"])

    a, b = fit_market_mapping(X["spread_home"].to_numpy(dtype=float), X["home_win"].to_numpy(dtype=float))
    X["market_home_prob"] = [market_prob(s, a, b) if pd.notna(s) else np.nan for s in X["spread_home"]]
    mkt_season_mean = X.groupby("season")["market_home_prob"].transform("mean")
    X["market_home_prob"] = X["market_home_prob"].fillna(mkt_season_mean)

    feature_cols = diff_cols + [f"home_R{n}_count", f"away_R{n}_count"] + ENG_FEATURES_BASE + LINE_FEATURES + PROB_FEATURES

    train_df = X.dropna(subset=["home_points","away_points"]).copy()
    return train_df, feature_cols, {"market_a": a, "market_b": b}

# =========================================================
# Season-ahead validation & final fit (classifier)
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
# Score prediction models (regressors)
# =========================================================
def fit_score_models(df: pd.DataFrame, features: List[str]):
    """
    Train two regressors to predict home_points and away_points from the same features.
    Also estimate sigma for margin residuals for cover probability.
    """
    # Keep rows with valid scores
    d = df.dropna(subset=["home_points","away_points"]).copy()
    y_home = pd.to_numeric(d["home_points"], errors="coerce")
    y_away = pd.to_numeric(d["away_points"], errors="coerce")

    rg_home = HistGradientBoostingRegressor(max_depth=None, learning_rate=0.08, max_iter=600,
                                            min_samples_leaf=20, l2_regularization=0.0)
    rg_away = HistGradientBoostingRegressor(max_depth=None, learning_rate=0.08, max_iter=600,
                                            min_samples_leaf=20, l2_regularization=0.0)

    rg_home.fit(d[features], y_home)
    rg_away.fit(d[features], y_away)

    # Estimate residual variance for margin
    pred_home = rg_home.predict(d[features])
    pred_away = rg_away.predict(d[features])
    resid_margin = (y_home - y_away) - (pred_home - pred_away)
    sigma_margin = float(np.nanstd(resid_margin)) if len(resid_margin) else 13.0  # fallback ~2 TDs

    return rg_home, rg_away, sigma_margin

def std_norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

# =========================================================
# games.txt parsing (neutral support)
# =========================================================
GAMES_PATTERNS = [
    re.compile(r"^\s*(?P<away>.+?)\s*@\s*(?P<home>.+?)\s*$", re.I),
    re.compile(r"^\s*(?P<home>.+?)\s*vs\.?\s*(?P<away>.+?)(?:\s*\(N\))?\s*$", re.I),
    re.compile(r"^\s*(?P<home>.+?)\s*,\s*(?P
