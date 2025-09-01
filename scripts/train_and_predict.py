#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a calibrated logistic model on rolling team stats (+ context features),
blend with Elo, and write website-friendly predictions.

Outputs: docs/data/predictions.json

Inputs (preferred local, else fallback URLs for schedule/stats):
- data/raw/cfbd/cfb_schedule.csv
- data/raw/cfbd/cfb_game_team_stats.csv
- data/raw/cfbd/cfb_lines.csv (optional, but used if present)
- data/raw/cfbd/cfbd_venues.csv (optional: travel features)
- data/raw/cfbd/cfbd_teams.csv  (optional: team lat/long & for travel)
- data/raw/cfbd/cfbd_talent.csv (optional: preseason Elo seeding)
- docs/input/games.txt          (your weekly input; various formats)
- docs/input/aliases.json       (optional: map nicknames → canonical)
- docs/input/lines.csv          (optional: manual lines for *input* games)

Requires: pandas, numpy, scikit-learn, requests
"""

import os
import re
import json
import math
from io import StringIO
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
import datetime as dt

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss

# ----------------------------
# Paths & constants
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

INPUT_GAMES_TXT = os.path.join("docs", "input", "games.txt")
INPUT_ALIASES_JSON = os.path.join("docs", "input", "aliases.json")
INPUT_LINES_CSV = os.path.join("docs", "input", "lines.csv")

PRED_OUT_JSON = os.path.join("docs", "data", "predictions.json")

# ----------------------------
# Model settings
# ----------------------------
LAST_N = 5                         # rolling window size (form)
RECENT_YEARS_ONLY = None           # e.g., 6 to use only last 6 seasons
USE_SEASON_AHEAD_CV = True

# Stats chosen from cfbd team game stats
STAT_FEATURES = [
    "totalYards",        # offense output proxy
    "netPassingYards",
    "rushingYards",
    "firstDowns",
    "turnovers",
    "sacks",
    "tacklesForLoss",
    "thirdDownEff",      # as ratio made/att
    "fourthDownEff",     # as ratio made/att
    "kickingPoints"
]
ENG_FEATURES = [
    "rest_diff", "shortweek_diff", "bye_diff", "travel_diff_km",
    "spread_home", "over_under",
    "neutral_site", "is_postseason",
]

# Elo blend
ELO_WEIGHT  = 0.55
STAT_WEIGHT = 0.45
ELO_START = 1500.0
ELO_K_BASE = 20.0
ELO_K_EARLY = 32.0              # weeks 1–4
ELO_HFA = 55.0
MEAN_REVERT = 0.30              # pull to mean each Jan 1

# Built-in alias hints (you can extend via docs/input/aliases.json)
BUILTIN_ALIASES = {
    "ohio state": "Ohio State Buckeyes",
    "texas": "Texas Longhorns",
    "northwestern": "Northwestern Wildcats",
    "tulane": "Tulane Green Wave",
    "lsu": "LSU Tigers",
    "clemson": "Clemson Tigers",
    "utep": "UTEP Miners",
    "utah state": "Utah State Aggies",
    "fresno state": "Fresno State Bulldogs",
    "georgia southern": "Georgia Southern Eagles",
    "arizona": "Arizona Wildcats",
    "hawaii": "Hawai'i Rainbow Warriors",
    "hawai'i": "Hawai'i Rainbow Warriors",
    "utah": "Utah Utes",
    "ucla": "UCLA Bruins",
    "south carolina": "South Carolina Gamecocks",
    "virginia": "Virginia Cavaliers",
    "oregon": "Oregon Ducks",
    "california": "California Golden Bears",
    "notre dame": "Notre Dame Fighting Irish",
    "miami": "Miami (FL) Hurricanes",
    "miami (fl)": "Miami (FL) Hurricanes",
}

# =========================================================
# Utility
# =========================================================
def load_csv_local_or_url(local_path: str, fallback_url: str) -> pd.DataFrame:
    if os.path.exists(local_path):
        return pd.read_csv(local_path)
    r = requests.get(fallback_url, timeout=60)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text))

def load_alias_map() -> Dict[str, str]:
    alias = dict(BUILTIN_ALIASES)
    if os.path.exists(INPUT_ALIASES_JSON):
        try:
            with open(INPUT_ALIASES_JSON, "r") as f:
                extra = json.load(f)
            for k, v in extra.items():
                alias[k.strip().lower()] = v.strip()
        except Exception as e:
            print(f"[WARN] Could not load aliases.json: {e}")
    return alias

def normalize_name(name: str, alias_map: Dict[str, str]) -> str:
    if not name: return ""
    key = name.strip().lower()
    return alias_map.get(key, name.strip())

def parse_games_txt(path: str, alias_map: Dict[str, str]) -> List[Dict[str, str]]:
    """Accept lines like: 'Away @ Home', 'Home vs Away', or 'Home, Away'."""
    patterns = [
        re.compile(r"^\s*(?P<away>.+?)\s*@\s*(?P<home>.+?)\s*$", re.I),
        re.compile(r"^\s*(?P<home>.+?)\s*vs\.?\s*(?P<away>.+?)\s*$", re.I),
        re.compile(r"^\s*(?P<home>.+?)\s*,\s*(?P<away>.+?)\s*$", re.I),
    ]
    games = []
    if not os.path.exists(path):
        print(f"[WARN] {path} not found.")
        return games
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"): continue
            matched = None
            for pat in patterns:
                m = pat.match(line)
                if m: matched = m.groupdict(); break
            if not matched:
                print(f"[SKIP] Unrecognized line: {line}"); continue
            home = normalize_name(" ".join(matched["home"].split()), alias_map)
            away = normalize_name(" ".join(matched["away"].split()), alias_map)
            games.append({"home": home, "away": away})
    return games

def numericize_stat(cat, val):
    # convert "5-12" to ratio for specific categories
    if isinstance(val, str) and "-" in val and cat in ["thirdDownEff","fourthDownEff",
                                                       "completionAttempts","totalPenaltiesYards"]:
        try:
            a, b = val.split("-"); a = float(a); b = float(b)
            return a / b if b != 0 else np.nan
        except Exception:
            return np.nan
    try:
        return float(val)
    except Exception:
        return np.nan

def long_stats_to_wide(df_stats: pd.DataFrame) -> pd.DataFrame:
    keep = df_stats[df_stats["category"].isin(STAT_FEATURES)].copy()
    keep["stat_value_num"] = [numericize_stat(c, v) for c, v in zip(keep["category"], keep["stat_value"])]
    wide = (keep.pivot_table(index=["game_id","team","homeAway"],
                             columns="category", values="stat_value_num", aggfunc="mean")
                 .reset_index())
    for c in STAT_FEATURES:
        if c not in wide.columns:
            wide[c] = np.nan
    return wide

def to_int(x, default=0):
    try: return int(float(x))
    except Exception: return default

def to_dt(x):
    try: return pd.to_datetime(x, utc=True)
    except Exception: return pd.NaT

def ensure_schedule_columns(schedule: pd.DataFrame) -> pd.DataFrame:
    df = schedule.rename(columns=lambda c: c.strip())
    # required-ish columns
    for col, default in [
        ("season_type", "regular"),
        ("neutral_site", False),
        ("venue_id", np.nan),
        ("home_points", np.nan),
        ("away_points", np.nan),
    ]:
        if col not in df.columns:
            df[col] = default
    if "season" in df.columns:
        df["season"] = df["season"].apply(to_int)
    else:
        df["season"] = 0
    if "week" in df.columns:
        df["week"] = df["week"].apply(to_int)
    else:
        df["week"] = 0

    date_col = None
    for cand in ["date","startDate","start_date","game_date","startTime","start_time"]:
        if cand in df.columns:
            date_col = cand; break
    if date_col is not None:
        dt_series = df[date_col].apply(to_dt)
    else:
        # synthesize something reasonable
        def synth(row):
            try:
                base = dt.datetime(int(row["season"]), 8, 1, tzinfo=dt.timezone.utc)
                return base + dt.timedelta(days=max(0, int(row["week"])-1)*7)
            except Exception:
                return pd.NaT
        dt_series = df.apply(synth, axis=1)
    df["date"] = dt_series

    if "game_id" not in df.columns:
        df["game_id"] = pd.util.hash_pandas_object(
            df[["season","week","home_team","away_team"]].fillna(""),
            index=False
        ).astype(np.int64)
    return df

def team_rolling_home_away(wide: pd.DataFrame, schedule: pd.DataFrame, n: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sw = schedule[["game_id","season","week","neutral_site","date"]].copy()
    w = wide.merge(sw, on="game_id", how="left")
    sort_keys = ["team","date","game_id"] if "date" in w.columns else ["team","season","week","game_id"]
    w = w.sort_values(sort_keys).reset_index(drop=True)

    for c in STAT_FEATURES:
        w[f"R{n}H_{c}"] = w.groupby(["team","homeAway"])[c].transform(
            lambda s: s.rolling(window=n, min_periods=1).mean().shift(1)
        )

    home = w[w["homeAway"]=="home"].copy()
    away = w[w["homeAway"]=="away"].copy()

    home = home.rename(columns={f"R{n}H_{c}": f"home_R{n}_{c}" for c in STAT_FEATURES})
    away = away.rename(columns={f"R{n}H_{c}": f"away_R{n}_{c}" for c in STAT_FEATURES})

    home = home[["game_id","team"] + [f"home_R{n}_{c}" for c in STAT_FEATURES]]
    away = away[["game_id","team"] + [f"away_R{n}_{c}" for c in STAT_FEATURES]]
    return home, away

def haversine_km(lat1, lon1, lat2, lon2):
    if None in [lat1, lon1, lat2, lon2] or any(pd.isna(v) for v in [lat1, lon1, lat2, lon2]):
        return np.nan
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2-lat1)
    dlmb = math.radians(lon2-lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlmb/2)**2
    return 2*R*math.asin(math.sqrt(a))

def rest_and_travel(schedule: pd.DataFrame, teams_df: pd.DataFrame, venues_df: pd.DataFrame) -> pd.DataFrame:
    df = schedule[["game_id","season","week","date","home_team","away_team","neutral_site","venue_id","season_type"]].copy()
    all_games = []
    for side in ["home","away"]:
        temp = df[["game_id","date",f"{side}_team"]].rename(columns={f"{side}_team": "team"})
        all_games.append(temp)
    tg = pd.concat(all_games, ignore_index=True).sort_values(["team","date"]).reset_index(drop=True)
    tg["prev_date"] = tg.groupby("team")["date"].shift(1)
    tg["rest_days"] = (tg["date"] - tg["prev_date"]).dt.days
    rest_map = tg[["game_id","team","rest_days"]]

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

    def latlon_team(school: str):
        if teams_df.empty: return (None, None)
        row = teams_df[teams_df["school"]==school]
        if row.empty: return (None, None)
        r = row.iloc[0]
        return (r.get("latitude"), r.get("longitude"))

    def latlon_venue(venue_id):
        if venues_df.empty: return (None, None)
        row = venues_df[venues_df["venue_id"]==venue_id]
        if row.empty: return (None, None)
        r = row.iloc[0]
        return (r.get("latitude"), r.get("longitude"))

    m["home_lat"], m["home_lon"] = zip(*m["home_team"].map(latlon_team))
    m["away_lat"], m["away_lon"] = zip(*m["away_team"].map(latlon_team))
    m["ven_lat"], m["ven_lon"] = zip(*m["venue_id"].map(latlon_venue))

    def travel_km(row):
        if bool(row["neutral_site"]) and pd.notna(row["ven_lat"]) and pd.notna(row["ven_lon"]):
            hd = haversine_km(row["home_lat"], row["home_lon"], row["ven_lat"], row["ven_lon"])
            ad = haversine_km(row["away_lat"], row["away_lon"], row["ven_lat"], row["ven_lon"])
            return hd, ad
        hd = 0.0
        ad = haversine_km(row["away_lat"], row["away_lon"], row["home_lat"], row["home_lon"])
        return hd, ad

    m["home_travel_km"], m["away_travel_km"] = zip(*m.apply(travel_km, axis=1))
    m["rest_diff"] = m["home_rest_days"] - m["away_rest_days"]
    m["shortweek_diff"] = m["home_short_week"] - m["away_short_week"]
    m["bye_diff"] = m["home_bye"] - m["away_bye"]
    m["travel_diff_km"] = m["home_travel_km"] - m["away_travel_km"]
    m["is_postseason"] = (m["season_type"].astype(str) != "regular").astype(int)

    keep = m[["game_id","rest_diff","shortweek_diff","bye_diff","travel_diff_km","neutral_site","is_postseason"]]
    return keep

def median_lines(lines: pd.DataFrame) -> pd.DataFrame:
    if lines is None or lines.empty:
        return pd.DataFrame(columns=["game_id","spread_home","over_under"])
    lines = lines.copy()
    for old, new in [("spread","spread"), ("overUnder","over_under"), ("overunder","over_under")]:
        if old in lines.columns and new not in lines.columns:
            lines[new] = lines[old]
    lines["spread"] = pd.to_numeric(lines.get("spread"), errors="coerce")
    lines["over_under"] = pd.to_numeric(lines.get("over_under"), errors="coerce")
    if "game_id" not in lines.columns:
        return pd.DataFrame(columns=["game_id","spread_home","over_under"])
    grp = lines.groupby("game_id")[["spread","over_under"]].median().reset_index()
    grp = grp.rename(columns={"spread":"spread_home"})
    return grp

def build_training_examples(schedule: pd.DataFrame,
                            wide: pd.DataFrame,
                            lines_df: pd.DataFrame,
                            teams_df: pd.DataFrame,
                            venues_df: pd.DataFrame,
                            n: int) -> Tuple[pd.DataFrame, List[str]]:
    home_roll, away_roll = team_rolling_home_away(wide, schedule, n)

    base_cols = ["game_id","home_team","away_team","home_points","away_points","season","week","season_type","date","neutral_site"]
    for bc in base_cols:
        if bc not in schedule.columns:
            schedule[bc] = np.nan if bc not in ["season","week","neutral_site","season_type"] else (0 if bc in ["season","week"] else (False if bc=="neutral_site" else "regular"))
    base = schedule[base_cols].copy()

    hr_needed = ["game_id","team"] + [f"home_R{n}_{c}" for c in STAT_FEATURES]
    ar_needed = ["game_id","team"] + [f"away_R{n}_{c}" for c in STAT_FEATURES]
    home_roll_feat = home_roll[[c for c in hr_needed if c in home_roll.columns]].copy()
    away_roll_feat = away_roll[[c for c in ar_needed if c in away_roll.columns]].copy()

    X = base.merge(home_roll_feat, left_on=["game_id","home_team"], right_on=["game_id","team"], how="left").drop(columns=["team"])
    X = X.merge(away_roll_feat, left_on=["game_id","away_team"], right_on=["game_id","team"], how="left").drop(columns=["team"])

    diff_cols = []
    for c in STAT_FEATURES:
        hc, ac = f"home_R{n}_{c}", f"away_R{n}_{c}"
        dc = f"diff_R{n}_{c}"
        if hc not in X.columns: X[hc] = 0.0
        if ac not in X.columns: X[ac] = 0.0
        X[dc] = X[hc].fillna(0.0) - X[ac].fillna(0.0)
        diff_cols.append(dc)

    eng = rest_and_travel(schedule, teams_df, venues_df)
    overlap = [c for c in eng.columns if c in X.columns and c != "game_id"]
    if overlap: eng = eng.drop(columns=overlap)
    X = X.merge(eng, on="game_id", how="left")

    med = median_lines(lines_df)
    X = X.merge(med, on="game_id", how="left")
    if "spread_home" not in X.columns: X["spread_home"] = 0.0
    if "over_under" not in X.columns:  X["over_under"] = 0.0

    X["home_win"] = (pd.to_numeric(X["home_points"], errors="coerce") > pd.to_numeric(X["away_points"], errors="coerce")).astype(int)

    feature_cols = diff_cols + ENG_FEATURES
    X[feature_cols] = X[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    if RECENT_YEARS_ONLY:
        max_season = int(pd.to_numeric(X["season"], errors="coerce").max())
        X = X[pd.to_numeric(X["season"], errors="coerce") >= max_season - RECENT_YEARS_ONLY + 1]

    X = X.dropna(subset=["home_points","away_points"])
    return X, feature_cols

def season_ahead_metrics(df: pd.DataFrame, features: List[str]) -> Dict[str, float]:
    seasons = sorted(pd.to_numeric(df["season"], errors="coerce").dropna().unique().tolist())
    res = []
    for i in range(2, len(seasons)):
        test_season = seasons[i]
        calib_season = seasons[i-1]
        train_seasons = seasons[:i-1]
        train_df = df[df["season"].isin(train_seasons)]
        calib_df = df[df["season"]==calib_season]
        test_df  = df[df["season"]==test_season]

        if len(train_df) < 200 or len(calib_df) < 100 or len(test_df) < 100:
            continue

        X_tr, y_tr = train_df[features].values, train_df["home_win"].values
        X_ca, y_ca = calib_df[features].values, calib_df["home_win"].values
        X_te, y_te = test_df[features].values, test_df["home_win"].values

        base = LogisticRegression(max_iter=500)
        base.fit(X_tr, y_tr)

        method = "isotonic" if len(calib_df) >= 400 else "sigmoid"
        calib = CalibratedClassifierCV(estimator=base, method=method, cv="prefit")
        calib.fit(X_ca, y_ca)

        p = calib.predict_proba(X_te)[:,1]
        acc = accuracy_score(y_te, (p>=0.5).astype(int))
        auc = roc_auc_score(y_te, p)
        brier = brier_score_loss(y_te, p)
        res.append({"season": int(test_season), "acc": acc, "auc": auc, "brier": brier})

    if not res:
        return {"acc": np.nan, "auc": np.nan, "brier": np.nan}

    dfm = pd.DataFrame(res)
    return {"acc": float(dfm["acc"].mean()),
            "auc": float(dfm["auc"].mean()),
            "brier": float(dfm["brier"].mean())}

def fit_final_calibrated(df: pd.DataFrame, features: List[str]) -> CalibratedClassifierCV:
    seasons = sorted(pd.to_numeric(df["season"], errors="coerce").dropna().unique().tolist())
    if len(seasons) >= 2:
        calib_season = seasons[-1]
        train_df = df[df["season"] < calib_season]
        calib_df = df[df["season"] == calib_season]
        method = "isotonic" if len(calib_df) >= 400 else "sigmoid"
    else:
        df = df.sort_values("date")
        split = int(len(df)*0.9)
        train_df, calib_df = df.iloc[:split], df.iloc[split:]
        method = "sigmoid"

    X_tr, y_tr = train_df[features].values, train_df["home_win"].values
    X_ca, y_ca = calib_df[features].values, calib_df["home_win"].values

    base = LogisticRegression(max_iter=600)
    base.fit(X_tr, y_tr)
    calib = CalibratedClassifierCV(estimator=base, method=method, cv="prefit")
    calib.fit(X_ca, y_ca)
    return calib

# ------- Elo helpers -------
def elo_expect(a: float, b: float) -> float:
    return 1.0 / (1.0 + 10 ** (-(a - b) / 400.0))

def mov_multiplier(point_diff: float, rating_diff: float) -> float:
    diff = abs(point_diff)
    if diff <= 0: return 1.0
    return math.log(diff + 1.0) * (2.2 / ((rating_diff * 0.001) + 2.2))

def train_elo(schedule: pd.DataFrame, talent_df: pd.DataFrame) -> Dict[str, float]:
    """Season-by-season Elo with mean reversion each offseason and larger K in weeks 1–4."""
    sched_cols = ["season","week","home_team","away_team","home_points","away_points","neutral_site"]
    for c in sched_cols:
        if c not in schedule.columns:
            schedule[c] = 0 if c in ["season","week"] else (False if c=="neutral_site" else np.nan)
    sched = schedule[sched_cols].dropna(subset=["home_team","away_team"]).copy()
    sched["season"] = pd.to_numeric(sched["season"], errors="coerce").fillna(0).astype(int)
    sched["week"] = pd.to_numeric(sched["week"], errors="coerce").fillna(0).astype(int)
    sched = sched.sort_values(["season","week"]).reset_index(drop=True)

    # preseason seed using talent (if available)
    seed = {}
    if isinstance(talent_df, pd.DataFrame) and not talent_df.empty:
        t = talent_df.copy()
        t.columns = [c.strip() for c in t.columns]
        if "school" in t.columns and "talent" in t.columns:
            t["talent"] = pd.to_numeric(t["talent"], errors="coerce")
            m = t["talent"].mean()
            s = t["talent"].std() if pd.notna(t["talent"].std()) else 1.0
            for _, r in t.iterrows():
                sc = str(r["school"]).strip()
                z = 0.0 if s == 0 else (float(r["talent"]) - m) / s
                seed[sc] = ELO_START + 60.0 * z  # scale talent to ~ +- 2 SD → +-120 Elo

    ratings: Dict[str, float] = {}
    cur_season = None
    for _, row in sched.iterrows():
        season = int(row["season"])
        week = int(row["week"])
        home = str(row["home_team"])
        away = str(row["away_team"])
        hp = pd.to_numeric(row["home_points"], errors="coerce")
        ap = pd.to_numeric(row["away_points"], errors="coerce")
        if pd.isna(hp) or pd.isna(ap):
            continue  # skip future/unplayed

        # offseason mean reversion at season change
        if cur_season is None or season != cur_season:
            for k in list(ratings.keys()):
                ratings[k] = ELO_START + (ratings[k] - ELO_START) * (1.0 - MEAN_REVERT)
            # apply seeds on top of reversion
            for sc, val in seed.items():
                ratings[sc] = 0.5*ratings.get(sc, ELO_START) + 0.5*val
            cur_season = season

        ra = ratings.get(home, seed.get(home, ELO_START))
        rb = ratings.get(away, seed.get(away, ELO_START))
        hfa = 0.0 if bool(row.get("neutral_site", False)) else ELO_HFA
        exp_home = elo_expect(ra + hfa, rb)
        k = ELO_K_EARLY if week <= 4 else ELO_K_BASE
        diff = float(hp - ap)
        movm = mov_multiplier(diff, (ra + hfa) - rb)
        score_home = 1.0 if diff > 0 else (0.5 if diff == 0 else 0.0)

        ratings[home] = ra + k * movm * (score_home - exp_home)
        ratings[away] = rb + k * movm * ((1.0 - score_home) - (1.0 - exp_home))
    return ratings

# ----------------------------
# Inference NaN guard
# ----------------------------
def _clean_row_for_model(X, columns):
    """
    Return a 1-row DataFrame with exactly `columns`, all numeric, no NaNs.
    """
    if isinstance(X, dict):
        df = pd.DataFrame([X])
    elif isinstance(X, pd.DataFrame):
        df = X.copy()
    else:
        df = pd.DataFrame([X], columns=columns[:len(X)])
    df = df.reindex(columns=columns, fill_value=0.0)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df

# ----------------------------
# Predict custom games
# ----------------------------
def predict_games(cal_model: CalibratedClassifierCV,
                  features_lastn: List[str],
                  schedule: pd.DataFrame,
                  elo_ratings: Dict[str, float],
                  raw_games: List[Dict[str, str]],
                  alias_map: Dict[str, str],
                  manual_lines: pd.DataFrame) -> Tuple[List[Dict], List[str]]:
    """
    Build a features row for each custom matchup and output probabilities.
    For simplicity:
      - rolling stat diffs use last known R{n} home/away means up to latest played
      - rest/bye/shortweek set to 0 (unknown future)
      - travel_diff_km approximates away travel to home campus (no neutral)
      - spread_home/over_under come from manual_lines if provided else 0
    """
    # Build per-team last-N by side (home/away) from historical wide stats
    # We’ll re-use schedule+stats logic: compute per-team rolling means then keep the last observed values.
    # To avoid recomputing, we’ll derive from the training examples perspective:
    # Get the last game per team where they were home (for home_R{n}_*) and away (for away_R{n}_*).
    # We’ll reconstruct quickly from schedule merges.
    # Load base data to construct a "latest per team" table.
    # For efficiency, reuse the same pipeline used earlier.

    # We'll prepare a minimal "per-team" summary from schedule: last observed rolling values.
    # Start by creating a fake "wide" from existing engineered training examples is not trivial,
    # so we instead compute raw per-team aggregates directly from past games.
    # Simple approach: compute the last-N overall (not split) and use same values for home/away.
    # But we want split; fallback to overall if side not found.

    # Make a quick game list to know unique teams referenced
    teams_needed = set()
    for g in raw_games:
        teams_needed.add(g["home"])
        teams_needed.add(g["away"])

    # We'll compute last-N per team separately for home and away using schedule/stats already merged in training.
    # To keep this function independent, we rebuild a mini table from schedule with rolling means we can query.
    # For prediction-time, side-specific stats improve a bit but we allow fallback to overall.

    # Precompute geographic coords if we have teams file
    teams_df = pd.read_csv(LOCAL_TEAMS) if os.path.exists(LOCAL_TEAMS) else pd.DataFrame()
    lat_map = {}
    lon_map = {}
    if not teams_df.empty and {"school","latitude","longitude"}.issubset(teams_df.columns):
        for _, r in teams_df.iterrows():
            lat_map[str(r["school"]).strip()] = r.get("latitude")
            lon_map[str(r["school"]).strip()] = r.get("longitude")

    def approx_travel_km(home_team, away_team):
        lat_h = lat_map.get(home_team)
        lon_h = lon_map.get(home_team)
        lat_a = lat_map.get(away_team)
        lon_a = lon_map.get(away_team)
        if any(v is None or pd.isna(v) for v in [lat_h, lon_h, lat_a, lon_a]):
            return 0.0
        return haversine_km(lat_a, lon_a, lat_h, lon_h)  # away travels to home

    # Manual lines lookup
    man = pd.DataFrame()
    if isinstance(manual_lines, pd.DataFrame) and not manual_lines.empty:
        man = manual_lines.copy()
        man.columns = [c.strip().lower() for c in man.columns]
        if {"home","away","spread","over_under"}.issubset(set(man.columns)):
            man["spread"] = pd.to_numeric(man["spread"], errors="coerce")
            man["over_under"] = pd.to_numeric(man["over_under"], errors="coerce")
        else:
            man = pd.DataFrame()  # ignore invalid file

    def line_for(home, away):
        if man.empty:
            return 0.0, 0.0
        row = man[(man["home"]==home) & (man["away"]==away)]
        if row.empty:
            return 0.0, 0.0
        return float(row.iloc[0]["spread"]), float(row.iloc[0]["over_under"])

    # Build a light per-team rolling table (overall last-N across home/away)
    # Using schedule+team stats long file is heavy here; instead, compute from points (proxy) if stats unavailable.
    # We’ll read stats wide if available to get the actual STAT_FEATURES; else zeros.
    wide = pd.DataFrame()
    try:
        stats_long = pd.read_csv(LOCAL_TEAM_STATS)
        wide = long_stats_to_wide(stats_long)
    except Exception:
        wide = pd.DataFrame(columns=["game_id","team","homeAway"] + STAT_FEATURES)

    sched = pd.read_csv(LOCAL_SCHEDULE) if os.path.exists(LOCAL_SCHEDULE) else pd.DataFrame()
    if not sched.empty:
        sched = ensure_schedule_columns(sched)
        sw = sched[["game_id","date"]].copy()
        wj = wide.merge(sw, on="game_id", how="left")
        wj = wj.sort_values(["team","date"]).reset_index(drop=True)
        # overall last-N mean per team (shift 1 to avoid peeking)
        overall = wj.groupby("team")[STAT_FEATURES].rolling(LAST_N, min_periods=1).mean().shift(1).reset_index()
        overall = overall.rename(columns={"level_1":"row"})
        overall["team"] = wj.loc[overall["row"], "team"].values
        last_overall = overall.dropna(how="all", subset=STAT_FEATURES).groupby("team").tail(1)
        team_overall = last_overall.set_index("team")[STAT_FEATURES].to_dict(orient="index")
    else:
        team_overall = {}

    # Results
    rows = []
    unknown = []

    for g in raw_games:
        home = g["home"]; away = g["away"]
        if not home or not away:
            continue

        # Feature vector
        feats = {}

        # Stat diffs: use overall last-N; if missing, zeros
        h_stats = team_overall.get(home, {})
        a_stats = team_overall.get(away, {})
        for c in STAT_FEATURES:
            hc = float(h_stats.get(c, 0.0) if pd.notna(h_stats.get(c, 0.0)) else 0.0)
            ac = float(a_stats.get(c, 0.0) if pd.notna(a_stats.get(c, 0.0)) else 0.0)
            feats[f"diff_R{LAST_N}_{c}"] = hc - ac

        # Context
        feats["rest_diff"] = 0.0
        feats["shortweek_diff"] = 0.0
        feats["bye_diff"] = 0.0
        feats["neutral_site"] = 0.0
        feats["is_postseason"] = 0.0
        feats["travel_diff_km"] = approx_travel_km(home, away)

        # Lines (home-relative spread)
        sp, ou = line_for(home, away)
        feats["spread_home"] = float(sp if pd.notna(sp) else 0.0)
        feats["over_under"] = float(ou if pd.notna(ou) else 0.0)

        X = _clean_row_for_model(feats, features_lastn)          # <<< crucial NaN guard
        p_stat = float(cal_model.predict_proba(X)[0,1])          # Prob home wins from stats model

        # Elo probability
        ra = elo_ratings.get(home, ELO_START)
        rb = elo_ratings.get(away, ELO_START)
        p_elo = elo_expect(ra + ELO_HFA, rb)                     # assume true home site

        p_home = STAT_WEIGHT * p_stat + ELO_WEIGHT * p_elo
        p_home = min(max(p_home, 1e-4), 1 - 1e-4)
        p_away = 1.0 - p_home
        pick = home if p_home >= 0.5 else away

        rows.append({
            "home": home,
            "away": away,
            "home_prob": round(p_home, 4),
            "away_prob": round(p_away, 4),
            "pick": pick,
            "spread_home": None if sp is None or (isinstance(sp, float) and np.isnan(sp)) else float(sp),
            "over_under": None if ou is None or (isinstance(ou, float) and np.isnan(ou)) else float(ou),
        })

    return rows, unknown

# ----------------------------
# main
# ----------------------------
def main():
    print("Loading schedule & team stats ...")
    alias_map = load_alias_map()

    # Load source data
    schedule = load_csv_local_or_url(LOCAL_SCHEDULE, FALLBACK_SCHEDULE_URL)
    schedule = ensure_schedule_columns(schedule)

    team_stats = load_csv_local_or_url(LOCAL_TEAM_STATS, FALLBACK_TEAM_STATS_URL)
    wide = long_stats_to_wide(team_stats)

    lines_df = pd.read_csv(LOCAL_LINES) if os.path.exists(LOCAL_LINES) else pd.DataFrame()
    venues_df = pd.read_csv(LOCAL_VENUES) if os.path.exists(LOCAL_VENUES) else pd.DataFrame()
    teams_df = pd.read_csv(LOCAL_TEAMS) if os.path.exists(LOCAL_TEAMS) else pd.DataFrame()
    talent_df = pd.read_csv(LOCAL_TALENT) if os.path.exists(LOCAL_TALENT) else pd.DataFrame()

    # Build training examples & features
    examples, feature_cols = build_training_examples(schedule, wide, lines_df, teams_df, venues_df, LAST_N)
    # Safety: numeric + no NaNs (training path)
    examples[feature_cols] = examples[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # Season-ahead validation metrics (optional)
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
            metric = {}

    # Fit final calibrated model
    cal_model = fit_final_calibrated(examples, feature_cols)

    # Elo ratings
    elo_ratings = train_elo(schedule, talent_df)

    # Read user games
    raw_games = parse_games_txt(INPUT_GAMES_TXT, alias_map)

    # Manual input lines for those games (optional)
    manual_lines = pd.read_csv(INPUT_LINES_CSV) if os.path.exists(INPUT_LINES_CSV) else pd.DataFrame()

    rows, unknown = predict_games(cal_model, feature_cols, schedule, elo_ratings, raw_games, alias_map, manual_lines)

    out = {
        "generated": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": f"ensemble_last{LAST_N} (Elo {int(ELO_WEIGHT*100)}% + Calibrated stats {int(STAT_WEIGHT*100)}%)",
        "metric": metric,
        "games": rows,
        "unknown_teams": sorted(list(set(unknown))),
    }

    os.makedirs(os.path.dirname(PRED_OUT_JSON), exist_ok=True)
    with open(PRED_OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {PRED_OUT_JSON}")

if __name__ == "__main__":
    main()
