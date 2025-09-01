import json, os, re, math
import datetime as dt
from io import StringIO
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss

# --- NaN guard for inference ---
def _clean_row_for_model(X, columns):
    """
    Return a 1-row DataFrame with exactly `columns`, all numeric, no NaNs.
    Accepts dict, list/ndarray, or DataFrame.
    """
    import numpy as np
    import pandas as pd

    if isinstance(X, dict):
        df = pd.DataFrame([X])
    elif isinstance(X, (list, tuple, np.ndarray)):
        df = pd.DataFrame([X], columns=columns[:len(X)])
    elif isinstance(X, pd.DataFrame):
        df = X.copy()
    else:
        # last resort: try to wrap whatever it is
        df = pd.DataFrame(X)

    # ensure exact column set & order; fill missing with 0
    df = df.reindex(columns=columns, fill_value=0.0)
    # force numeric and kill NaNs
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df


# =========================
# Paths (prefer local CFBD; fallback to snapshot)
# =========================
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
INPUT_ALIASES_JSON = os.path.join("docs", "input", "aliases.json")  # optional
INPUT_LINES_CSV = os.path.join("docs", "input", "lines.csv")        # optional manual lines for predictions

PRED_OUT_JSON = os.path.join("docs", "data", "predictions.json")

# =========================
# Settings
# =========================
LAST_N = 5  # last-N games (pre-game rolling)
USE_SEASON_AHEAD_CV = True  # compute season-ahead metrics
RECENT_YEARS_ONLY = None    # e.g., 10 to limit training to last 10 seasons; None = all

# Core stats from box score long-form
STAT_FEATURES = [
    "totalYards","netPassingYards","rushingYards",
    "firstDowns","turnovers","sacks","tacklesForLoss",
    "thirdDownEff","fourthDownEff","kickingPoints"
]
# Additional engineered features names
ENGINEERED = ["rest_diff", "shortweek_diff", "bye_diff", "travel_diff_km",
              "spread_home", "over_under", "neutral_site", "is_postseason"]

# Ensemble weights
ELO_WEIGHT  = 0.55    # Elo probability weight
STAT_WEIGHT = 0.45    # Calibrated logistic probability weight

# Elo params
ELO_START = 1500.0
ELO_K_BASE = 20.0
ELO_K_EARLY = 32.0     # higher K weeks 1-4
ELO_HFA = 55.0         # home field advantage points
MEAN_REVERT = 0.30     # off-season pull toward 1500 (30%)

# =========================
# Aliases (expand as needed)
# =========================
BUILTIN_ALIASES = {
    "ohio state": "Ohio State Buckeyes", "texas": "Texas Longhorns",
    "northwestern": "Northwestern Wildcats", "tulane": "Tulane Green Wave",
    "lsu": "LSU Tigers", "clemson": "Clemson Tigers",
    "utep": "UTEP Miners", "utah state": "Utah State Aggies",
    "fresno state": "Fresno State Bulldogs", "georgia southern": "Georgia Southern Eagles",
    "arizona": "Arizona Wildcats", "hawaii": "Hawai'i Rainbow Warriors",
    "hawai'i": "Hawai'i Rainbow Warriors", "utah": "Utah Utes",
    "ucla": "UCLA Bruins", "south carolina": "South Carolina Gamecocks",
    "virginia": "Virginia Cavaliers", "oregon": "Oregon Ducks",
    "california": "California Golden Bears", "notre dame": "Notre Dame Fighting Irish",
    "miami": "Miami (FL) Hurricanes", "miami (fl)": "Miami (FL) Hurricanes",
}

# =========================
# Utilities
# =========================
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
    patterns = [
        re.compile(r"^\s*(?P<away>.+?)\s*@\s*(?P<home>.+?)\s*$", re.I),      # Away @ Home
        re.compile(r"^\s*(?P<home>.+?)\s*vs\.?\s*(?P<away>.+?)\s*$", re.I),  # Home vs Away
        re.compile(r"^\s*(?P<home>.+?)\s*,\s*(?P<away>.+?)\s*$", re.I),      # Home, Away
    ]
    games = []
    if not os.path.exists(path):
        print(f"[WARN] {path} not found.")
        return games
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            matched = None
            for pat in patterns:
                m = pat.match(line)
                if m:
                    matched = m.groupdict()
                    break
            if not matched:
                print(f"[SKIP] Unrecognized line: {line}")
                continue
            home = normalize_name(" ".join(matched["home"].split()), alias_map)
            away = normalize_name(" ".join(matched["away"].split()), alias_map)
            games.append({"home": home, "away": away})
    return games

def numericize_stat(cat, val):
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

# ---------- Schedule normalization ----------
def ensure_schedule_columns(schedule: pd.DataFrame) -> pd.DataFrame:
    """Make sure required columns exist and 'date' is available or synthesized."""
    df = schedule.rename(columns=lambda c: c.strip())
    for col, default in [
        ("season_type", "regular"),
        ("neutral_site", False),
        ("venue_id", np.nan),
        ("home_points", np.nan),
        ("away_points", np.nan),
    ]:
        if col not in df.columns:
            df[col] = default
    df["season"] = df["season"].apply(to_int) if "season" in df.columns else 0
    df["week"] = df["week"].apply(to_int) if "week" in df.columns else 0

    date_col = None
    for cand in ["date", "startDate", "start_date", "game_date", "startTime", "start_time"]:
        if cand in df.columns:
            date_col = cand
            break
    if date_col is not None:
        dt_series = df[date_col].apply(to_dt)
    else:
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

# -------------------------
# Rolling form: home-only & away-only last-N pregame means
# -------------------------
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

# -------------------------
# Rest/Travel features
# -------------------------
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

# -------------------------
# Lines: median per game across providers
# -------------------------
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

# -------------------------
# Build training set with last-N home/away form + engineered features + lines
# -------------------------
def build_training_examples(schedule: pd.DataFrame,
                            wide: pd.DataFrame,
                            lines_df: pd.DataFrame,
                            teams_df: pd.DataFrame,
                            venues_df: pd.DataFrame,
                            n: int) -> Tuple[pd.DataFrame, List[str]]:
    # rolling form
    home_roll, away_roll = team_rolling_home_away(wide, schedule, n)

    # Base schedule columns
    base_cols = ["game_id","home_team","away_team","home_points","away_points","season","week","season_type","date","neutral_site"]
    for bc in base_cols:
        if bc not in schedule.columns:
            schedule[bc] = np.nan if bc not in ["season","week","neutral_site","season_type"] else (0 if bc in ["season","week"] else (False if bc=="neutral_site" else "regular"))
    base = schedule[base_cols].copy()

    # Keep ONLY feature columns from rolling frames to avoid overlap explosions
    hr_needed = ["game_id","team"] + [f"home_R{n}_{c}" for c in STAT_FEATURES]
    ar_needed = ["game_id","team"] + [f"away_R{n}_{c}" for c in STAT_FEATURES]
    home_roll_feat = home_roll[[c for c in hr_needed if c in home_roll.columns]].copy()
    away_roll_feat = away_roll[[c for c in ar_needed if c in away_roll.columns]].copy()

    # Merge rolling features
    X = base.merge(home_roll_feat, left_on=["game_id","home_team"], right_on=["game_id","team"], how="left").drop(columns=["team"])
    X = X.merge(away_roll_feat, left_on=["game_id","away_team"], right_on=["game_id","team"], how="left").drop(columns=["team"])

    # diffs
    diff_cols = []
    for c in STAT_FEATURES:
        hc, ac = f"home_R{n}_{c}", f"away_R{n}_{c}"
        dc = f"diff_R{n}_{c}"
        if hc not in X.columns: X[hc] = 0.0
        if ac not in X.columns: X[ac] = 0.0
        X[dc] = X[hc].fillna(0.0) - X[ac].fillna(0.0)
        diff_cols.append(dc)

    # engineered features
    eng = rest_and_travel(schedule, teams_df, venues_df)
    overlap = [c for c in eng.columns if c in X.columns and c != "game_id"]
    if overlap:
        eng = eng.drop(columns=overlap)
    X = X.merge(eng, on="game_id", how="left")

    # lines
    med = median_lines(lines_df)
    X = X.merge(med, on="game_id", how="left")
    if "spread_home" not in X.columns: X["spread_home"] = 0.0
    if "over_under" not in X.columns:  X["over_under"] = 0.0

    # label
    X["home_win"] = (pd.to_numeric(X["home_points"], errors="coerce") > pd.to_numeric(X["away_points"], errors="coerce")).astype(int)

    # feature columns
    feature_cols = diff_cols + ["rest_diff","shortweek_diff","bye_diff","travel_diff_km","spread_home","over_under","neutral_site","is_postseason"]

    # >>> fix dtype & FutureWarning: force numeric then fill
    X[feature_cols] = X[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # Optionally limit to recent years
    if RECENT_YEARS_ONLY:
        max_season = int(pd.to_numeric(X["season"], errors="coerce").max())
        X = X[pd.to_numeric(X["season"], errors="coerce") >= max_season - RECENT_YEARS_ONLY + 1]

    # Only completed games for training
    X = X.dropna(subset=["home_points","away_points"])
    return X, feature_cols

# -------------------------
# Season-ahead CV + calibration
# -------------------------
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
        # sklearn 1.5+: use 'estimator', not 'base_estimator'
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
    return {
        "acc": float(dfm["acc"].mean()),
        "auc": float(dfm["auc"].mean()),
        "brier": float(dfm["brier"].mean())
    }

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
    # sklearn 1.5+: use 'estimator', not 'base_estimator'
    calib = CalibratedClassifierCV(estimator=base, method=method, cv="prefit")
    calib.fit(X_ca, y_ca)
    return calib

# -------------------------
# Elo with off-season mean reversion + early-season higher K
# -------------------------
def elo_expect(a: float, b: float) -> float:
    return 1.0 / (1.0 + 10 ** (-(a - b) / 400.0))

def mov_multiplier(point_diff: float, rating_diff: float) -> float:
    diff = abs(point_diff)
    if diff <= 0:
        return 1.0
    return math.log(diff + 1.0) * (2.2 / ((rating_diff * 0.001) + 2.2))

def train_elo(schedule: pd.DataFrame, talent_df: pd.DataFrame) -> Dict[str, float]:
    sched_cols = ["season","week","home_team","away_team","home_points","away_points","neutral_site"]
    for c in sched_cols:
        if c not in schedule.columns:
            schedule[c] = 0 if c in ["season","week"] else (False if c=="neutral_site" else np.nan)
    sched = schedule[sched_cols].dropna(subset=["home_team","away_team"]).copy()
    sched["season"] = pd.to_numeric(sched["season"], errors="coerce").fillna(0).astype(int)
    sched["week"] = pd.to_numeric(sched["week"], errors="coerce").fillna(0).astype(int)
    sched = sched.sort_values(["season","week"]).reset_index(drop=True)

    talent = pd.DataFrame()
    if isinstance(talent_df, pd.DataFrame) and not talent_df.empty:
        t = talent_df.copy()
        t["year"] = pd.to_numeric(t["year"], errors="coerce")
        grp = t.groupby("year")["talent"].agg(["mean","std"]).reset_index().rename(columns={"mean":"mu","std":"sd"})
        t = t.merge(grp, on="year", how="left")
        t["talent_z"] = (t["talent"] - t["mu"]) / t["sd"].replace(0, np.nan)
        talent = t[["year","school","talent_z"]]

    R: Dict[str, float] = {}
    current_season = None

    def preseason_seed(team: str, year: int) -> float:
        base = R.get(team, ELO_START)
        base = ELO_START + (base - ELO_START) * (1.0 - MEAN_REVERT)
        if not talent.empty:
            z = talent[(talent["year"]==year) & (talent["school"]==team)]
            if len(z):
                base += float(z.iloc[0]["talent_z"]) * 25.0
        return base

    for _, row in sched.iterrows():
        season = int(row["season"]); week = int(row["week"])
        if current_season is None or season != current_season:
            teams = set(list(R.keys()) + [row["home_team"], row["away_team"]])
            new_R = {}
            for tm in teams:
                new_R[tm] = preseason_seed(tm, season)
            R = new_R
            current_season = season

        h, a = row["home_team"], row["away_team"]
        hp = float(pd.to_numeric(row["home_points"], errors="coerce") or 0.0)
        ap = float(pd.to_numeric(row["away_points"], errors="coerce") or 0.0)
        ra, rb = R.get(h, ELO_START), R.get(a, ELO_START)

        hfa = 0.0 if bool(row["neutral_site"]) else ELO_HFA
        exp_h = elo_expect(ra + hfa, rb)
        exp_a = 1.0 - exp_h

        if hp == ap:
            score_h, score_a = 0.5, 0.5
            mov = 0.0
        else:
            score_h = 1.0 if hp > ap else 0.0
            score_a = 1.0 - score_h
            mov = abs(hp - ap)

        K = ELO_K_EARLY if week <= 4 else ELO_K_BASE
        if mov > 0:
            K = K * mov_multiplier(mov, abs(ra - rb))

        R[h] = ra + K * (score_h - exp_h)
        R[a] = rb + K * (score_a - exp_a)

    return R

def prob_from_elo(elo: Dict[str,float], home: str, away: str, neutral: bool) -> float:
    rh = elo.get(home, ELO_START)
    ra = elo.get(away, ELO_START)
    hfa = 0.0 if neutral else ELO_HFA
    return elo_expect(rh + hfa, ra)

# -------------------------
# Predict for input games (supports optional manual lines)
# -------------------------
def predict_games(cal_model: CalibratedClassifierCV,
                  features_lastn: pd.DataFrame,
                  schedule: pd.DataFrame,
                  elo_ratings: Dict[str,float],
                  games: List[Dict[str,str]],
                  alias_map: Dict[str,str],
                  manual_lines: pd.DataFrame) -> Tuple[List[Dict], List[str]]:
    teams_in_dataset = set(pd.concat([schedule["home_team"], schedule["away_team"]]).dropna().unique())
    def safe_name(n):
        m = normalize_name(n, alias_map)
        return m if m in teams_in_dataset else n

    rows, unknown = [], set()

    man = pd.DataFrame()
    if isinstance(manual_lines, pd.DataFrame) and not manual_lines.empty:
        man = manual_lines.copy()
        man.columns = [c.strip().lower() for c in man.columns]
        for col in ["home","away"]:
            if col in man.columns:
                man[col] = man[col].apply(lambda x: safe_name(str(x)) if pd.notna(x) else x)

    for g in games:
        home, away = safe_name(g["home"]), safe_name(g["away"])
        try:
            vh = features_lastn.loc[(home, "home")].values.astype(float)
        except KeyError:
            vh = np.zeros(len(STAT_FEATURES))
            unknown.add(home)
        try:
            va = features_lastn.loc[(away, "away")].values.astype(float)
        except KeyError:
            va = np.zeros(len(STAT_FEATURES))
            unknown.add(away)

        diff = vh - va
        feat = {f"diff_R{LAST_N}_{c}": diff[i] for i, c in enumerate(STAT_FEATURES)}
        feat.update({
            "rest_diff": 0.0, "shortweek_diff": 0.0, "bye_diff": 0.0,
            "travel_diff_km": 0.0, "neutral_site": 0.0, "is_postseason": 0.0,
            "spread_home": 0.0, "over_under": 0.0
        })

        if not man.empty and "home" in man.columns and "away" in man.columns:
            mrow = man[(man["home"]==home) & (man["away"]==away)]
            if not mrow.empty:
                feat["spread_home"] = float(pd.to_numeric(mrow.iloc[0].get("spread", 0), errors="coerce") or 0.0)
                feat["over_under"] = float(pd.to_numeric(mrow.iloc[0].get("over_under", 0), errors="coerce") or 0.0)

        feature_cols = [f"diff_R{LAST_N}_{c}" for c in STAT_FEATURES] + ENGINEERED
        X = np.array([[feat[col] for col in feature_cols]])

        p_stat = float(cal_model.predict_proba(X)[0,1])
        p_elo  = prob_from_elo(elo_ratings, home, away, neutral=False)
        p_home = float(ELO_WEIGHT * p_elo + STAT_WEIGHT * p_stat)

        rows.append({
            "home": home, "away": away,
            "home_prob": round(p_home, 4),
            "away_prob": round(1.0 - p_home, 4),
            "pick": home if p_home >= 0.5 else away
        })

    return rows, sorted(unknown)

# -------------------------
# Main
# -------------------------
def main():
    print("Loading schedule & team stats ...")
    schedule_raw = load_csv_local_or_url(LOCAL_SCHEDULE, FALLBACK_SCHEDULE_URL)
    stats    = load_csv_local_or_url(LOCAL_TEAM_STATS, FALLBACK_TEAM_STATS_URL).rename(columns=str.strip)

    # Normalize schedule (creates 'date' if missing)
    schedule = ensure_schedule_columns(schedule_raw)

    # Optional data
    lines   = pd.read_csv(LOCAL_LINES)   if os.path.exists(LOCAL_LINES)   else pd.DataFrame()
    venues  = pd.read_csv(LOCAL_VENUES)  if os.path.exists(LOCAL_VENUES)  else pd.DataFrame()
    teams   = pd.read_csv(LOCAL_TEAMS)   if os.path.exists(LOCAL_TEAMS)   else pd.DataFrame()
    talent  = pd.read_csv(LOCAL_TALENT)  if os.path.exists(LOCAL_TALENT)  else pd.DataFrame()

    # Build wide stats
    wide = long_stats_to_wide(stats)

    # Build training examples
    examples, feature_cols = build_training_examples(schedule, wide, lines, teams, venues, LAST_N)

    # Season-ahead metrics (no leakage)
    metrics_cv = {"acc": np.nan, "auc": np.nan, "brier": np.nan}
    if USE_SEASON_AHEAD_CV:
        try:
            metrics_cv = season_ahead_metrics(examples, feature_cols)
        except Exception as e:
            print(f"[WARN] Season-ahead CV failed: {e}")

    # Fit final calibrated model
    cal_model = fit_final_calibrated(examples, feature_cols)

    # Train Elo with mean reversion & talent
    elo_ratings = train_elo(schedule.dropna(subset=["home_team","away_team"]), talent)

    # Prepare last-N feature lookup for predictions:
    home_roll, away_roll = team_rolling_home_away(wide, schedule, LAST_N)

    def latest(df, role):
        # take latest row per team for that role
        d = df.copy()
        d["__order"] = range(len(d))
        d = d.sort_values("__order")
        d = d.groupby("team").tail(1).drop(columns="__order")
        d["role"] = role
        d = d.set_index(["team","role"])
        cols = [f"{role}_R{LAST_N}_{c}" for c in STAT_FEATURES]
        d = d[cols]
        return d.rename(columns={f"{role}_R{LAST_N}_{c}": f"R{LAST_N}_{c}" for c in STAT_FEATURES})

    latest_home = latest(home_roll, "home")
    latest_away = latest(away_roll, "away")
    features_lastn = pd.concat([latest_home, latest_away], axis=0).sort_index()

    # Input games + optional manual lines
    alias_map = load_alias_map()
    raw_games = parse_games_txt(INPUT_GAMES_TXT, alias_map)
    manual_lines = pd.read_csv(INPUT_LINES_CSV) if os.path.exists(INPUT_LINES_CSV) else pd.DataFrame()

    rows, unknown = predict_games(cal_model, features_lastn, schedule, elo_ratings, raw_games, alias_map, manual_lines)

    out = {
        "generated_at": dt.datetime.utcnow().replace(microsecond=0).isoformat()+"Z",
        "model": f"ensemble_last{LAST_N} (Elo {ELO_WEIGHT:.0%} + Calibrated stats {STAT_WEIGHT:.0%})",
        "metric": {
            "season_ahead_acc": round(float(metrics_cv["acc"]), 4) if not pd.isna(metrics_cv["acc"]) else None,
            "season_ahead_auc": round(float(metrics_cv["auc"]), 4) if not pd.isna(metrics_cv["auc"]) else None,
            "season_ahead_brier": round(float(metrics_cv["brier"]), 4) if not pd.isna(metrics_cv["brier"]) else None
        },
        "unknown_teams": unknown,
        "games": rows
    }
    os.makedirs(os.path.dirname(PRED_OUT_JSON), exist_ok=True)
    with open(PRED_OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {PRED_OUT_JSON}")

if __name__ == "__main__":
    main()
