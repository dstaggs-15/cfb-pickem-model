import json, os, re, math
import datetime as dt
from io import StringIO
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss

# =========================
# Config
# =========================
# Local (preferred if present)
LOCAL_SCHEDULE = "data/raw/cfbd/cfb_schedule.csv"
LOCAL_TEAM_STATS = "data/raw/cfbd/cfb_game_team_stats.csv"

# Fallback snapshot (older GitHub CSVs)
RAW_BASE = "https://raw.githubusercontent.com/moneyball-ab/cfb-data/master/csv"
FALLBACK_SCHEDULE_URL = f"{RAW_BASE}/cfb_schedule.csv"
FALLBACK_TEAM_STATS_URL = f"{RAW_BASE}/cfb_game_team_stats.csv"

INPUT_GAMES_TXT = os.path.join("docs", "input", "games.txt")
INPUT_ALIASES_JSON = os.path.join("docs", "input", "aliases.json")  # optional
PRED_OUT_JSON = os.path.join("docs", "data", "predictions.json")

# Use LAST_N recent games (pre-game rolling means)
LAST_N = 5

# Box-score stats we use (long -> numeric -> averaged)
STAT_FEATURES = [
    "totalYards","netPassingYards","rushingYards",
    "firstDowns","turnovers","sacks","tacklesForLoss",
    "thirdDownEff","fourthDownEff","kickingPoints"
]

# Ensemble weights
ELO_WEIGHT  = 0.60    # Elo probability weight
STAT_WEIGHT = 0.40    # Logistic (last-N diff) probability weight

# Elo params
ELO_START = 1500.0
ELO_K = 20.0
ELO_HFA = 55.0          # home field advantage in Elo points
ELO_MOV_SCALING = True  # margin-of-victory scaling

# =========================
# Team aliases (common names -> dataset names)
# Augmented by docs/input/aliases.json if present.
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
    # …(many more in your previous version; keep expanding as needed)
}

# =========================
# Helpers
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

def long_stats_to_wide(df_stats: pd.DataFrame) -> pd.DataFrame:
    keep = df_stats[df_stats["category"].isin(STAT_FEATURES)].copy()

    def numericize(cat, val):
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

    keep["stat_value_num"] = [numericize(c, v) for c, v in zip(keep["category"], keep["stat_value"])]
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

def team_rolling_means(wide_stats: pd.DataFrame, schedule: pd.DataFrame, n: int) -> pd.DataFrame:
    season_week = schedule[["game_id","season","week","neutral_site"]].copy()
    season_week["season"] = season_week["season"].apply(to_int)
    season_week["week"] = season_week["week"].apply(to_int)
    w = wide_stats.merge(season_week, on="game_id", how="left")
    w["season"] = w["season"].apply(to_int)
    w["week"] = w["week"].apply(to_int)
    w = w.sort_values(["team","season","week","game_id"]).reset_index(drop=True)

    # pre-game rolling mean (shifted by 1)
    for c in STAT_FEATURES:
        w[f"R{n}_{c}"] = w.groupby("team")[c].transform(
            lambda s: s.rolling(window=n, min_periods=1).mean().shift(1)
        )

    keep_cols = ["game_id","team","homeAway","season","week","neutral_site"] + [f"R{n}_{c}" for c in STAT_FEATURES]
    return w[keep_cols]

def build_examples_lastN(schedule: pd.DataFrame, wide_stats: pd.DataFrame, n: int):
    roll = team_rolling_means(wide_stats, schedule, n)
    home = roll[roll["homeAway"]=="home"].copy()
    away = roll[roll["homeAway"]=="away"].copy()

    home = home.rename(columns={f"R{n}_{c}": f"home_R{n}_{c}" for c in STAT_FEATURES})
    away = away.rename(columns={f"R{n}_{c}": f"away_R{n}_{c}" for c in STAT_FEATURES})

    base = schedule[["game_id","home_team","away_team","home_points","away_points","season","week","neutral_site"]].copy()
    merged = base.merge(home[["game_id","team"] + [f"home_R{n}_{c}" for c in STAT_FEATURES]],
                        left_on=["game_id","home_team"], right_on=["game_id","team"], how="left").drop(columns=["team"])
    merged = merged.merge(away[["game_id","team"] + [f"away_R{n}_{c}" for c in STAT_FEATURES]],
                          left_on=["game_id","away_team"], right_on=["game_id","team"], how="left").drop(columns=["team"])

    # diffs (home - away)
    diff_cols = []
    for c in STAT_FEATURES:
        hc, ac = f"home_R{n}_{c}", f"away_R{n}_{c}"
        dc = f"diff_R{n}_{c}"
        merged[dc] = merged[hc].fillna(0.0) - merged[ac].fillna(0.0)
        diff_cols.append(dc)

    merged["home_win"] = (merged["home_points"] > merged["away_points"]).astype(int)
    merged = merged.dropna(subset=["home_win"])
    return merged, diff_cols

# -------- Elo --------
def elo_expect(a: float, b: float) -> float:
    return 1.0 / (1.0 + 10 ** (-(a - b) / 400.0))

def mov_multiplier(point_diff: float, rating_diff: float) -> float:
    diff = abs(point_diff)
    if diff <= 0:
        return 1.0
    return math.log(diff + 1.0) * (2.2 / ((rating_diff * 0.001) + 2.2))

def train_elo(schedule: pd.DataFrame) -> Dict[str, float]:
    sched = schedule[["season","week","home_team","away_team","home_points","away_points","neutral_site"]].dropna().copy()
    sched["season"] = sched["season"].apply(to_int)
    sched["week"] = sched["week"].apply(to_int)
    sched = sched.sort_values(["season","week"]).reset_index(drop=True)

    R: Dict[str, float] = {}
    def get(team): return R.get(team, ELO_START)

    for _, row in sched.iterrows():
        h, a = row["home_team"], row["away_team"]
        hp, ap = float(row["home_points"]), float(row["away_points"])
        ra, rb = get(h), get(a)

        hfa = 0.0 if bool(row.get("neutral_site", False)) else ELO_HFA
        exp_h = elo_expect(ra + hfa, rb)
        exp_a = 1.0 - exp_h

        if hp == ap:
            score_h, score_a = 0.5, 0.5
            mov = 0.0
        else:
            score_h = 1.0 if hp > ap else 0.0
            score_a = 1.0 - score_h
            mov = abs(hp - ap)

        k = ELO_K
        if ELO_MOV_SCALING and mov > 0:
            k = ELO_K * mov_multiplier(mov, abs(ra - rb))

        R[h] = ra + k * (score_h - exp_h)
        R[a] = rb + k * (score_a - exp_a)

    return R

def prob_from_elo(elo: Dict[str,float], home: str, away: str, neutral: bool) -> float:
    rh = elo.get(home, ELO_START)
    ra = elo.get(away, ELO_START)
    hfa = 0.0 if neutral else ELO_HFA
    return elo_expect(rh + hfa, ra)

# -------- Train / Predict --------
def train_stats_model_lastN(schedule: pd.DataFrame, wide_stats: pd.DataFrame, n: int):
    examples, diff_cols = build_examples_lastN(schedule, wide_stats, n)
    train, test = train_test_split(examples, test_size=0.2, random_state=42, shuffle=True)
    X_train = train[diff_cols].values; y_train = train["home_win"].values
    X_test  = test[diff_cols].values;  y_test  = test["home_win"].values
    model = LogisticRegression(max_iter=400)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    # probability quality (nice to track)
    proba = model.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, proba)
    brier = brier_score_loss(y_test, proba)
    return model, acc, auc, brier, diff_cols

def latest_team_lastN_vectors(wide_stats: pd.DataFrame, schedule: pd.DataFrame, n: int) -> pd.DataFrame:
    roll = team_rolling_means(wide_stats, schedule, n)
    roll = roll.sort_values(["team","season","week","game_id"])
    rcols = [f"R{n}_{c}" for c in STAT_FEATURES]
    latest = roll.groupby("team").tail(1).set_index("team")[rcols + ["neutral_site"]].copy()
    return latest.fillna(0.0)

def predict_games(elo: Dict[str,float],
                  stats_model: LogisticRegression,
                  diff_cols: List[str],
                  team_lastN: pd.DataFrame,
                  games: List[Dict[str,str]],
                  n: int) -> Tuple[List[Dict], List[str]]:
    unknown = set()
    rows = []

    def team_vec(team: str):
        if team in team_lastN.index:
            return team_lastN.loc[team, [f"R{n}_{c}" for c in STAT_FEATURES]].values.astype(float)
        unknown.add(team)
        return np.zeros(len(STAT_FEATURES), dtype=float)

    for g in games:
        home, away = g["home"], g["away"]
        vh = team_vec(home)
        va = team_vec(away)
        diff = vh - va
        X = np.array([diff])  # matches training order
        p_stat = float(stats_model.predict_proba(X)[0,1])

        # For user-entered games we don't know neutral site -> assume false
        p_elo  = prob_from_elo(elo, home, away, neutral=False)

        p_home = float(ELO_WEIGHT * p_elo + STAT_WEIGHT * p_stat)
        rows.append({
            "home": home, "away": away,
            "home_prob": round(p_home, 4),
            "away_prob": round(1.0 - p_home, 4),
            "pick": home if p_home >= 0.5 else away
        })

    return rows, sorted(unknown)

def main():
    # Prefer local historic CSVs (if fetch_cfbd.py has been run)
    print("Loading schedule & team stats ...")
    schedule = load_csv_local_or_url(LOCAL_SCHEDULE, FALLBACK_SCHEDULE_URL).rename(columns=str.strip)
    stats    = load_csv_local_or_url(LOCAL_TEAM_STATS, FALLBACK_TEAM_STATS_URL).rename(columns=str.strip)

    # Canonical team list
    teams_in_dataset = set(pd.concat([schedule["home_team"], schedule["away_team"]]).dropna().unique())

    # Build wide stats
    wide = long_stats_to_wide(stats)

    # Train last-N stats model
    stat_model, acc, auc, brier, diff_cols = train_stats_model_lastN(schedule, wide, LAST_N)
    print(f"[STAT last{LAST_N}] acc={acc:.3f}, AUC={auc:.3f}, Brier={brier:.3f}")

    # Elo (neutral-site aware if present in schedule)
    elo_ratings = train_elo(schedule)
    print(f"[ELO] ratings for {len(elo_ratings)} teams.")

    # Latest per-team last-N vectors at prediction time
    team_lastN = latest_team_lastN_vectors(wide, schedule, LAST_N)

    # Parse games with aliases
    alias_map = load_alias_map()
    def safe_name(n):
        mapped = normalize_name(n, alias_map)
        return mapped if mapped in teams_in_dataset else n

    raw_games = parse_games_txt(INPUT_GAMES_TXT, alias_map)
    games = [{"home": safe_name(g["home"]), "away": safe_name(g["away"])} for g in raw_games]

    rows, unknown = predict_games(elo_ratings, stat_model, diff_cols, team_lastN, games, LAST_N)

    out = {
        "generated_at": dt.datetime.utcnow().replace(microsecond=0).isoformat()+"Z",
        "season": 0, "week": 0,
        "model": f"ensemble_last{LAST_N} (Elo {ELO_WEIGHT:.0%} + stats {STAT_WEIGHT:.0%})",
        "metric": {
            "test_accuracy": round(float(acc), 4),
            "auc": round(float(auc), 4),
            "brier": round(float(brier), 4)
        },
        "unknown_teams": unknown,
        "games": rows,
    }
    os.makedirs(os.path.dirname(PRED_OUT_JSON), exist_ok=True)
    with open(PRED_OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {PRED_OUT_JSON}")

if __name__ == "__main__":
    main()
