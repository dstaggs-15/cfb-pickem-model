import json, os, re, math
import datetime as dt
from io import StringIO
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# =========================
# Config
# =========================
RAW_BASE = "https://raw.githubusercontent.com/moneyball-ab/cfb-data/master/csv"
SCHEDULE_URL = f"{RAW_BASE}/cfb_schedule.csv"
TEAM_STATS_URL = f"{RAW_BASE}/cfb_game_team_stats.csv"

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
    # Common teams / ones you’ve used
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

    # Broader coverage (Power + many G5)
    "alabama": "Alabama Crimson Tide", "auburn": "Auburn Tigers",
    "arkansas": "Arkansas Razorbacks", "georgia": "Georgia Bulldogs",
    "florida": "Florida Gators", "tennessee": "Tennessee Volunteers",
    "kentucky": "Kentucky Wildcats", "ole miss": "Ole Miss Rebels",
    "mississippi": "Ole Miss Rebels", "missouri": "Missouri Tigers",
    "vanderbilt": "Vanderbilt Commodores", "texas a&m": "Texas A&M Aggies",
    "texas a & m": "Texas A&M Aggies",

    "michigan": "Michigan Wolverines", "penn state": "Penn State Nittany Lions",
    "michigan state": "Michigan State Spartans", "maryland": "Maryland Terrapins",
    "rutgers": "Rutgers Scarlet Knights", "indiana": "Indiana Hoosiers",
    "illinois": "Illinois Fighting Illini", "iowa": "Iowa Hawkeyes",
    "minnesota": "Minnesota Golden Gophers", "nebraska": "Nebraska Cornhuskers",
    "purdue": "Purdue Boilermakers", "wisconsin": "Wisconsin Badgers",

    "oklahoma": "Oklahoma Sooners", "oklahoma state": "Oklahoma State Cowboys",
    "kansas": "Kansas Jayhawks", "kansas state": "Kansas State Wildcats",
    "baylor": "Baylor Bears", "tcu": "TCU Horned Frogs",
    "texas tech": "Texas Tech Red Raiders", "iowa state": "Iowa State Cyclones",
    "west virginia": "West Virginia Mountaineers", "ucf": "UCF Knights",
    "cincinnati": "Cincinnati Bearcats", "houston": "Houston Cougars", "byu": "BYU Cougars",

    "oregon state": "Oregon State Beavers", "washington": "Washington Huskies",
    "washington state": "Washington State Cougars", "usc": "USC Trojans",
    "stanford": "Stanford Cardinal", "cal": "California Golden Bears",
    "arizona state": "Arizona State Sun Devils",

    "florida state": "Florida State Seminoles", "duke": "Duke Blue Devils",
    "north carolina": "North Carolina Tar Heels", "nc state": "NC State Wolfpack",
    "wake forest": "Wake Forest Demon Deacons", "virginia tech": "Virginia Tech Hokies",
    "boston college": "Boston College Eagles", "syracuse": "Syracuse Orange",
    "louisville": "Louisville Cardinals", "georgia tech": "Georgia Tech Yellow Jackets",
    "pitt": "Pittsburgh Panthers", "pittsburgh": "Pittsburgh Panthers",

    "boise state": "Boise State Broncos", "san diego state": "San Diego State Aztecs",
    "san jose state": "San Jose State Spartans", "air force": "Air Force Falcons",
    "colorado state": "Colorado State Rams", "wyoming": "Wyoming Cowboys",
    "unlv": "UNLV Rebels", "nevada": "Nevada Wolf Pack", "new mexico": "New Mexico Lobos",

    "memphis": "Memphis Tigers", "tulsa": "Tulsa Golden Hurricane",
    "utsa": "UTSA Roadrunners", "north texas": "North Texas Mean Green",
    "rice": "Rice Owls", "smu": "SMU Mustangs", "navy": "Navy Midshipmen",
    "army": "Army Black Knights", "temple": "Temple Owls", "uab": "UAB Blazers",
    "charlotte": "Charlotte 49ers", "fau": "Florida Atlantic Owls",
    "fiu": "Florida International Panthers", "wku": "Western Kentucky Hilltoppers",
    "middle tennessee": "Middle Tennessee Blue Raiders", "louisiana tech": "Louisiana Tech Bulldogs",

    "appalachian state": "Appalachian State Mountaineers", "coastal carolina": "Coastal Carolina Chanticleers",
    "georgia state": "Georgia State Panthers", "james madison": "James Madison Dukes",
    "marshall": "Marshall Thundering Herd", "old dominion": "Old Dominion Monarchs",
    "south alabama": "South Alabama Jaguars", "southern miss": "Southern Miss Golden Eagles",
    "troy": "Troy Trojans", "louisiana": "Louisiana Ragin' Cajuns",
    "ulm": "Louisiana-Monroe Warhawks", "texas state": "Texas State Bobcats",
    "arkansas state": "Arkansas State Red Wolves",
}

# =========================
# Helpers
# =========================
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

def download_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text))

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

def to_numeric(x, default=0):
    try:
        return int(float(x))
    except Exception:
        return default

# ---- Build rolling (pre-game) last-N means for each team & game
def team_rolling_means(wide_stats: pd.DataFrame, schedule: pd.DataFrame, n: int) -> pd.DataFrame:
    season_week = schedule[["game_id","season","week"]].copy()
    season_week["season"] = season_week["season"].apply(lambda v: to_numeric(v, 0))
    season_week["week"] = season_week["week"].apply(lambda v: to_numeric(v, 0))
    w = wide_stats.merge(season_week, on="game_id", how="left")
    w["season"] = w["season"].apply(lambda v: to_numeric(v, 0))
    w["week"] = w["week"].apply(lambda v: to_numeric(v, 0))

    w = w.sort_values(["team","season","week","game_id"]).reset_index(drop=True)

    roll_cols = [f"R{n}_{c}" for c in STAT_FEATURES]
    for c in STAT_FEATURES:
        # rolling window per team, shifted so it's pre-game (no leakage)
        w[f"R{n}_{c}"] = w.groupby("team")[c].transform(
            lambda s: s.rolling(window=n, min_periods=1).mean().shift(1)
        )

    keep_cols = ["game_id","team","homeAway"] + roll_cols
    return w[keep_cols]

def build_examples_lastN(schedule: pd.DataFrame, wide_stats: pd.DataFrame, n: int):
    roll = team_rolling_means(wide_stats, schedule, n)

    # Join pre-game rolling means for home and away teams for each game
    home = roll[roll["homeAway"]=="home"].copy()
    away = roll[roll["homeAway"]=="away"].copy()

    home = home.rename(columns={f"R{n}_{c}": f"home_R{n}_{c}" for c in STAT_FEATURES})
    away = away.rename(columns={f"R{n}_{c}": f"away_R{n}_{c}" for c in STAT_FEATURES})

    base = schedule[["game_id","home_team","away_team","home_points","away_points","season","week"]].copy()
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

# =========================
# Elo rating (chronological)
# =========================
def elo_expect(a: float, b: float) -> float:
    return 1.0 / (1.0 + 10 ** (-(a - b) / 400.0))

def mov_multiplier(point_diff: float, rating_diff: float) -> float:
    diff = abs(point_diff)
    if diff <= 0:
        return 1.0
    return math.log(diff + 1.0) * (2.2 / ((rating_diff * 0.001) + 2.2))

def train_elo(schedule: pd.DataFrame) -> Dict[str, float]:
    sched = schedule[["season","week","home_team","away_team","home_points","away_points"]].dropna().copy()
    sched["season"] = sched["season"].apply(lambda v: to_numeric(v, 0))
    sched["week"] = sched["week"].apply(lambda v: to_numeric(v, 0))
    sched = sched.sort_values(["season","week"]).reset_index(drop=True)

    R: Dict[str, float] = {}
    def get(team): return R.get(team, ELO_START)

    for _, row in sched.iterrows():
        h, a = row["home_team"], row["away_team"]
        hp, ap = float(row["home_points"]), float(row["away_points"])
        ra, rb = get(h), get(a)
        exp_h = elo_expect(ra + ELO_HFA, rb)
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

def prob_from_elo(elo: Dict[str,float], home: str, away: str) -> float:
    rh = elo.get(home, ELO_START)
    ra = elo.get(away, ELO_START)
    return elo_expect(rh + ELO_HFA, ra)

# =========================
# Train stats model on last-N diffs; predict with last-N team means
# =========================
def train_stats_model_lastN(schedule: pd.DataFrame, wide_stats: pd.DataFrame, n: int):
    examples, diff_cols = build_examples_lastN(schedule, wide_stats, n)
    train, test = train_test_split(examples, test_size=0.2, random_state=42, shuffle=True)
    X_train = train[diff_cols].values; y_train = train["home_win"].values
    X_test  = test[diff_cols].values;  y_test  = test["home_win"].values
    model = LogisticRegression(max_iter=400)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc, diff_cols

def latest_team_lastN_vectors(wide_stats: pd.DataFrame, schedule: pd.DataFrame, n: int) -> pd.DataFrame:
    # Build per-team pregame rolling means, then keep the latest available per team
    roll = team_rolling_means(wide_stats, schedule, n)
    # take last non-null row for each team
    roll = roll.sort_values(["team","game_id"])
    rcols = [f"R{n}_{c}" for c in STAT_FEATURES]
    latest = roll.groupby("team").tail(1).set_index("team")[rcols].copy()
    # fill any remaining NaNs with 0
    latest = latest.fillna(0.0)
    return latest

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
        X = np.array([diff])  # same order as trained: diff_R{n}_*
        p_stat = float(stats_model.predict_proba(X)[0,1])
        p_elo  = prob_from_elo(elo, home, away)
        p_home = float(ELO_WEIGHT * p_elo + STAT_WEIGHT * p_stat)
        rows.append({
            "home": home, "away": away,
            "home_prob": round(p_home, 4),
            "away_prob": round(1.0 - p_home, 4),
            "pick": home if p_home >= 0.5 else away
        })

    return rows, sorted(unknown)

# =========================
# Main
# =========================
def main():
    # Load data
    print("Downloading schedule & team stats ...")
    schedule = download_csv(SCHEDULE_URL).rename(columns=str.strip)
    stats    = download_csv(TEAM_STATS_URL).rename(columns=str.strip)

    # Canonical team list from schedule file (what the aliases must map into)
    teams_in_dataset = set(pd.concat([schedule["home_team"], schedule["away_team"]]).dropna().unique())

    # Prepare wide stats (game/team rows with numeric columns)
    wide = long_stats_to_wide(stats)

    # Train last-N stats model
    stat_model, stat_acc, diff_cols = train_stats_model_lastN(schedule, wide, LAST_N)
    print(f"[STAT last{LAST_N}] Test accuracy: {stat_acc:.4f}")

    # Train Elo
    elo_ratings = train_elo(schedule)
    print(f"[ELO] Trained ratings for {len(elo_ratings)} teams.")

    # Latest per-team last-N vectors for prediction time
    team_lastN = latest_team_lastN_vectors(wide, schedule, LAST_N)

    # Parse games with alias mapping
    alias_map = load_alias_map()

    def safe_name(n):
        mapped = normalize_name(n, alias_map)
        return mapped if mapped in teams_in_dataset else n

    raw_games = parse_games_txt(INPUT_GAMES_TXT, alias_map)
    games = [{"home": safe_name(g["home"]), "away": safe_name(g["away"])} for g in raw_games]

    # Predict (ensemble Elo + last-N stats)
    rows, unknown = predict_games(elo_ratings, stat_model, diff_cols, team_lastN, games, LAST_N)

    out = {
        "generated_at": dt.datetime.utcnow().replace(microsecond=0).isoformat()+"Z",
        "season": 0, "week": 0,
        "model": f"ensemble_v2 (Elo {ELO_WEIGHT:.0%} + last{LAST_N} stats {STAT_WEIGHT:.0%})",
        "metric": {"test_accuracy": round(float(stat_acc), 4)},
        "unknown_teams": unknown,
        "games": rows,
    }
    os.makedirs(os.path.dirname(PRED_OUT_JSON), exist_ok=True)
    with open(PRED_OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {PRED_OUT_JSON}")

if __name__ == "__main__":
    main()
