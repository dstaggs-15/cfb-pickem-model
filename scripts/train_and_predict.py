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

# Features for the stat-diff model
STAT_FEATURES = [
    "totalYards","netPassingYards","rushingYards",
    "firstDowns","turnovers","sacks","tacklesForLoss",
    "thirdDownEff","fourthDownEff","kickingPoints"
]

# Ensemble weights (tune these)
ELO_WEIGHT  = 0.60    # weight for Elo probability
STAT_WEIGHT = 0.40    # weight for logistic-regression probability

# Elo params
ELO_START = 1500.0
ELO_K = 20.0
ELO_HFA = 55.0          # home field advantage in Elo points
ELO_MOV_SCALING = True  # margin-of-victory scaling

# =========================
# Team aliases (built-in) — augmented by docs/input/aliases.json if present
# Map common short names -> dataset names (as they appear in cfb_schedule.csv)
# =========================
BUILTIN_ALIASES = {
    # Your recent games / screenshot teams
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
    "pittsburgh": "Pittsburgh Panthers", "pitt": "Pittsburgh Panthers",

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

    "tulane green wave": "Tulane Green Wave", # in case raw names already match
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

def build_examples(schedule: pd.DataFrame, wide_stats: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    merged = schedule[["game_id","home_team","away_team","home_points","away_points","season","week"]].copy()
    home = wide_stats[wide_stats["homeAway"]=="home"].rename(columns={c:f"home_{c}" for c in STAT_FEATURES})
    away = wide_stats[wide_stats["homeAway"]=="away"].rename(columns={c:f"away_{c}" for c in STAT_FEATURES})

    merged = merged.merge(home[["game_id","team"]+[f"home_{c}" for c in STAT_FEATURES]],
                          left_on=["game_id","home_team"], right_on=["game_id","team"], how="left").drop(columns=["team"])
    merged = merged.merge(away[["game_id","team"]+[f"away_{c}" for c in STAT_FEATURES]],
                          left_on=["game_id","away_team"], right_on=["game_id","team"], how="left").drop(columns=["team"])

    for c in STAT_FEATURES:
        merged[f"diff_{c}"] = merged[f"home_{c}"] - merged[f"away_{c}"]

    merged["home_win"] = (merged["home_points"] > merged["away_points"]).astype(int)
    merged = merged.dropna(subset=["home_win"])

    feature_cols = [f"diff_{c}" for c in STAT_FEATURES]
    merged[feature_cols] = merged[feature_cols].fillna(0.0)
    return merged, feature_cols

# =========================
# Elo rating
# =========================
def elo_expect(a: float, b: float) -> float:
    return 1.0 / (1.0 + 10 ** (-(a - b) / 400.0))

def mov_multiplier(point_diff: float, rating_diff: float) -> float:
    diff = abs(point_diff)
    if diff <= 0:
        return 1.0
    # Common chess/football Elo MoV scaling
    return math.log(diff + 1.0) * (2.2 / ((rating_diff * 0.001) + 2.2))

def train_elo(schedule: pd.DataFrame) -> Dict[str, float]:
    # Sort by season, then week to ensure temporal order
    sched = schedule[["season","week","home_team","away_team","home_points","away_points"]].dropna().copy()
    sched["week"] = pd.to_numeric(sched["week"], errors="coerce").fillna(0).astype(int)
    sched = sched.sort_values(["season","week"]).reset_index(drop=True)

    R: Dict[str, float] = {}
    def get(team): return R.get(team, ELO_START)

    for _, row in sched.iterrows():
        h, a = row["home_team"], row["away_team"]
        hp, ap = float(row["home_points"]), float(row["away_points"])
        ra, rb = get(h), get(a)
        # expected with home field advantage
        exp_h = elo_expect(ra + ELO_HFA, rb)
        exp_a = 1.0 - exp_h
        # actual scores
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
# Team feature vectors for predictions
# =========================
def team_recent_averages(wide_stats: pd.DataFrame, schedule: pd.DataFrame, years_back: int = 5) -> pd.DataFrame:
    # join season onto wide_stats to compute season-wise means
    season_map = schedule[["game_id","season"]]
    w = wide_stats.merge(season_map, on="game_id", how="left")
    # limit to recent seasons if available
    if "season" in w.columns and w["season"].notna().any():
        max_season = int(pd.to_numeric(w["season"], errors="coerce").max())
        min_keep = max_season - years_back + 1
        w = w[pd.to_numeric(w["season"], errors="coerce").fillna(0).astype(int) >= (min_keep if not math.isnan(min_keep) else 0)]
    # team-season means, then weighted by recency
    w["season_num"] = pd.to_numeric(w["season"], errors="ignore")
    grp = w.groupby(["team","season"])[STAT_FEATURES].mean().reset_index()

    # recency weights: newer season = higher weight (1..years_back)
    if "season" in grp.columns and grp["season"].notna().any():
        base = pd.to_numeric(grp["season"], errors="coerce")
        if base.notna().any():
            min_s = int(base.min())
            grp["weight"] = 1 + (pd.to_numeric(grp["season"], errors="coerce") - min_s)
        else:
            grp["weight"] = 1.0
    else:
        grp["weight"] = 1.0

    for c in STAT_FEATURES:
        grp[c] = grp[c].fillna(0.0)

    # weighted average per team
    weighted = (grp
                .assign(w=grp["weight"])
                .groupby("team")
                .apply(lambda d: pd.Series({c: np.average(d[c].values, weights=d["w"].values)
                                            for c in STAT_FEATURES}))
                .reset_index())
    return weighted.set_index("team")

# =========================
# Training (stats model) + Prediction
# =========================
def train_stats_model(schedule: pd.DataFrame, wide_stats: pd.DataFrame):
    examples, feature_cols = build_examples(schedule, wide_stats)
    train, test = train_test_split(examples, test_size=0.2, random_state=42, shuffle=True)
    X_train = train[feature_cols].values; y_train = train["home_win"].values
    X_test  = test[feature_cols].values;  y_test  = test["home_win"].values
    model = LogisticRegression(max_iter=300)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc, feature_cols

def predict_games(elo: Dict[str,float],
                  stats_model: LogisticRegression,
                  feature_cols: List[str],
                  team_feature_df: pd.DataFrame,
                  games: List[Dict[str,str]]) -> Tuple[List[Dict], List[str]]:
    unknown = set()
    rows = []

    for g in games:
        home, away = g["home"], g["away"]

        def vec(team):
            if team in team_feature_df.index:
                return team_feature_df.loc[team, STAT_FEATURES].values.astype(float)
            unknown.add(team)
            return np.zeros(len(STAT_FEATURES), dtype=float)

        v_home = vec(home)
        v_away = vec(away)
        diff = v_home - v_away

        X = np.array([[diff[i] for i,_ in enumerate(STAT_FEATURES)]])
        p_stat = float(stats_model.predict_proba(X)[0,1])

        p_elo = prob_from_elo(elo, home, away)

        p_home = float(ELO_WEIGHT * p_elo + STAT_WEIGHT * p_stat)
        pick = home if p_home >= 0.5 else away

        rows.append({
            "home": home, "away": away,
            "home_prob": round(p_home, 4),
            "away_prob": round(1.0 - p_home, 4),
            "pick": pick
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

    # Normalize team names inside schedule to be consistent keys
    # (Some datasets already use the long names used above.)
    # We'll treat whatever appears in schedule['home_team']/['away_team'] as canonical.
    teams_in_dataset = set(pd.concat([schedule["home_team"], schedule["away_team"]]).dropna().unique())

    # Prepare wide stats
    wide = long_stats_to_wide(stats)

    # Train stat model
    stat_model, stat_acc, feature_cols = train_stats_model(schedule, wide)
    print(f"[STAT] Test accuracy: {stat_acc:.4f}")

    # Train Elo on chronological games
    elo_ratings = train_elo(schedule)
    print(f"[ELO] Trained ratings for {len(elo_ratings)} teams.")

    # Aggregate recent team features for prediction
    team_feats = team_recent_averages(wide, schedule, years_back=5)

    # Parse games.txt with alias mapping -> dataset names
    alias_map = load_alias_map()

    # Safety: if alias points to a name not in dataset, try pass-through
    def safe_name(n):
        mapped = normalize_name(n, alias_map)
        return mapped if mapped in teams_in_dataset else n

    raw_games = parse_games_txt(INPUT_GAMES_TXT, alias_map)
    safe_games = [{"home": safe_name(g["home"]), "away": safe_name(g["away"])} for g in raw_games]

    # Predict
    rows, unknown = predict_games(elo_ratings, stat_model, feature_cols, team_feats, safe_games)

    out = {
        "generated_at": dt.datetime.utcnow().replace(microsecond=0).isoformat()+"Z",
        "season": 0, "week": 0,
        "model": "ensemble_v1 (elo {:.0%} / stats {:.0%})".format(ELO_WEIGHT, STAT_WEIGHT),
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
