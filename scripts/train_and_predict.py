import json, os, re
import datetime as dt
from io import StringIO

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

RAW_BASE = "https://raw.githubusercontent.com/moneyball-ab/cfb-data/master/csv"
SCHEDULE_URL = f"{RAW_BASE}/cfb_schedule.csv"
TEAM_STATS_URL = f"{RAW_BASE}/cfb_game_team_stats.csv"

GAMES_TXT = os.path.join("docs", "input", "games.txt")
PRED_OUT   = os.path.join("docs", "data", "predictions.json")

STAT_FEATURES = [
    "totalYards","netPassingYards","rushingYards",
    "firstDowns","turnovers","sacks","tacklesForLoss",
    "thirdDownEff","fourthDownEff","kickingPoints"
]

GAME_PATTERNS = [
    re.compile(r"^\s*(?P<away>.+?)\s*@\s*(?P<home>.+?)\s*$", re.I),      # Away @ Home
    re.compile(r"^\s*(?P<home>.+?)\s*vs\.?\s*(?P<away>.+?)\s*$", re.I),  # Home vs Away
    re.compile(r"^\s*(?P<home>.+?)\s*,\s*(?P<away>.+?)\s*$", re.I),      # Home, Away
]

def _download_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text))

def _long_stats_to_wide(df_stats: pd.DataFrame) -> pd.DataFrame:
    keep = df_stats[df_stats["category"].isin(STAT_FEATURES)].copy()

    def numericize(cat, val):
        if isinstance(val, str) and "-" in val and cat in ["thirdDownEff","fourthDownEff",
                                                           "completionAttempts","totalPenaltiesYards"]:
            try:
                a, b = val.split("-")
                a = float(a); b = float(b)
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

def _build_examples(schedule: pd.DataFrame, wide_stats: pd.DataFrame):
    merged = schedule[["game_id","home_team","away_team","home_points","away_points"]].copy()
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

def train_model():
    print("Downloading data...")
    schedule = _download_csv(SCHEDULE_URL).rename(columns=str.strip)
    stats    = _download_csv(TEAM_STATS_URL).rename(columns=str.strip)

    wide = _long_stats_to_wide(stats)
    examples, feature_cols = _build_examples(schedule, wide)

    train, test = train_test_split(examples, test_size=0.2, random_state=42, shuffle=True)
    X_train = train[feature_cols].values; y_train = train["home_win"].values
    X_test  = test[feature_cols].values;  y_test  = test["home_win"].values

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Test accuracy: {acc:.3f}  (n={len(y_test)})")
    return model, acc, feature_cols, wide

def _team_means(wide_stats: pd.DataFrame) -> pd.DataFrame:
    feats = ["team","homeAway"] + STAT_FEATURES
    df = wide_stats[feats].copy()
    team_means = df.groupby("team")[STAT_FEATURES].mean().reset_index()
    return team_means.set_index("team")

def read_games_from_txt(path=GAMES_TXT):
    games = []
    if not os.path.exists(path):
        print(f"[WARN] {path} not found. No games parsed.")
        return games
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"): 
                continue
            matched = None
            for pat in GAME_PATTERNS:
                m = pat.match(line)
                if m: matched = m.groupdict(); break
            if not matched:
                print(f"[SKIP] Unrecognized line: {line}")
                continue
            home = " ".join(matched["home"].split())
            away = " ".join(matched["away"].split())
            games.append({"home": home, "away": away})
    return games

def predict_and_write(model, feature_cols, wide_stats, out_json_path, season=None, week=None, test_acc=0.0):
    team_avgs = _team_means(wide_stats)
    rows = []
    unknown = set()
    for g in read_games_from_txt():
        home, away = g["home"], g["away"]

        def vec(team):
            if team in team_avgs.index:
                return team_avgs.loc[team, STAT_FEATURES].values.astype(float)
            else:
                unknown.add(team)
                return np.zeros(len(STAT_FEATURES), dtype=float)

        diff = vec(home) - vec(away)
        feats = np.array([[diff[i] for i, _ in enumerate(STAT_FEATURES)]])
        p_home = float(model.predict_proba(feats)[0,1])
        rows.append({
            "home": home, "away": away,
            "home_prob": round(p_home, 4),
            "away_prob": round(1.0 - p_home, 4),
            "pick": home if p_home >= 0.5 else away
        })

    out = {
        "generated_at": dt.datetime.utcnow().replace(microsecond=0).isoformat()+"Z",
        "season": season or 0, "week": week or 0,
        "model": "logreg_v1",
        "metric": {"test_accuracy": round(float(test_acc), 4)},
        "unknown_teams": sorted(list(unknown)),
        "games": rows
    }
    os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
    with open(out_json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_json_path}")
    if unknown:
        print("[WARN] Unknown team names:", ", ".join(sorted(unknown)))

def main():
    model, acc, feature_cols, wide = train_model()
    predict_and_write(model, feature_cols, wide, PRED_OUT, test_acc=acc)

if __name__ == "__main__":
    main()
