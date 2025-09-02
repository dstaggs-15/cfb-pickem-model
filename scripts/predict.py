# scripts/predict.py
#!/usr/bin/env python3
import os, json, datetime as dt
import numpy as np, pandas as pd
from joblib import load

from lib.io_utils import load_csv_local_or_url, save_json
from lib.parsing import ensure_schedule_columns, load_alias_map, parse_games_txt
from lib.rolling import long_stats_to_wide, build_sidewise_rollups, latest_per_team, STAT_FEATURES
from lib.market import median_lines, market_prob
from lib.elo import end_of_history_ratings, ELO_HFA

LOCAL_DIR = "data/raw/cfbd"
LOCAL_SCHEDULE = f"{LOCAL_DIR}/cfb_schedule.csv"
LOCAL_TEAM_STATS = f"{LOCAL_DIR}/cfb_game_team_stats.csv"
LOCAL_LINES = f"{LOCAL_DIR}/cfb_lines.csv"
LOCAL_TEAMS = f"{LOCAL_DIR}/cfbd_teams.csv"
LOCAL_TALENT = f"{LOCAL_DIR}/cfbd_talent.csv"

RAW_BASE = "https://raw.githubusercontent.com/moneyball-ab/cfb-data/master/csv"
FALLBACK_SCHEDULE_URL = f"{RAW_BASE}/cfb_schedule.csv"
FALLBACK_TEAM_STATS_URL = f"{RAW_BASE}/cfb_game_team_stats.csv"

INPUT_GAMES_TXT = "docs/input/games.txt"
INPUT_ALIASES_JSON = "docs/input/aliases.json"
INPUT_LINES_CSV = "docs/input/lines.csv"

MODEL_PATH = "data/derived/model.joblib"
META_JSON = "docs/data/train_meta.json"
PRED_OUT_JSON = "docs/data/predictions.json"

def main():
    meta=json.load(open(META_JSON,"r"))
    LAST_N=int(meta["last_n"])
    feature_cols=list(meta["features"])
    market_params=meta["market_params"]
    model=load(MODEL_PATH)

    alias_map = load_alias_map(INPUT_ALIASES_JSON)

    schedule = load_csv_local_or_url(LOCAL_SCHEDULE, FALLBACK_SCHEDULE_URL)
    schedule = ensure_schedule_columns(schedule)
    team_stats = load_csv_local_or_url(LOCAL_TEAM_STATS, FALLBACK_TEAM_STATS_URL)
    wide = long_stats_to_wide(team_stats)
    lines_df = pd.read_csv(INPUT_LINES_CSV) if os.path.exists(INPUT_LINES_CSV) else pd.DataFrame()

    # Build rollups once
    home_roll, away_roll = build_sidewise_rollups(schedule, wide, LAST_N)
    last_home = latest_per_team(home_roll, "home", LAST_N)
    last_away = latest_per_team(away_roll, "away", LAST_N)

    # Neutral hint from schedule (this season)
    sched_now = schedule.copy()
    season_max = int(pd.to_numeric(sched_now["season"], errors="coerce").max()) if "season" in sched_now.columns else None
    if season_max is not None:
        sched_now = sched_now[sched_now["season"]==season_max]
    pair_neutral = {}
    if {"home_team","away_team","neutral_site"}.issubset(sched_now.columns):
        tmp = sched_now.sort_values(["week","date"]).drop_duplicates(subset=["home_team","away_team"], keep="last")
        for _, r in tmp.iterrows():
            pair_neutral[(str(r["home_team"]), str(r["away_team"]))] = bool(r["neutral_site"])

    # Current Elo ratings
    talent_df = pd.read_csv(LOCAL_TALENT) if os.path.exists(LOCAL_TALENT) else pd.DataFrame()
    current_ratings = end_of_history_ratings(schedule, talent_df)

    # Parse input games + manual lines
    raw_games = parse_games_txt(INPUT_GAMES_TXT, alias_map)
    man = lines_df.copy()
    man.columns = [c.strip().lower() for c in man.columns]
    if not {"home","away","spread","over_under"}.issubset(set(man.columns)):
        man = pd.DataFrame()

    def lines_for(home, away):
        if man.empty: return (np.nan, np.nan)
        r = man[(man["home"]==home) & (man["away"]==away)]
        if r.empty: return (np.nan, np.nan)
        return float(r.iloc[0]["spread"]), float(r.iloc[0]["over_under"])

    rows=[]
    for g in raw_games:
        home=str(g["home"]); away=str(g["away"]); neutral=bool(g.get("neutral", False)) or pair_neutral.get((home,away), False)

        # counts
        feats = {}
        feats[f"home_R{LAST_N}_count"] = float(last_home.loc[home][f"home_R{LAST_N}_count"]) if home in last_home.index else 0.0
        feats[f"away_R{LAST_N}_count"] = float(last_away.loc[away][f"away_R{LAST_N}_count"]) if away in last_away.index else 0.0

        # diffs
        for c in STAT_FEATURES:
            hv = float(last_home.loc[home][f"home_R{LAST_N}_{c}"]) if (home in last_home.index and pd.notna(last_home.loc[home][f"home_R{LAST_N}_{c}"])) else np.nan
            av = float(last_away.loc[away][f"away_R{LAST_N}_{c}"]) if (away in last_away.index and pd.notna(last_away.loc[away][f"away_R{LAST_N}_{c}"])) else np.nan
            feats[f"diff_R{LAST_N}_{c}"] = hv - av if (pd.notna(hv) and pd.notna(av)) else np.nan

        feats["rest_diff"]=0.0; feats["shortweek_diff"]=0.0; feats["bye_diff"]=0.0
        feats["neutral_site"]=1.0 if neutral else 0.0
        feats["is_postseason"]=0.0

        sp, ou = lines_for(home, away)
        feats["spread_home"] = sp if pd.notna(sp) else np.nan
        feats["over_under"]  = ou if pd.notna(ou) else np.nan
        feats["market_home_prob"] = market_prob(feats["spread_home"], market_params["a"], market_params["b"]) if pd.notna(feats["spread_home"]) else np.nan

        ra = current_ratings.get(home, 1500.0)
        rb = current_ratings.get(away, 1500.0)
        hfa = 0.0 if neutral else ELO_HFA
        # Elo expect
        feats["elo_home_prob"] = 1.0 / (1.0 + 10 ** (-( (ra+hfa) - rb) / 400.0))

        # align features
        X = pd.DataFrame([{k: feats.get(k, np.nan) for k in feature_cols}])
        p_home = float(model.predict_proba(X)[0,1])

        rows.append({
            "home": home, "away": away,
            "home_prob": round(p_home,4),
            "away_prob": round(1.0-p_home,4),
            "pick": home if p_home>=0.5 else away,
            "neutral": bool(neutral),
            "spread_home": None if pd.isna(sp) else float(sp),
            "over_under": None if pd.isna(ou) else float(ou),
            "p_elo": round(float(feats["elo_home_prob"]),4),
            "p_market": None if pd.isna(feats["market_home_prob"]) else round(float(feats["market_home_prob"]),4),
        })

    out={
        "generated": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": f"HGB + calib, last{LAST_N} side-split; Elo/Market as feats",
        "games": rows,
        "unknown_teams": [],  # now handled via aliases + schedule names
    }
    save_json(PRED_OUT_JSON, out)
    print(f"Wrote {PRED_OUT_JSON}")

if __name__ == "__main__":
    main()
