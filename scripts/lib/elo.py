# scripts/lib/elo.py
import math, pandas as pd, numpy as np

ELO_START = 1500.0
ELO_K_BASE = 20.0
ELO_K_EARLY = 32.0
ELO_HFA = 55.0
MEAN_REVERT = 0.30

def elo_expect(a: float, b: float) -> float:
    return 1.0 / (1.0 + 10 ** (-(a - b) / 400.0))

def mov_multiplier(point_diff: float, rating_diff: float) -> float:
    d = abs(point_diff)
    if d <= 0: return 1.0
    return math.log(d + 1.0) * (2.2 / ((rating_diff * 0.001) + 2.2))

def preseason_seed_from_talent(talent_df: pd.DataFrame):
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

def pregame_probs(schedule: pd.DataFrame, talent_df: pd.DataFrame) -> pd.DataFrame:
    sched = schedule.sort_values(["season","week","date","game_id"]).copy()
    seed = preseason_seed_from_talent(talent_df)
    ratings = {}
    cur_season = None
    rows = []
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
        rows.append({"game_id": r["game_id"], "elo_home_prob": p_home})
        if pd.notna(hp) and pd.notna(ap):
            score_home = 1.0 if hp > ap else (0.5 if hp == ap else 0.0)
            k = ELO_K_EARLY if week <= 4 else ELO_K_BASE
            movm = mov_multiplier(hp - ap, (ra + hfa) - rb)
            ratings[home] = ra + k * movm * (score_home - p_home)
            ratings[away] = rb + k * movm * ((1.0 - score_home) - (1.0 - p_home))
    return pd.DataFrame(rows)

def end_of_history_ratings(schedule: pd.DataFrame, talent_df: pd.DataFrame):
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
