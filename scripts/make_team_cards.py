#!/usr/bin/env python3
import os, json
import pandas as pd
from io import StringIO
import requests
import numpy as np

# Inputs
LOCAL_DIR = "data/raw/cfbd"
LOCAL_SCHEDULE = f"{LOCAL_DIR}/cfb_schedule.csv"
LOCAL_TEAM_STATS = f"{LOCAL_DIR}/cfb_game_team_stats.csv"

RAW_BASE = "https://raw.githubusercontent.com/moneyball-ab/cfb-data/master/csv"
FALLBACK_SCHEDULE_URL = f"{RAW_BASE}/cfb_schedule.csv"
FALLBACK_TEAM_STATS_URL = f"{RAW_BASE}/cfb_game_team_stats.csv"

OUT_JSON = "docs/data/team_summaries.json"

SEASON_MIN_GAMES = 3  # don’t rank teams with fewer than this many games
STAT_CATS = ["totalYards","thirdDownEff","turnovers","sacks","tacklesForLoss"]

def load_csv_local_or_url(local_path: str, fallback_url: str) -> pd.DataFrame:
    if os.path.exists(local_path):
        return pd.read_csv(local_path)
    r = requests.get(fallback_url, timeout=60)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text))

def to_num(x):
    try: return float(x)
    except: return np.nan

def parse_ratio(x):
    # format "5-12" → 5/12
    if isinstance(x, str) and "-" in x:
        a,b = x.split("-", 1)
        try:
            a = float(a); b = float(b)
            return a / b if b else np.nan
        except:
            return np.nan
    return to_num(x)

def pct_rank(series: pd.Series, asc: bool) -> pd.Series:
    # percentile rank (1 best)
    return 1.0 - series.rank(pct=True, ascending=asc)

def main():
    schedule = load_csv_local_or_url(LOCAL_SCHEDULE, FALLBACK_SCHEDULE_URL)
    stats = load_csv_local_or_url(LOCAL_TEAM_STATS, FALLBACK_TEAM_STATS_URL)

    # latest season
    schedule["season"] = pd.to_numeric(schedule["season"], errors="coerce")
    latest = int(schedule["season"].dropna().max())
    sch = schedule[schedule["season"]==latest].copy()

    # Build per-team points for/against from schedule
    # Two rows per game: one for home team, one for away team
    home_rows = sch[["game_id","home_team","home_points","away_points"]].rename(
        columns={"home_team":"team","home_points":"points_for","away_points":"points_against"})
    away_rows = sch[["game_id","away_team","away_points","home_points"]].rename(
        columns={"away_team":"team","away_points":"points_for","home_points":"points_against"})
    points_long = pd.concat([home_rows, away_rows], ignore_index=True)
    points_long["points_for"] = pd.to_numeric(points_long["points_for"], errors="coerce")
    points_long["points_against"] = pd.to_numeric(points_long["points_against"], errors="coerce")

    # Offensive team stats (per game)
    keep = stats[stats["category"].isin(STAT_CATS)].copy()
    # numericize
    keep["val"] = [
        parse_ratio(v) if c in ["thirdDownEff"] else to_num(v)
        for c, v in zip(keep["category"], keep["stat_value"])
    ]
    # For defensive allowed stats, join opponent's value by game_id
    off = keep.pivot_table(index=["game_id","team"], columns="category", values="val", aggfunc="mean").reset_index()
    # Build opponent map by game: for each game, there are two teams; self-merge
    opp = off.merge(off, on="game_id", suffixes=("", "_opp"))
    opp = opp[opp["team"] != opp["team_opp"]].copy()
    # Drop the duplicate mirrored pair by sorting team names
    opp["pair_key"] = opp.apply(lambda r: "|".join(sorted([r["team"], r["team_opp"]])), axis=1)
    opp = opp.sort_values(["pair_key","team"]).drop_duplicates(subset=["pair_key","team"])

    # Aggregate per team
    # offense
    off_agg = off.groupby("team").agg({
        "totalYards":"mean",
        "thirdDownEff":"mean",
        "turnovers":"mean",
        "sacks":"mean",
        "tacklesForLoss":"mean",
    }).rename(columns={
        "totalYards":"off_ypg",
        "thirdDownEff":"off_3rd_pct",
        "turnovers":"off_to_g",
        "sacks":"off_sacks_g",
        "tacklesForLoss":"off_tfl_g",
    })
    # defense (allowed)
    def_agg = opp.groupby("team").agg({
        "totalYards_opp":"mean",
        "thirdDownEff_opp":"mean",
        "turnovers_opp":"mean",
        "sacks_opp":"mean",
        "tacklesForLoss_opp":"mean",
    }).rename(columns={
        "totalYards_opp":"def_yapg",
        "thirdDownEff_opp":"def_3rd_pct",
        "turnovers_opp":"def_to_g_allowed",
        "sacks_opp":"def_sacks_allowed_g",
        "tacklesForLoss_opp":"def_tfl_allowed_g",
    })

    # Points per game
    p_agg = points_long.groupby("team").agg({
        "points_for":"mean",
        "points_against":"mean",
        "game_id":"count"
    }).rename(columns={"points_for":"ppg","points_against":"papg","game_id":"games"})

    # Combine
    teams = sorted(set(list(off_agg.index) + list(def_agg.index) + list(p_agg.index)))
    df = pd.DataFrame(index=teams).join([off_agg, def_agg, p_agg], how="left")
    df["season"] = latest
    # Filter to real samples
    df = df[df["games"].fillna(0) >= SEASON_MIN_GAMES].copy()

    # Ranks: offense higher is better, defense lower is better
    df["rank_off_ppg_pct"] = pct_rank(df["ppg"], asc=False)
    df["rank_off_ypg_pct"] = pct_rank(df["off_ypg"], asc=False)
    df["rank_off_3rd_pct"] = pct_rank(df["off_3rd_pct"], asc=False)

    df["rank_def_papg_pct"] = pct_rank(df["papg"], asc=True)
    df["rank_def_yapg_pct"] = pct_rank(df["def_yapg"], asc=True)
    df["rank_def_3rd_pct"] = pct_rank(df["def_3rd_pct"], asc=True)

    # Build output records
    records = []
    for team, row in df.sort_index().iterrows():
        rec = {
            "team": team,
            "season": int(row["season"]),
            "games": int(row["games"]),
            "ppg": round(float(row["ppg"]), 2) if pd.notna(row["ppg"]) else None,
            "papg": round(float(row["papg"]), 2) if pd.notna(row["papg"]) else None,
            "off_ypg": round(float(row["off_ypg"]), 1) if pd.notna(row["off_ypg"]) else None,
            "def_yapg": round(float(row["def_yapg"]), 1) if pd.notna(row["def_yapg"]) else None,
            "off_3rd_pct": round(float(row["off_3rd_pct"]), 3) if pd.notna(row["off_3rd_pct"]) else None,
            "def_3rd_pct": round(float(row["def_3rd_pct"]), 3) if pd.notna(row["def_3rd_pct"]) else None,
            "ranks": {
                "off_ppg_pct": round(float(row["rank_off_ppg_pct"]), 3) if pd.notna(row["rank_off_ppg_pct"]) else None,
                "off_ypg_pct": round(float(row["rank_off_ypg_pct"]), 3) if pd.notna(row["rank_off_ypg_pct"]) else None,
                "off_3rd_pct": round(float(row["rank_off_3rd_pct"]), 3) if pd.notna(row["rank_off_3rd_pct"]) else None,
                "def_papg_pct": round(float(row["rank_def_papg_pct"]), 3) if pd.notna(row["rank_def_papg_pct"]) else None,
                "def_yapg_pct": round(float(row["rank_def_yapg_pct"]), 3) if pd.notna(row["rank_def_yapg_pct"]) else None,
                "def_3rd_pct": round(float(row["rank_def_3rd_pct"]), 3) if pd.notna(row["rank_def_3rd_pct"]) else None,
            }
        }
        records.append(rec)

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump({"season": latest, "teams": records}, f, indent=2)
    print(f"Wrote {OUT_JSON} with {len(records)} teams")

if __name__ == "__main__":
    main()
