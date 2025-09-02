# scripts/lib/context.py
import math, pandas as pd, numpy as np

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
    return m[["game_id","rest_diff","shortweek_diff","bye_diff","travel_diff_km","neutral_site","is_postseason"]]
