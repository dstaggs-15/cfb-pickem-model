# scripts/lib/context.py

from __future__ import annotations
import pandas as pd
import numpy as np
from haversine import haversine

def _std_venue_table(venues_df: pd.DataFrame) -> pd.DataFrame:
    """Return a venues table with columns: venue_id(str), venue_name, venue_lat, venue_lon"""
    if venues_df is None or venues_df.empty:
        return pd.DataFrame(columns=["venue_id", "venue_name", "venue_lat", "venue_lon"])

    v = venues_df.copy()

    # Common CFBD variants
    id_col = None
    for cand in ["venue_id", "id", "venueId"]:
        if cand in v.columns:
            id_col = cand
            break
    name_col = None
    for cand in ["name", "venue", "venue_name"]:
        if cand in v.columns:
            name_col = cand
            break
    lat_col = None
    for cand in ["latitude", "lat", "location.latitude"]:
        if cand in v.columns:
            lat_col = cand
            break
    lon_col = None
    for cand in ["longitude", "lon", "lng", "location.longitude"]:
        if cand in v.columns:
            lon_col = cand
            break

    out = pd.DataFrame()
    if id_col is not None:
        out["venue_id"] = v[id_col].astype(str)
    else:
        out["venue_id"] = pd.Series(dtype="object")
    out["venue_name"] = v[name_col].astype(str) if name_col else pd.Series(dtype="object")
    out["venue_lat"] = pd.to_numeric(v[lat_col], errors="coerce") if lat_col else np.nan
    out["venue_lon"] = pd.to_numeric(v[lon_col], errors="coerce") if lon_col else np.nan
    return out.drop_duplicates()


def _std_team_table(teams_df: pd.DataFrame) -> pd.DataFrame:
    """Return a teams table with columns: team(str), team_lat, team_lon"""
    if teams_df is None or teams_df.empty:
        return pd.DataFrame(columns=["team", "team_lat", "team_lon"])

    t = teams_df.copy()

    # Team name
    team_col = None
    for cand in ["school", "team", "name"]:
        if cand in t.columns:
            team_col = cand
            break

    # Latitude/Longitude (various shapes)
    lat_col = None
    for cand in ["latitude", "lat", "location.latitude", "venue.latitude", "home_latitude"]:
        if cand in t.columns:
            lat_col = cand
            break
    lon_col = None
    for cand in ["longitude", "lon", "location.longitude", "venue.longitude", "home_longitude"]:
        if cand in t.columns:
            lon_col = cand
            break

    out = pd.DataFrame()
    out["team"] = t[team_col].astype(str) if team_col else pd.Series(dtype="object")
    out["team_lat"] = pd.to_numeric(t[lat_col], errors="coerce") if lat_col else np.nan
    out["team_lon"] = pd.to_numeric(t[lon_col], errors="coerce") if lon_col else np.nan
    return out.drop_duplicates()


def _attach_venue_coords(schedule: pd.DataFrame, venues_df: pd.DataFrame) -> pd.DataFrame:
    """Attach venue coordinates by venue_id if present, else try venue name match."""
    df = schedule.copy()

    # Ensure we have venue_id and venue (name) columns even if empty
    if "venue_id" not in df.columns:
        df["venue_id"] = pd.NA
    if "venue" not in df.columns:
        df["venue"] = pd.NA

    venues = _std_venue_table(venues_df)

    # Try join on venue_id first (string on both sides)
    left = df.copy()
    left["venue_id"] = left["venue_id"].astype(str)
    venues_id = venues[["venue_id", "venue_lat", "venue_lon"]].dropna(subset=["venue_id"])
    venues_id["venue_id"] = venues_id["venue_id"].astype(str)

    df1 = left.merge(venues_id, on="venue_id", how="left", suffixes=("", "_idmatch"))

    # For rows with no lat/lon from id, try name join
    missing = df1["venue_lat"].isna() | df1["venue_lon"].isna()
    if missing.any():
        venues_name = venues[["venue_name", "venue_lat", "venue_lon"]].dropna(subset=["venue_name"])
        # Normalize for fuzzy-ish exact match
        name_left = df1.loc[missing, ["game_id", "venue"]].copy()
        name_left["venue_name"] = name_left["venue"].astype(str).str.strip().str.lower()
        venues_name["venue_name"] = venues_name["venue_name"].astype(str).str.strip().str.lower()

        name_join = name_left.merge(venues_name, on="venue_name", how="left")
        name_join = name_join[["game_id", "venue_lat", "venue_lon"]]

        df1 = df1.merge(name_join, on="game_id", how="left", suffixes=("", "_name"))
        # Prefer id-match coords; fill with name-match coords
        for coord in ["venue_lat", "venue_lon"]:
            df1[coord] = df1[coord].where(~df1[coord].isna(), df1[f"{coord}_name"])
        df1 = df1.drop(columns=[c for c in df1.columns if c.endswith("_name")])

    return df1


def _compute_rest_days(schedule: pd.DataFrame) -> pd.DataFrame:
    """Compute rest days per team per game."""
    df = schedule.copy()
    # Days since previous game for each team, then merge home/away
    def side_rest(side_team_col: str, prefix: str) -> pd.DataFrame:
        side = df[["game_id", "date", "season", side_team_col]].rename(columns={side_team_col: "team"}).copy()
        side = side.dropna(subset=["team"])
        side["date"] = pd.to_datetime(side["date"], errors="coerce", utc=True)
        side = side.sort_values(["team", "date"])
        side["prev_date"] = side.groupby("team")["date"].shift(1)
        side[f"rest_{prefix}_days"] = (side["date"] - side["prev_date"]).dt.total_seconds() / (60 * 60 * 24)
        return side[["game_id", f"rest_{prefix}_days"]]

    home_rest = side_rest("home_team", "home")
    away_rest = side_rest("away_team", "away")

    out = df[["game_id"]].drop_duplicates()
    out = out.merge(home_rest, on="game_id", how="left")
    out = out.merge(away_rest, on="game_id", how="left")
    return out


def _compute_travel_km(schedule_with_coords: pd.DataFrame, teams_df: pd.DataFrame) -> pd.DataFrame:
    """Compute great-circle distance from team home coords to venue coords (km)."""
    df = schedule_with_coords.copy()
    teams = _std_team_table(teams_df)

    # Home team coords
    h = df[["game_id", "home_team", "venue_lat", "venue_lon"]].copy()
    h = h.merge(teams.rename(columns={"team": "home_team", "team_lat": "home_lat", "team_lon": "home_lon"}),
                on="home_team", how="left")
    # Away team coords
    a = df[["game_id", "away_team", "venue_lat", "venue_lon"]].copy()
    a = a.merge(teams.rename(columns={"team": "away_team", "team_lat": "away_lat", "team_lon": "away_lon"}),
                on="away_team", how="left")

    def _dist(row_lat1, row_lon1, row_lat2, row_lon2):
        if pd.isna(row_lat1) or pd.isna(row_lon1) or pd.isna(row_lat2) or pd.isna(row_lon2):
            return np.nan
        try:
            return haversine((row_lat1, row_lon1), (row_lat2, row_lon2))
        except Exception:
            return np.nan

    h["travel_home_km"] = [
        _dist(h.loc[i, "home_lat"], h.loc[i, "home_lon"], h.loc[i, "venue_lat"], h.loc[i, "venue_lon"])
        for i in h.index
    ]
    a["travel_away_km"] = [
        _dist(a.loc[i, "away_lat"], a.loc[i, "away_lon"], a.loc[i, "venue_lat"], a.loc[i, "venue_lon"])
        for i in a.index
    ]

    out = df[["game_id"]].drop_duplicates()
    out = out.merge(h[["game_id", "travel_home_km"]], on="game_id", how="left")
    out = out.merge(a[["game_id", "travel_away_km"]], on="game_id", how="left")
    return out


def rest_and_travel(
    schedule: pd.DataFrame,
    teams_df: pd.DataFrame | None = None,
    venues_df: pd.DataFrame | None = None,
    predict_df: pd.DataFrame | None = None
) -> pd.DataFrame:
    """
    Returns a DataFrame keyed by game_id with:
      - rest_home_days, rest_away_days
      - travel_home_km, travel_away_km  (NaN if coords missing)
      - neutral_site, postseason (if present on schedule; otherwise left alone)
    """
    if schedule is None or schedule.empty:
        return pd.DataFrame(columns=[
            "game_id", "rest_home_days", "rest_away_days", "travel_home_km", "travel_away_km"
        ])

    df = schedule.copy()
    # Dates normalized in features._prep_schedule, but make sure:
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)

    # Attach venue coordinates via id or name (robust)
    df = _attach_venue_coords(df, venues_df if venues_df is not None else pd.DataFrame())

    # Rest days
    rest = _compute_rest_days(df)

    # Travel distances (may be NaN when coords absent)
    travel = _compute_travel_km(df, teams_df if teams_df is not None else pd.DataFrame())

    out = df[["game_id", "neutral_site"]].drop_duplicates()
    if "postseason" in df.columns:
        out = out.merge(df[["game_id", "postseason"]].drop_duplicates(), on="game_id", how="left")
    out = out.merge(rest, on="game_id", how="left")
    out = out.merge(travel, on="game_id", how="left")

    return out
