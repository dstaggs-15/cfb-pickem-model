#!/usr/bin/env python3
"""
Build docs/data/team_colors.json

Priority for a team's primary color:
1) data/raw/cfbd/cfbd_teams.csv -> 'color' (hex)
2) data/raw/cfbd/cfbd_teams.csv -> 'alt_color' (hex)
3) Deterministic fallback color derived from team name (hash → HSL)

Text color (#000 or #fff) is chosen by luminance for contrast.

Teams included:
- Any team present in cfbd_teams.csv (if available)
- Any team appearing in the latest schedule CSV (home_team/away_team)
- Any team appearing in docs/data/predictions.json (home/away)

Result:
{
  "Alabama Crimson Tide": { "primary": "#9e1b32", "text": "#ffffff" },
  ...
}
"""

import os
import sys
import json
import math
import hashlib
from typing import Dict, Optional

import pandas as pd

# Inputs
TEAMS_CSV = "data/raw/cfbd/cfbd_teams.csv"
SCHEDULE_CSV = "data/raw/cfbd/cfb_schedule.csv"
PREDICTIONS_JSON = "docs/data/predictions.json"

# Output
OUT_JSON = "docs/data/team_colors.json"

# Fallbacks
DEF_PRIMARY = "#2a3244"  # used only if hashing somehow fails (shouldn't)
DEF_TEXT = "#ffffff"

def hex_ok(s: Optional[str]) -> Optional[str]:
    if not isinstance(s, str):
        return None
    s = s.strip()
    if not s:
        return None
    if not s.startswith("#"):
        s = "#" + s
    if len(s) not in (4, 7):
        return None
    if len(s) == 4:  # #abc -> #aabbcc
        s = "#" + "".join(ch*2 for ch in s[1:])
    try:
        int(s[1:], 16)
    except Exception:
        return None
    return s.lower()

def rel_luminance(hex_color: str) -> float:
    h = hex_color.lstrip("#")
    r = int(h[0:2], 16) / 255.0
    g = int(h[2:4], 16) / 255.0
    b = int(h[4:6], 16) / 255.0
    def to_lin(v): return v/12.92 if v <= 0.04045 else ((v+0.055)/1.055) ** 2.4
    rL, gL, bL = to_lin(r), to_lin(g), to_lin(b)
    return 0.2126*rL + 0.7152*gL + 0.0722*bL

def text_for_bg(bg_hex: str) -> str:
    try:
        return "#000000" if rel_luminance(bg_hex) >= 0.6 else "#ffffff"
    except Exception:
        return DEF_TEXT

def hsl_to_hex(h: float, s: float, l: float) -> str:
    # h in [0,360), s,l in [0,1]
    c = (1 - abs(2*l - 1)) * s
    x = c * (1 - abs((h/60.0) % 2 - 1))
    m = l - c/2
    if   0 <= h < 60:   rp, gp, bp = c, x, 0
    elif 60 <= h < 120: rp, gp, bp = x, c, 0
    elif 120 <= h < 180:rp, gp, bp = 0, c, x
    elif 180 <= h < 240:rp, gp, bp = 0, x, c
    elif 240 <= h < 300:rp, gp, bp = x, 0, c
    else:               rp, gp, bp = c, 0, x
    r = int(round((rp + m) * 255))
    g = int(round((gp + m) * 255))
    b = int(round((bp + m) * 255))
    return "#{:02x}{:02x}{:02x}".format(r, g, b)

def stable_color_from_name(name: str) -> str:
    """
    Deterministic fallback:
    - Hash team name → hue
    - Moderate saturation & lightness to avoid unreadable colors
    """
    if not name:
        return DEF_PRIMARY
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()
    # Use first 8 hex digits -> int -> hue [0,360)
    hue = (int(digest[:8], 16) % 360)
    sat = 0.62
    lig = 0.38
    return hsl_to_hex(hue, sat, lig)

def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] Could not read {path}: {e}")
        return pd.DataFrame()

def load_predictions(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Could not read {path}: {e}")
        return {}

def full_team_name(row: pd.Series) -> str:
    # Prefer "School Mascot" if available; otherwise, best effort
    school = str(row.get("school") or "").strip()
    mascot = str(row.get("mascot") or "").strip()
    if school and mascot:
        return f"{school} {mascot}"
    team = str(row.get("team") or "").strip()
    return team or school or mascot

def main():
    teams_df = load_csv(TEAMS_CSV)
    schedule_df = load_csv(SCHEDULE_CSV)
    preds = load_predictions(PREDICTIONS_JSON)

    want_names = set()

    # 1) From cfbd teams (if present)
    if not teams_df.empty:
        teams_df.columns = [c.strip() for c in teams_df.columns]
        # normalize common column names if they came in as different cases
        lcmap = {c.lower(): c for c in teams_df.columns}
        for want in ["school", "mascot", "color", "alt_color", "classification", "division"]:
            if want not in teams_df.columns and want in lcmap:
                teams_df.rename(columns={lcmap[want]: want}, inplace=True)

        # Limit to FBS if classification available
        if "classification" in teams_df.columns:
            mask = teams_df["classification"].astype(str).str.lower().eq("fbs")
            subset = teams_df[mask]
            if not subset.empty:
                teams_df = subset

        for _, r in teams_df.iterrows():
            name = full_team_name(r)
            if name:
                want_names.add(name)

    # 2) From schedule (home_team/away_team)
    if not schedule_df.empty:
        for col in ("home_team", "away_team"):
            if col in schedule_df.columns:
                vals = schedule_df[col].dropna().astype(str).str.strip().tolist()
                want_names.update(vals)

    # 3) From predictions.json
    for g in preds.get("games", []):
        for k in ("home", "away"):
            nm = str(g.get(k) or "").strip()
            if nm:
                want_names.add(nm)

    # If we still have zero names, bail gracefully (create empty mapping)
    if not want_names:
        print("[WARN] No team names discovered; writing empty team_colors.json")
        os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
        with open(OUT_JSON, "w") as f:
            json.dump({}, f, indent=2)
        print(f"Wrote {OUT_JSON} (empty)")
        return

    # Build color map
    out: Dict[str, Dict[str, str]] = {}
    used_fallback = []

    # If we have official colors, index them now
    official: Dict[str, Dict[str, str]] = {}
    if not teams_df.empty and {"color", "alt_color"}.issubset(teams_df.columns):
        for _, r in teams_df.iterrows():
            name = full_team_name(r)
            if not name:
                continue
            primary = hex_ok(r.get("color"))
            alt = hex_ok(r.get("alt_color"))
            if primary is None or primary in ("#ffffff", "#fff"):
                primary = alt or primary
            if primary:
                official[name] = {"primary": primary, "text": text_for_bg(primary)}

    for name in sorted(want_names):
        if name in official:
            out[name] = official[name]
            continue
        # Try to find by school-only if we have teams_df but a name mismatch
        if not teams_df.empty and "school" in teams_df.columns and "mascot" in teams_df.columns:
            school_only = name.rsplit(" ", 1)[0] if " " in name else name
            cand = teams_df[teams_df["school"].astype(str).str.strip() == school_only]
            if not cand.empty:
                primary = hex_ok(cand.iloc[0].get("color")) or hex_ok(cand.iloc[0].get("alt_color"))
                if primary:
                    out[name] = {"primary": primary, "text": text_for_bg(primary)}
                    continue
        # Fallback: deterministic color from name
        primary = stable_color_from_name(name)
        text = text_for_bg(primary)
        out[name] = {"primary": primary, "text": text}
        used_fallback.append(name)

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {OUT_JSON} with {len(out)} teams.")
    if used_fallback:
        print(f"[INFO] Used generated fallback colors for {len(used_fallback)} team(s).")

if __name__ == "__main__":
    main()
