#!/usr/bin/env python3
"""
Build docs/data/team_colors.json with real school colors when possible.

Priority per team:
1) data/raw/cfbd/cfbd_teams.csv  -> color / alt_color
2) Gist fallback logos.csv       -> color / alt_color
3) Deterministic, readable hash  -> stable fallback

Also:
- Includes EVERY team that appears in schedule (home/away) or predictions.json (home/away)
- Picks readable text color (#000 or #fff) by luminance
- Understands common alt names (alt_name1..alt_name3 in gist)
"""

import os, sys, json, hashlib
from typing import Dict, Optional, Iterable

import pandas as pd
import requests

TEAMS_CSV = "data/raw/cfbd/cfbd_teams.csv"
SCHEDULE_CSV = "data/raw/cfbd/cfb_schedule.csv"
PREDICTIONS_JSON = "docs/data/predictions.json"
OUT_JSON = "docs/data/team_colors.json"

# Gist fallback with columns: school, mascot, color, alt_color, alt_name1..3, conference, division
# If GitHub changes the raw URL format, we try two variants.
GIST_URLS = [
    "https://gist.githubusercontent.com/saiemgilani/c6596f0e1c8b148daabc2b7f1e6f6add/raw/logos.csv",
    "https://gist.githubusercontent.com/saiemgilani/c6596f0e1c8b148daabc2b7f1e6f6add/raw/0/logos",  # alt format
]

DEF_PRIMARY = "#2a3244"
DEF_TEXT = "#ffffff"

# ---------- small utils ----------
def hex_ok(s: Optional[str]) -> Optional[str]:
    if not isinstance(s, str): return None
    s = s.strip()
    if not s: return None
    if not s.startswith("#"): s = "#" + s
    if len(s) not in (4, 7): return None
    if len(s) == 4: s = "#" + "".join(ch*2 for ch in s[1:])
    try: int(s[1:], 16)
    except Exception: return None
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
    try: return "#000000" if rel_luminance(bg_hex) >= 0.6 else "#ffffff"
    except Exception: return DEF_TEXT

def hsl_to_hex(h: float, s: float, l: float) -> str:
    import math
    c = (1 - abs(2*l - 1)) * s
    x = c * (1 - abs((h/60.0) % 2 - 1))
    m = l - c/2
    if   0 <= h < 60:   rp, gp, bp = c, x, 0
    elif 60 <= h < 120: rp, gp, bp = x, c, 0
    elif 120 <= h < 180:rp, gp, bp = 0, c, x
    elif 180 <= h < 240:rp, gp, bp = 0, x, c
    elif 240 <= h < 300:rp, gp, bp = x, 0, c
    else:               rp, gp, bp = c, 0, x
    r = int(round((rp + m) * 255)); g = int(round((gp + m) * 255)); b = int(round((bp + m) * 255))
    return "#{:02x}{:02x}{:02x}".format(r, g, b)

def stable_color_from_name(name: str) -> str:
    if not name: return DEF_PRIMARY
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()
    hue = (int(digest[:8], 16) % 360)
    sat = 0.62
    lig = 0.38
    return hsl_to_hex(hue, sat, lig)

def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path): return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] Could not read {path}: {e}")
        return pd.DataFrame()

def fetch_csv(urls: Iterable[str]) -> pd.DataFrame:
    for u in urls:
        try:
            r = requests.get(u, timeout=45)
            r.raise_for_status()
            return pd.read_csv(pd.compat.StringIO(r.text))
        except Exception as e:
            print(f"[WARN] fetch failed {u}: {e}")
    return pd.DataFrame()

def load_predictions(path: str) -> dict:
    if not os.path.exists(path): return {}
    try:
        with open(path, "r") as f: return json.load(f)
    except Exception as e:
        print(f"[WARN] Could not read {path}: {e}")
        return {}

# ---------- building name sets ----------
def full_team_name_from_cfbd(row: pd.Series) -> str:
    school = str(row.get("school") or "").strip()
    mascot = str(row.get("mascot") or "").strip()
    if school and mascot: return f"{school} {mascot}"
    team = str(row.get("team") or "").strip()
    return team or school or mascot

def full_team_name_from_gist(row: pd.Series) -> str:
    school = str(row.get("school") or "").strip()
    mascot = str(row.get("mascot") or "").strip()
    if school and mascot: return f"{school} {mascot}"
    return school or mascot

def schedule_names(sched: pd.DataFrame) -> set:
    names = set()
    for col in ("home_team","away_team"):
        if col in sched.columns:
            names.update(sched[col].dropna().astype(str).str.strip().tolist())
    return names

# ---------- main ----------
def main():
    teams_cfbd = load_csv(TEAMS_CSV)
    schedule = load_csv(SCHEDULE_CSV)
    preds = load_predictions(PREDICTIONS_JSON)

    want_names = set()
    if not teams_cfbd.empty:
        teams_cfbd.columns = [c.strip() for c in teams_cfbd.columns]
        lcmap = {c.lower(): c for c in teams_cfbd.columns}
        for want in ["school","mascot","color","alt_color","classification"]:
            if want not in teams_cfbd.columns and want in lcmap:
                teams_cfbd.rename(columns={lcmap[want]: want}, inplace=True)
        # Limit to FBS if we have classification
        if "classification" in teams_cfbd.columns:
            mask = teams_cfbd["classification"].astype(str).str.lower().eq("fbs")
            subset = teams_cfbd[mask]
            if not subset.empty: teams_cfbd = subset
        for _, r in teams_cfbd.iterrows():
            nm = full_team_name_from_cfbd(r)
            if nm: want_names.add(nm)

    if not schedule.empty:
        want_names.update(schedule_names(schedule))

    for g in preds.get("games", []):
        for k in ("home","away"):
            nm = str(g.get(k) or "").strip()
            if nm: want_names.add(nm)

    if not want_names:
        # Nothing to do, write empty map
        os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
        with open(OUT_JSON, "w") as f: json.dump({}, f, indent=2)
        print(f"Wrote {OUT_JSON} (empty)")
        return

    # Build official color lookup from cfbd_teams.csv, if present
    official: Dict[str, Dict[str, str]] = {}
    if not teams_cfbd.empty and {"color","alt_color"}.issubset(teams_cfbd.columns):
        for _, r in teams_cfbd.iterrows():
            nm = full_team_name_from_cfbd(r)
            if not nm: continue
            c = hex_ok(r.get("color")); a = hex_ok(r.get("alt_color"))
            if c is None or c in ("#ffffff","#fff"): c = a or c
            if c:
                official[nm] = {"primary": c, "text": text_for_bg(c)}

    # If we still need help, pull gist and index by multiple keys (school+mascot and alt_names)
    gist_df = pd.DataFrame()
    if not official or len(official) < len(want_names):
        gist_df = fetch_csv(GIST_URLS)
        if not gist_df.empty:
            gist_df.columns = [c.strip() for c in gist_df.columns]
            lower = {c.lower(): c for c in gist_df.columns}
            for need in ["school","mascot","color","alt_color","division","alt_name1","alt_name2","alt_name3"]:
                if need not in gist_df.columns and need in lower:
                    gist_df.rename(columns={lower[need]: need}, inplace=True)
            # Prefer top division (“FBS”, “FBS Independent”, etc.) if present
            if "division" in gist_df.columns:
                mask = gist_df["division"].astype(str).str.contains("FBS", case=False, na=False)
                sub = gist_df[mask]
                if not sub.empty: gist_df = sub

    gist_index: Dict[str, Dict[str,str]] = {}
    if not gist_df.empty:
        for _, r in gist_df.iterrows():
            base = full_team_name_from_gist(r)
            c = hex_ok(r.get("color")); a = hex_ok(r.get("alt_color"))
            if c is None or c in ("#ffffff","#fff"): c = a or c
            if not c: continue
            rec = {"primary": c, "text": text_for_bg(c)}
            keys = set()
            if base: keys.add(base)
            # also store school-only and any alt names
            school = str(r.get("school") or "").strip()
            if school: keys.add(school)
            for k in ["alt_name1","alt_name2","alt_name3"]:
                alt = str(r.get(k) or "").strip()
                if alt: keys.add(alt)
            for k in keys:
                gist_index[k] = rec

    # Build final output covering all wanted names
    out: Dict[str, Dict[str, str]] = {}
    used_gist = []; used_fallback = []

    for name in sorted(want_names):
        # 1) exact cfbd match
        if name in official:
            out[name] = official[name]; continue
        # 2) gist exact
        if name in gist_index:
            out[name] = gist_index[name]; used_gist.append(name); continue
        # 3) try by school-only (drop last token = mascot-ish)
        school_guess = name.rsplit(" ", 1)[0] if " " in name else name
        if school_guess in official:
            out[name] = official[school_guess]; continue
        if school_guess in gist_index:
            out[name] = gist_index[school_guess]; used_gist.append(name); continue
        # 4) deterministic fallback
        c = stable_color_from_name(name)
        out[name] = {"primary": c, "text": text_for_bg(c)}
        used_fallback.append(name)

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote {OUT_JSON} with {len(out)} teams.")
    if used_gist:
        print(f"[INFO] Used gist colors for {len(used_gist)} team(s).")
    if used_fallback:
        print(f"[INFO] Used generated fallback colors for {len(used_fallback)} team(s).")

if __name__ == "__main__":
    main()
