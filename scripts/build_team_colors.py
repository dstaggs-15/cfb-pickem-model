#!/usr/bin/env python3
"""
Build docs/data/team_colors.json from CFBD teams data.

- Uses data/raw/cfbd/cfbd_teams.csv (created by your fetch step).
- Falls back to schedule to decide which team-name format to emit (School + Mascot).
- Chooses a readable text color (#000 or #fff) based on luminance.
- If a team is missing colors, picks a safe dark default and logs it.

Output schema:
{
  "Alabama Crimson Tide": { "primary": "#9e1b32", "text": "#ffffff" },
  ...
}
"""
import os, json, sys
from io import StringIO
from typing import Dict, Tuple, Optional

import pandas as pd

TEAMS_CSV = "data/raw/cfbd/cfbd_teams.csv"
SCHEDULE_CSV = "data/raw/cfbd/cfb_schedule.csv"  # used only to infer naming
OUT_JSON = "docs/data/team_colors.json"

DEF_PRIMARY = "#2a3244"  # fallback bubble background
DEF_TEXT = "#ffffff"

def _hex_ok(s: Optional[str]) -> Optional[str]:
    if not isinstance(s, str): return None
    s = s.strip()
    if not s: return None
    if s[0] != "#": s = "#" + s
    if len(s) not in (4, 7): return None
    # Expand #abc -> #aabbcc
    if len(s) == 4:
        s = "#" + "".join([ch*2 for ch in s[1:]])
    # Quick sanity: all hex
    try:
        int(s[1:], 16)
    except Exception:
        return None
    return s.lower()

def _rel_luminance(hex_color: str) -> float:
    # WCAG-ish luminance (sRGB to linear)
    h = hex_color.lstrip("#")
    r = int(h[0:2], 16) / 255.0
    g = int(h[2:4], 16) / 255.0
    b = int(h[4:6], 16) / 255.0
    def to_lin(v): return v/12.92 if v <= 0.04045 else ((v+0.055)/1.055) ** 2.4
    rL, gL, bL = to_lin(r), to_lin(g), to_lin(b)
    return 0.2126*rL + 0.7152*gL + 0.0722*bL

def _text_for_bg(bg_hex: str) -> str:
    # Simple cutoff; higher luminance -> use dark text
    try:
        return "#000000" if _rel_luminance(bg_hex) >= 0.6 else "#ffffff"
    except Exception:
        return DEF_TEXT

def _load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)

def _full_team_name(row: pd.Series) -> str:
    # CFBD teams file typically has 'school' and 'mascot'
    school = str(row.get("school") or "").strip()
    mascot = str(row.get("mascot") or "").strip()
    if school and mascot:
        return f"{school} {mascot}"
    # Fallback to 'team' if present
    t = str(row.get("team") or "").strip()
    return t or school or mascot

def _derive_set_of_current_names(schedule: pd.DataFrame) -> set:
    # Helps us confirm the "School Mascot" format we should emit
    names = set()
    for col in ("home_team", "away_team"):
        if col in schedule.columns:
            names.update(schedule[col].dropna().astype(str).str.strip().tolist())
    return names

def main():
    try:
        teams = _load_csv(TEAMS_CSV)
    except FileNotFoundError:
        print(f"[ERROR] {TEAMS_CSV} not found. Run your CFBD fetch step first.", file=sys.stderr)
        sys.exit(1)

    # Normalize columns
    teams.columns = [c.strip() for c in teams.columns]
    cols_lower = {c.lower(): c for c in teams.columns}
    # Try to standardize common column names
    for want in ["school", "mascot", "classification", "division", "color", "alt_color"]:
        if want not in teams.columns and want in cols_lower:
            teams.rename(columns={cols_lower[want]: want}, inplace=True)

    # Optional: limit to FBS if classification exists
    if "classification" in teams.columns:
        mask = teams["classification"].astype(str).str.lower().eq("fbs")
        teams_fbs = teams[mask].copy()
        if not teams_fbs.empty:
            teams = teams_fbs

    # Try to align with the naming in schedule (e.g., "Miami (FL) Hurricanes")
    sched_names = set()
    if os.path.exists(SCHEDULE_CSV):
        try:
            schedule = _load_csv(SCHEDULE_CSV)
            sched_names = _derive_set_of_current_names(schedule)
        except Exception as e:
            print(f"[WARN] Could not load schedule for name alignment: {e}")

    out: Dict[str, Dict[str, str]] = {}
    missing_color = []
    produced = 0

    for _, row in teams.iterrows():
        name = _full_team_name(row)
        if not name:
            continue

        primary = _hex_ok(row.get("color"))
        alt = _hex_ok(row.get("alt_color"))

        # If primary is invalid or too light, switch to alt if valid
        if (primary is None) or primary in ("#ffffff", "#fff"):
            primary = alt or primary
        if primary is None:
            missing_color.append(name)
            primary = DEF_PRIMARY

        text = _text_for_bg(primary)
        out[name] = {"primary": primary, "text": text}
        produced += 1

    # If the schedule suggests names that aren't in teams.csv format, try to map
    # Example: teams.csv might have "UTSA Roadrunners" but schedule could use the same.
    # We'll check for schedule names that already exist; if not, try a loose match on 'school'.
    if sched_names:
        # Build quick index by school only, in case some sched uses school-only names elsewhere
        by_school = {}
        if "school" in teams.columns:
            for _, r in teams.iterrows():
                school = str(r.get("school") or "").strip()
                if school:
                    key = f"{school} {str(r.get('mascot') or '').strip()}".strip()
                    by_school[school] = out.get(key) or out.get(school)

        for sname in sorted(sched_names):
            if sname in out:
                continue
            # If schedule name is "School Mascot", we already tried. Try by school-only
            school_part = sname.rsplit(" ", 1)[0] if " " in sname else sname
            rec = by_school.get(school_part)
            if rec:
                out[sname] = rec

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    # Sort keys for stable diffs
    with open(OUT_JSON, "w") as f:
        json.dump(dict(sorted(out.items(), key=lambda kv: kv[0])), f, indent=2)

    print(f"Wrote {OUT_JSON} with {len(out)} teams (from {produced} rows).")
    if missing_color:
        print(f"[WARN] {len(missing_color)} teams missing color in source; used fallback:")
        for nm in sorted(missing_color):
            print(" -", nm)

if __name__ == "__main__":
    main()
