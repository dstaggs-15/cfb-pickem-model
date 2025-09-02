# scripts/lib/rolling.py
import pandas as pd, numpy as np
from .parsing import parse_ratio_val

STAT_FEATURES = [
    "totalYards","netPassingYards","rushingYards","firstDowns",
    "turnovers","sacks","tacklesForLoss","thirdDownEff","fourthDownEff","kickingPoints",
]

def numericize_stat(cat: str, val):
    if cat in ("thirdDownEff","fourthDownEff"):
        return parse_ratio_val(val)
    return pd.to_numeric(val, errors="coerce")

def long_stats_to_wide(df_stats: pd.DataFrame) -> pd.DataFrame:
    keep = df_stats[df_stats["category"].isin(STAT_FEATURES)].copy()
    keep["stat_value_num"] = [numericize_stat(c, v) for c, v in zip(keep["category"], keep["stat_value"])]
    wide = keep.pivot_table(index=["game_id","team","homeAway"],
                            columns="category", values="stat_value_num", aggfunc="mean").reset_index()
    for c in STAT_FEATURES:
        if c not in wide.columns:
            wide[c] = np.nan
    return wide

def build_sidewise_rollups(schedule: pd.DataFrame, wide: pd.DataFrame, n: int):
    sw = schedule[["game_id","date"]].copy()
    w = wide.merge(sw, on="game_id", how="left").sort_values(["team","date","game_id"]).reset_index(drop=True)
    outs = []
    for side in ["home","away"]:
        side_df = w[w["homeAway"]==side].copy()
        side_df = side_df.sort_values(["team","date","game_id"])
        grp = side_df.groupby("team", group_keys=False)
        side_df[f"{side}_games_so_far"] = grp.cumcount()
        for c in STAT_FEATURES:
            side_df[f"{side}_R{n}_{c}"] = grp[c].apply(lambda s: s.rolling(window=n, min_periods=1).mean()).shift(1)
        side_df[f"{side}_R{n}_count"] = grp.apply(lambda g: g[f"{side}_games_so_far"].shift(1).clip(lower=0)).values
        side_df[f"{side}_R{n}_count"] = side_df[f"{side}_R{n}_count"].fillna(0).clip(upper=n)
        cols = ["game_id","team", f"{side}_R{n}_count"] + [f"{side}_R{n}_{c}" for c in STAT_FEATURES]
        outs.append(side_df[cols])
    return outs[0], outs[1]

def latest_per_team(roll_df: pd.DataFrame, side: str, n: int):
    cols = [c for c in roll_df.columns if c.startswith(f"{side}_R{n}_")]
    need = ["team"] + cols
    v = roll_df[need].dropna(how="all", subset=[c for c in cols if c.endswith(tuple(['totalYards','firstDowns','thirdDownEff']))]) \
                     .sort_values(["team"]).groupby("team").tail(1).set_index("team")
    return v
