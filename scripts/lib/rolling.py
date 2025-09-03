import pandas as pd
import numpy as np
import os

# \--- NEW: Define path for the season averages file ---

DERIVED = "data/derived"
SEASON\_AVG\_PARQUET = f"{DERIVED}/season\_averages.parquet"

# Define the list of statistical features to be used for rolling averages

STAT\_FEATURES = [
"ppa", "success\_rate", "explosiveness", "power\_success", "stuff\_rate",
"line\_yards", "second\_level\_yards", "open\_field\_yards", "points\_per\_opportunity",
"havoc", "turnovers", "field\_pos\_avg\_start"
]

def long\_stats\_to\_wide(team\_stats):
"""Pivots the long-format team stats to a wide format."""
pivoted = team\_stats.pivot(
index="game\_id",
columns="home\_away",
values=[c for c in team\_stats.columns if c not in ["game\_id", "home\_away", "team"]]
)
pivoted.columns = [f'{col[0]}\_{col[1]}' for col in pivoted.columns]
return pivoted

def \_get\_rollups(df, last\_n, season\_averages\_df):
"""
\--- NEW AND IMPROVED ---
Helper to compute rolling stats, now with season-average carry-forward logic.
"""
df = df.sort\_values(by=["team", "date"]).reset\_index(drop=True)
