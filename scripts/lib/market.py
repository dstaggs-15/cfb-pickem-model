# scripts/lib/market.py
import numpy as np, pandas as pd, math
from sklearn.linear_model import LogisticRegression

def median_lines(lines: pd.DataFrame) -> pd.DataFrame:
    if lines is None or lines.empty:
        return pd.DataFrame(columns=["game_id","spread_home","over_under"])
    df = lines.copy()
    for old, new in [("spread","spread"),("overUnder","over_under"),("overunder","over_under")]:
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
    df["spread"] = pd.to_numeric(df.get("spread"), errors="coerce")
    df["over_under"] = pd.to_numeric(df.get("over_under"), errors="coerce")
    if "game_id" not in df.columns:
        return pd.DataFrame(columns=["game_id","spread_home","over_under"])
    med = df.groupby("game_id")[["spread","over_under"]].median().reset_index()
    med = med.rename(columns={"spread":"spread_home"})
    return med

def fit_market_mapping(spread: np.ndarray, y: np.ndarray):
    ok = ~np.isnan(spread) & ~np.isnan(y)
    spread = spread[ok]; y = y[ok]
    if len(spread) < 200:
        return {"a": 0.0, "b": 0.17}
    X = (-spread).reshape(-1,1)
    lr = LogisticRegression(max_iter=200)
    lr.fit(X, y)
    return {"a": float(lr.intercept_[0]), "b": float(lr.coef_[0][0])}

def market_prob(spread_home: float, a: float, b: float):
    if spread_home is None or pd.isna(spread_home): return None
    z = a + b * (-float(spread_home))
    return 1.0 / (1.0 + math.exp(-z))
