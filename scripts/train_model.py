#!/usr/bin/env python3
# --- bootstrap so 'scripts.lib' imports work when run as a file ---
import os, sys
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
# -----------------------------------------------------------------

import json, datetime as dt
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
from joblib import dump

TRAIN_PARQUET = "data/derived/training.parquet"
META_JSON = "docs/data/train_meta.json"
MODEL_PATH = "data/derived/model.joblib"
METRICS_JSON = "docs/data/train_metrics.json"

def season_ahead(df: pd.DataFrame, feats):
    seasons = sorted(pd.to_numeric(df["season"], errors="coerce").dropna().unique().tolist())
    res=[]
    for i in range(2,len(seasons)):
        test=seasons[i]; calib=seasons[i-1]; train=seasons[:i-1]
        tr=df[df["season"].isin(train)]; ca=df[df["season"]==calib]; te=df[df["season"]==test]
        if len(tr)<200 or len(ca)<80 or len(te)<80: continue
        base=HistGradientBoostingClassifier(max_depth=None,learning_rate=0.08,max_iter=400,min_samples_leaf=20)
        base.fit(tr[feats], tr["home_win"])
        method="isotonic" if len(ca)>=400 else "sigmoid"
        cal=CalibratedClassifierCV(estimator=base, method=method, cv="prefit")
        cal.fit(ca[feats], ca["home_win"])
        p=cal.predict_proba(te[feats])[:,1]
        res.append({
            "season": int(test),
            "acc": accuracy_score(te["home_win"], (p>=0.5).astype(int)),
            "auc": roc_auc_score(te["home_win"], p),
            "brier": brier_score_loss(te["home_win"], p),
        })
    if not res: return {"acc": None,"auc": None,"brier": None}
    import numpy as np, pandas as pd
    d=pd.DataFrame(res)
    return {"acc": float(d["acc"].mean()), "auc": float(d["auc"].mean()), "brier": float(d["brier"].mean())}

def main():
    meta=json.load(open(META_JSON,"r"))
    feats=meta["features"]
    df=pd.read_parquet(TRAIN_PARQUET).copy()

    seasons=sorted(pd.to_numeric(df["season"], errors="coerce").dropna().unique().tolist())
    if len(seasons)>=2:
        calib_season=seasons[-1]
        tr=df[df["season"]<calib_season]; ca=df[df["season"]==calib_season]
        method="isotonic" if len(ca)>=400 else "sigmoid"
    else:
        df=df.sort_values("date"); split=int(len(df)*0.9)
        tr,ca=df.iloc[:split], df.iloc[split:]
        method="sigmoid"

    base=HistGradientBoostingClassifier(max_depth=None,learning_rate=0.08,max_iter=400,min_samples_leaf=20)
    base.fit(tr[feats], tr["home_win"])
    cal=CalibratedClassifierCV(estimator=base, method=method, cv="prefit")
    cal.fit(ca[feats], ca["home_win"])

    m=season_ahead(df,feats)
    metrics={
        "generated": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "season_ahead_acc": None if m["acc"] is None else round(m["acc"],4),
        "season_ahead_auc": None if m["auc"] is None else round(m["auc"],4),
        "season_ahead_brier": None if m["brier"] is None else round(m["brier"],4),
        "cal_method": method,
    }
    json.dump(metrics, open(METRICS_JSON,"w"), indent=2)

    dump(cal, MODEL_PATH)
    print(f"Wrote {MODEL_PATH} and {METRICS_JSON}")

if __name__ == "__main__":
    main()
