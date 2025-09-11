#!/usr/bin/env python3
# scripts/evaluate_and_calibrate.py
#
# Evaluate predictions and fit a probability calibrator (Platt scaling).
# Now supports a rolling window: --window 4 (fit on last 4 completed weeks).
#
# Usage examples:
#   python -m scripts.evaluate_and_calibrate --exclude "SMU"
#   python -m scripts.evaluate_and_calibrate --window 4
#   python -m scripts.evaluate_and_calibrate --season 2025 --week 2 --exclude "SMU" --no-write-weekly-log

import os, json, argparse
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

PRED_PATH   = "docs/data/predictions.json"
SCHED_PATH  = "data/raw/cfbd/cfb_schedule.csv"
CALIB_PATH  = "data/derived/calibrator.json"
WEEKLY_LOG  = "docs/data/weekly_eval.json"

EPS = 1e-6

ALIASES = {
    "app state": "appalachian state",
    "southern miss": "southern mississippi",
    "cal": "california",
    "texas a and m": "texas a&m",
    "washington st": "washington state",
}

def canon(s: str) -> str:
    s = (s or "").lower().replace("&","and")
    s = "".join(ch if ch.isalnum() or ch==" " else " " for ch in s)
    s = " ".join(s.split())
    return ALIASES.get(s, s)

def _load_preds() -> pd.DataFrame:
    if not os.path.exists(PRED_PATH):
        raise FileNotFoundError(PRED_PATH)
    with open(PRED_PATH, "r") as f:
        data = json.load(f)
    arr = data["games"] if isinstance(data, dict) and "games" in data else data
    df = pd.DataFrame(arr)
    if "model_prob_home" not in df.columns:
        raise RuntimeError("predictions.json missing model_prob_home")
    df["p_home_raw"] = pd.to_numeric(df["model_prob_home"], errors="coerce").clip(EPS, 1-EPS)
    df["home_c"] = df["home_team"].map(canon)
    df["away_c"] = df["away_team"].map(canon)
    return df

def _load_sched() -> pd.DataFrame:
    df = pd.read_csv(SCHED_PATH)
    for c in ["season","week","home_points","away_points"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["home_team","away_team"])
    df["home_c"] = df["home_team"].map(canon)
    df["away_c"] = df["away_team"].map(canon)
    return df

def _completed_weeks(df: pd.DataFrame):
    played = df.dropna(subset=["home_points","away_points"])
    if played.empty:
        return []
    wk = played[["season","week"]].drop_duplicates().sort_values(["season","week"])
    return list(map(tuple, wk.itertuples(index=False, name=None)))

def _attach_week(preds: pd.DataFrame, sched: pd.DataFrame, season: int, week: int, excludes: list[str]) -> pd.DataFrame:
    s = sched[(sched["season"]==season) & (sched["week"]==week)].copy()
    # A: as listed
    a = s[["home_c","away_c","home_points","away_points"]].copy()
    # B: swapped (so predictions match either orientation)
    b = s.rename(columns={"home_c":"away_c","away_c":"home_c",
                          "home_points":"away_points","away_points":"home_points"})[["home_c","away_c","home_points","away_points"]]
    pool = pd.concat([a.assign(_flip=0), b.assign(_flip=1)], ignore_index=True)

    df = preds.merge(pool, on=["home_c","away_c"], how="inner")
    if excludes:
        ex = [x.lower() for x in excludes]
        df = df[~(df["home_team"].str.lower().str.contains("|".join(ex)) |
                  df["away_team"].str.lower().str.contains("|".join(ex)))]
    df = df.dropna(subset=["home_points","away_points"])
    df["y_true"] = (df["home_points"] > df["away_points"]).astype(int)
    df["season"] = season; df["week"] = week
    return df

def _metrics(y, p):
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1-EPS)
    return {
        "accuracy": float(np.mean((p >= 0.5) == (y == 1))),
        "brier": float(brier_score_loss(y, p)),
        "logloss": float(log_loss(y, p)),
        "n": int(len(y)),
    }

def _fit_platt(y, p):
    # p_cal = sigmoid(w*logit(p) + b)
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1-EPS)
    z = np.log(p/(1-p))
    w, b = 1.0, 0.0
    lr = 0.01
    for _ in range(2000):
        q = 1/(1+np.exp(-(w*z + b)))
        grad_w = np.sum((q - y) * z) / len(y)
        grad_b = np.sum(q - y) / len(y)
        w -= lr * grad_w
        b -= lr * grad_b
    return float(w), float(b)

def _apply_platt(p, w, b):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1-EPS)
    z = np.log(p/(1-p))
    return 1/(1+np.exp(-(w*z + b)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--week", type=int, default=None)
    ap.add_argument("--window", type=int, default=1, help="How many last completed weeks to use for fitting (default 1).")
    ap.add_argument("--exclude", type=str, default="", help="Comma-separated substrings to exclude (e.g. 'SMU')")
    ap.add_argument("--min-samples", type=int, default=8)
    ap.add_argument("--no-write-weekly-log", action="store_true")
    args = ap.parse_args()

    preds = _load_preds()
    sched = _load_sched()
    excludes = [x.strip() for x in args.exclude.split(",") if x.strip()]

    # Determine which week(s) to evaluate
    done = _completed_weeks(sched)
    if not done:
        print("[EVAL] No completed weeks with scores yet.")
        return 0

    if args.season is not None and args.week is not None:
        target_weeks = [(args.season, args.week)]
    else:
        # last `window` completed weeks
        target_weeks = done[-max(1, args.window):]

    # Attach outcomes for all chosen weeks
    frames = []
    for (ssn, wk) in target_weeks:
        frames.append(_attach_week(preds, sched, ssn, wk, excludes))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    if df.empty:
        print(f"[EVAL] No matches found for weeks {target_weeks}.")
        return 0

    # Report metrics on the last week individually (record you care about)
    ssn_last, wk_last = target_weeks[-1]
    last = df[(df["season"]==ssn_last) & (df["week"]==wk_last)]
    m_last = _metrics(last["y_true"], last["p_home_raw"])
    print(f"[EVAL] Week {wk_last}, {ssn_last} — n={m_last['n']}  "
          f"ACC={m_last['accuracy']:.3f}  BRIER={m_last['brier']:.4f}  LOGLOSS={m_last['logloss']:.4f}")

    # Fit calibrator on the whole window
    m_before = _metrics(df["y_true"], df["p_home_raw"])
    w = b = None
    if len(df) >= args.min_samples:
        w, b = _fit_platt(df["y_true"], df["p_home_raw"])
        p_cal = _apply_platt(df["p_home_raw"], w, b)
        m_after = _metrics(df["y_true"], p_cal)
        os.makedirs(os.path.dirname(CALIB_PATH), exist_ok=True)
        with open(CALIB_PATH, "w") as f:
            json.dump({
                "type": "platt",
                "w": w, "b": b,
                "window_weeks": target_weeks,
                "fitted_on": {"n": int(len(df))},
                "metrics_before": m_before,
                "metrics_after": m_after
            }, f, indent=2)
        print(f"[CAL]  Fitted on {len(df)} games across {len(target_weeks)} week(s). "
              f"w={w:.3f}, b={b:.3f}  -> wrote {CALIB_PATH}")
    else:
        print(f"[CAL]  Not enough games to fit calibrator (n={len(df)} < {args.min_samples}). Skipping.")

    # Update weekly record log for the LAST week only (website banner)
    if not args.no_write_weekly_log:
        wins = int(((last["p_home_raw"] >= 0.5) == (last["y_true"] == 1)).sum())
        losses = int(len(last) - wins)
        os.makedirs(os.path.dirname(WEEKLY_LOG), exist_ok=True)
        log = []
        if os.path.exists(WEEKLY_LOG):
            try:
                with open(WEEKLY_LOG, "r") as f:
                    log = json.load(f)
            except Exception:
                log = []
        log = [e for e in log if not (e.get("season")==ssn_last and e.get("week")==wk_last)]
        note = ("Excluded: " + ", ".join(excludes)) if excludes else ""
        log.append({"season": ssn_last, "week": wk_last, "wins": wins, "losses": losses, "notes": note})
        log.sort(key=lambda e: (e["season"], e["week"]))
        with open(WEEKLY_LOG, "w") as f:
            json.dump(log, f, indent=2)
        print(f"[EVAL] updated {WEEKLY_LOG}: {wins}-{losses} (week {wk_last}, {ssn_last})")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
