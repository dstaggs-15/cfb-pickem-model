# scripts/predict.py
import os
import json
import argparse
import numpy as np
import pandas as pd
import joblib

from .lib.features import create_feature_set
from .lib.parsing import parse_games_txt, load_aliases

DERIVED = "data/derived"
TRAIN_PARQUET = f"{DERIVED}/training.parquet"
MODEL_JOBLIB = f"{DERIVED}/model.joblib"
META_JSON = "docs/data/train_meta.json"
PREDICTIONS_JSON = "docs/data/predictions.json"

GAMES_TXT = "docs/input/games.txt"
ALIASES_JSON = "docs/input/aliases.json"
MANUAL_LINES_CSV = "docs/input/manual_lines.csv"

RAW_DIR = "data/raw/cfbd"
SCHED_CSV = f"{RAW_DIR}/cfb_schedule.csv"
STATS_CSV = f"{RAW_DIR}/cfb_game_team_stats.csv"
LINES_CSV = f"{RAW_DIR}/cfb_lines.csv"
TEAMS_CSV = f"{RAW_DIR}/cfbd_teams.csv"
VENUES_CSV = f"{RAW_DIR}/cfbd_venues.csv"
TALENT_CSV = f"{RAW_DIR}/cfbd_talent.csv"


def _load_df(path):
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--week", type=int, default=5, help="Current week of the season for labeling predictions (default 5).")
    args = ap.parse_args()

    # ----- 1) Load trained model + metadata -----
    meta = {}
    if os.path.exists(META_JSON):
        with open(META_JSON, "r") as f:
            meta = json.load(f)
    else:
        print(f"[WARN] {META_JSON} not found; continuing without meta.")

    if not os.path.exists(MODEL_JOBLIB):
        raise FileNotFoundError(f"{MODEL_JOBLIB} not found. Train the model first.")

    model_payload = joblib.load(MODEL_JOBLIB)
    model = model_payload.get("model", None)
    base_estimator = model_payload.get("base_estimator", None)
    if model is None:
        raise RuntimeError("Loaded model payload is missing 'model'.")

    features = meta.get("features", None)
    if not features:
        # Fallback: infer from the training parquet if meta missing
        if os.path.exists(TRAIN_PARQUET):
            train_df = pd.read_parquet(TRAIN_PARQUET)
            id_cols = ["game_id","season","week","home_team","away_team","neutral_site","home_points","away_points"]
            features = [c for c in train_df.columns if c not in id_cols]
        else:
            raise RuntimeError("Feature list not found in meta and training parquet is missing.")

    market_params = meta.get("market_params", None)

    # ----- 2) Load raw tables -----
    schedule = _load_df(SCHED_CSV)
    team_stats = _load_df(STATS_CSV)
    lines_df = _load_df(LINES_CSV)
    teams_df = _load_df(TEAMS_CSV)
    venues_df = _load_df(VENUES_CSV)
    talent_df = _load_df(TALENT_CSV)
    manual_lines_df = _load_df(MANUAL_LINES_CSV)

    # ----- 3) Parse games to predict from docs/input/games.txt -----
    aliases = load_aliases(ALIASES_JSON) if os.path.exists(ALIASES_JSON) else {}
    games = parse_games_txt(GAMES_TXT, aliases=aliases)
    if not games:
        raise RuntimeError(f"No games found in {GAMES_TXT}. Fill it, commit, and rerun.")

    predict_df = pd.DataFrame(games)
    predict_df["game_id"] = [f"predict_{i}" for i in range(len(predict_df))]
    # Force predictions to the next season relative to historical max
    hist_max_season = pd.to_numeric(schedule["season"], errors="coerce").max()
    predict_df["season"] = int(hist_max_season) + 1 if pd.notna(hist_max_season) else 2025
    predict_df["week"] = int(args.week)
    predict_df["home_points"] = np.nan
    predict_df["away_points"] = np.nan

    # ----- 4) Build features using unified engine -----
    X, _built_feats = create_feature_set(
        schedule=schedule,
        team_stats=team_stats,
        venues_df=venues_df,
        teams_df=teams_df,
        talent_df=talent_df,
        lines_df=lines_df,
        manual_lines_df=manual_lines_df,
        games_to_predict_df=predict_df
    )

    # Keep only predicted rows
    X = X[X["game_id"].str.startswith("predict_")].reset_index(drop=True)

    # Add market mapping if available
    if market_params and "a" in market_params and "b" in market_params and "spread_home" in X.columns:
        a, b = market_params["a"], market_params["b"]
        X["market_home_prob"] = 1.0 / (1.0 + np.exp(-(a + b * (-X["spread_home"]))))
    elif "market_home_prob" not in X.columns:
        X["market_home_prob"] = 0.5

    # Ensure all expected features exist + are numeric
    for col in features:
        if col not in X.columns:
            X[col] = 0.0
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)

    # ----- 5) Predict + SHAP -----
    probs = model.predict_proba(X[features])[:, 1]
    explanation = [[] for _ in range(len(X))]

    # Optional SHAP (only if base_estimator available)
    try:
        import shap
        if base_estimator is not None and os.path.exists(TRAIN_PARQUET):
            train_df = pd.read_parquet(TRAIN_PARQUET)
            explainer = shap.TreeExplainer(base_estimator, train_df[features])
            shap_values = explainer.shap_values(X[features])
            out_expl = []
            for i in range(len(X)):
                sv = shap_values[i]
                rank = sorted(
                    [{"feature": n, "value": float(v)} for n, v in zip(features, sv)],
                    key=lambda x: abs(x["value"]), reverse=True
                )
                out_expl.append(rank)
            explanation = out_expl
    except Exception as e:
        print(f"[WARN] SHAP explanation skipped: {e}")

    # ----- 6) Write predictions.json -----
    output = []
    for i in range(len(X)):
        prob = float(probs[i])
        home_team = X["home_team"].iloc[i]
        away_team = X["away_team"].iloc[i]
        neutral_site = bool(X["neutral_site"].iloc[i]) if "neutral_site" in X.columns else False
        pick = home_team if prob >= 0.5 else away_team

        output.append({
            "home_team": home_team,
            "away_team": away_team,
            "neutral_site": neutral_site,
            "model_prob_home": prob,
            "pick": pick,
            "explanation": explanation[i]
        })

    save_json(PREDICTIONS_JSON, {"games": output})
    print(f"✅ wrote {len(output)} predictions to {PREDICTIONS_JSON}")


if __name__ == "__main__":
    main()
