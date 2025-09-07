# ...imports and setup above...

def main():
    print("Generating predictions...")

    # Load model/meta
    model_payload = joblib.load(MODEL_JOBLIB)
    model = model_payload['model']
    base_estimator = model_payload.get('base_estimator')

    with open(META_JSON, 'r') as f:
        meta = json.load(f)
    features = meta["features"]
    market_params = meta.get("market_params", {})

    # Load raw data for context (season, games list, etc.)
    schedule = load_csv_local_or_url(LOCAL_SCHEDULE, FALLBACK_SCHEDULE_URL)
    schedule = ensure_schedule_columns(schedule)

    aliases = load_aliases(ALIASES_JSON)
    games_to_predict = parse_games_txt(GAMES_TXT, aliases)
    if not games_to_predict:
        save_json(PREDICTIONS_JSON, [])
        return

    predict_df = pd.DataFrame(games_to_predict)
    predict_df['game_id'] = [f"predict_{i}" for i in range(len(predict_df))]
    predict_df['season'] = schedule['season'].max()

    # Build features using the unified function (no extra args)
    X, _ = create_feature_set()

    # ---- the rest of your predict logic stays the same ----
    if market_params and 'a' in market_params and 'b' in market_params:
        a, b = market_params["a"], market_params["b"]
        X["market_home_prob"] = X["spread_home"].apply(
            lambda s: (1 / (1 + np.exp(-(a + b * (-(s))))) if pd.notna(s) else np.nan)
        )
    else:
        X["market_home_prob"] = 0.5

    for col in features:
        if col not in X.columns:
            X[col] = 0.0
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0.0)

    probs = model.predict_proba(X[features])[:, 1]

    shap_values = None
    if base_estimator:
        print("  Generating SHAP explanations...")
        train_df = pd.read_parquet(TRAIN_PARQUET)
        train_df_features = train_df[features]
        explainer = shap.TreeExplainer(base_estimator, train_df_features)
        shap_values = explainer.shap_values(X[features])

    output = []
    for i in range(len(X)):
        prob = probs[i]
        home_team = X['home_team'].iloc[i]
        away_team = X['away_team'].iloc[i]
        neutral_site = bool(X.get('neutral_site', pd.Series([0]*len(X))).iloc[i])
        pick = home_team if prob > 0.5 else away_team

        explanation = []
        if shap_values is not None:
            shap_row = shap_values[i]
            feature_names = X[features].columns
            explanation = sorted(
                [{'feature': name, 'value': val} for name, val in zip(feature_names, shap_row)],
                key=lambda x: abs(x['value']),
                reverse=True
            )

        output.append({
            'home_team': home_team,
            'away_team': away_team,
            'neutral_site': neutral_site,
            'model_prob_home': prob,
            'pick': pick,
            'explanation': explanation
        })

    save_json(PREDICTIONS_JSON, output)
    print(f"Successfully wrote {len(output)} predictions to {PREDICTIONS_JSON}")


if __name__ == "__main__":
    main()
