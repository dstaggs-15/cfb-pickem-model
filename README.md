# CFB Pick'em Model — Notebook

This project is a complete, automated data and machine learning pipeline for predicting weekly NCAA college football games. It is designed for transparency, reproducibility, and robust performance by leveraging a multi-faceted feature set, a carefully designed training strategy, and a powerful, heavily-regularized predictive model.

The entire system is orchestrated via **GitHub Actions** and serves its predictions—complete with explainability breakdowns—through a static **GitHub Pages** website.

### How it Works (The Short Version)
1.  **Data Pipeline:** Gathers years of historical game stats, schedules, and betting lines. It cleans this raw data, pivots it into a usable format, and engineers advanced features. The key output is a clean `training.parquet` file.
2.  **Model Training:** A single, powerful Gradient Boosting model is trained on all historical data. The model is heavily penalized (**regularized**) to prevent it from relying too much on any single feature, forcing it to consider a wide range of evidence for each prediction.
3.  **Prediction & Explanation:** The trained model is used to predict upcoming games. For each prediction, a second tool (**SHAP**) analyzes the model's "brain" to determine the top factors that influenced the pick. The final output is a `predictions.json` file containing the pick, win probability, and a human-readable explanation.

---

## 1. High-Level Architecture

The system is a three-stage, fully automated pipeline that transforms raw sports data into calibrated and explainable game predictions. A key design principle is the **"single source of truth"** for feature engineering, which guarantees consistency between training and prediction.

1.  **Shared Feature Engineering (`/scripts/lib/features.py`):** This new, central script contains one master function, `create_feature_set`. This function is the heart of the pipeline. It takes raw data (schedules, team stats, etc.) and performs the entire feature creation process: pivoting, cleaning, calculating rolling averages, and merging context, market, and Elo data. Both the data-building and prediction scripts call this *exact same function*, eliminating any chance of a mismatch.

2.  **Data Ingestion (`build_dataset.py`):** This script orchestrates the creation of the training data. It loads all raw historical data and passes it to the shared `create_feature_set` function. It then saves the final, clean feature set as `training.parquet` and a metadata file, `train_meta.json`, which contains the master list of features the model will use.

3.  **Model Training (`train_model.py`):** This stage consumes the `training.parquet` file. It trains a **single, heavily regularized `HistGradientBoostingClassifier`**. Using a high regularization penalty (`l2_regularization=20.0`) forces the model to be "humble" and build a balanced perspective from all available features, rather than becoming over-reliant on a single dominant signal like Elo. The final calibrated model is saved as `model.joblib`.

4.  **Prediction (`predict.py`):** This stage loads the trained model. It also loads the raw data and the user-provided matchups, and passes them to the *exact same* `create_feature_set` function to generate features for the new games. This guarantees consistency. It then makes a prediction and uses the **SHAP** library to generate a breakdown of the top 5 factors that influenced the outcome. The final `predictions.json` is then written for the website.

---

## 2. Feature Engineering

The model's performance relies on a set of carefully engineered features.

### 2.1 "Season-Average Carry-Forward" Rolling Stats

To solve the "cold start" problem where the model has no data in Week 1, we use a robust carry-forward system for our rolling statistics.

* **Logic:** The model calculates a team's rolling average performance (e.g., `ppa`, `success_rate`) over its **last 5 games**.
* **The Carry-Forward:** For the first few weeks of a new season, this rolling window is "padded" with the team's **full-season average from the prior season**.
    * **Week 1:** The "rolling average" is simply the team's full-season average from last year.
    * **Week 2:** The average is a mix of 4 parts last year's average and 1 part the new game from Week 1.
    * **Week 6:** The average is now composed entirely of 5 new games from the current season.
* **Benefit:** This provides a smooth transition from a team's established performance to its current-season form and ensures the stats-based features have a meaningful voice from day one.



### 2.2 Game Context, Market Data, and Elo Rating

The model also uses the same powerful context features as before:

* **Game Context:** Rest days, travel distance, and flags for neutral site or postseason games.
* **Market-Derived Features:** The betting market is a powerful signal. We process historical betting lines and convert the point spread into an implied win probability (`market_home_prob`), which serves as a key feature.
* **Elo Rating:** A long-term power rating for each team that includes home-field advantage and off-season regression to the mean. It provides a stable, historical baseline of team strength.

---

## 3. Model Training and Explainability

The modeling stage is designed for robustness, reliability, and transparency.

### 3.1 Model Choice and Regularization

We use a single **`HistGradientBoostingClassifier`**. The key to our new, more balanced approach is a very high **L2 Regularization penalty (`l2_regularization=20.0`)**.

* **Analogy:** Think of regularization as a "team salary cap." Without it, the model might "spend" all of its predictive power on one superstar feature like `Elo`. The high penalty acts like a luxury tax, forcing the model to "distribute the salary" more evenly among all the features (`PPA`, `Success Rate`, etc.), resulting in a more balanced and less volatile prediction.

### 3.2 Probability Calibration

The raw probabilities from the model are passed through a `CalibratedClassifierCV` layer. This ensures that when the model predicts a "70% chance to win," that outcome happens approximately 70% of the time in the long run, making the predictions reliable.

### 3.3 Prediction Explainability with SHAP

Transparency is a primary goal. After a prediction is made, we use the **SHAP (SHapley Additive exPlanations)** library to analyze the model's decision.

* **How it Works:** SHAP is a sophisticated tool that can precisely calculate the impact of each feature on a single prediction. It answers the question: "How much did knowing the `Elo Home Prob` push the final prediction up or down?"
* **The Output:** For each game, we save the top 5 features that had the biggest impact, which are then displayed on the website in a human-readable format (e.g., "Strongly favors Iowa").

---
## 4. How to Use

* **Weekly Use:** To get new predictions, edit the `/docs/input/games.txt` file. Each line should be in the format `Away Team @ Home Team` for a regular game, or `Team 1 vs Team 2` for a neutral site game. Pushing this change will automatically trigger the GitHub Actions workflow.
* **Extending:** The new `/scripts/lib/features.py` file is the central place to add new features. Any feature created there will be automatically available for both training and prediction, ensuring consistency.
