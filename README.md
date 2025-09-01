# CFB Pick’em Model — Engineering Notebook

Predict weekly winners for ESPN College Pick’em using a transparent, reproducible pipeline: **multi-season data**, **last-5 form**, **market lines**, **rest/travel**, **calibrated logistic regression**, and **a smarter Elo** — all glued together by GitHub Actions and rendered on GitHub Pages.

> Educational project. Not betting advice. Use responsibly.

---

## 0) TL;DR (How to use)

* Put matchups in `docs/input/games.txt` (one per line).
  Examples:

  * `Ohio State @ Texas`
  * `LSU vs Clemson`
  * `South Carolina, Virginia`

* (Optional) Put manual lines in `docs/input/lines.csv`:

  ```
  home,away,spread,over_under
  Texas Longhorns,Ohio State Buckeyes,-3.5,55.5
  ```

  (Spread is **home relative**: negative = home favored.)

* (One time) Add your CFBD API key as repo secret **CFBD\_API\_KEY**.
  Settings → Secrets and variables → Actions → *New repository secret*.

* The workflow:

  * **Daily** pulls history from CollegeFootballData (CFBD).
  * **Hourly** re-trains + predicts, writing `docs/data/predictions.json`.
  * GitHub Pages serves `docs/` → your site renders color-coded “bubbles.”

---

## 1) Repository layout

```
/scripts
  fetch_cfbd.py          # Pulls multi-season FBS schedule, per-team stats, lines, venues, teams, talent
  train_and_predict.py   # Feature engineering, model training, calibration, Elo, ensemble, predictions
  requirements.txt       # Python dependencies

/data/raw/cfbd
  cfb_schedule.csv            # schedule with scores
  cfb_game_team_stats.csv     # long-form per-game team stats
  cfb_lines.csv               # market lines (multiple providers; we take median)
  cfbd_venues.csv             # venue metadata (+coords, elevation)
  cfbd_teams.csv              # team metadata (+coords, conference)
  cfbd_talent.csv             # team talent by season (preseason priors)

/docs
  index.html
  app.js                      # fetches predictions.json, renders cards/bubbles w/ team colors
  styles.css
  /data/predictions.json      # output written by the workflow
  /input/games.txt            # your weekly matchups
  /input/lines.csv            # optional manual lines for next week (spread, total)
  /input/aliases.json         # optional name aliases (e.g., "hawaii" -> "Hawai'i Rainbow Warriors")

/.github/workflows
  predict.yml            # CI: fetch history (daily), train+predict (hourly), publish JSON
```

---

## 2) Data sources (what we actually use)

* **Schedule** (`cfb_schedule.csv`): season, week, teams, scores, neutral flag, date, venue id.
* **Per-team game stats** (`cfb_game_team_stats.csv`): long-form “box score” (e.g., `thirdDownEff` as “3-of-9”).
* **Market lines** (`cfb_lines.csv`): multiple books over time (we take **per-game median** of `spread`, `over_under`).
* **Venues** (`cfbd_venues.csv`): latitude/longitude, elevation, capacity.
* **Teams** (`cfbd_teams.csv`): team lat/lon (campus), conference.
* **Talent** (`cfbd_talent.csv`): season-level composite talent metric (recruiting proxy).

> The training script **prefers local CFBD CSVs**. If they don’t exist, it falls back to a public snapshot (older).

---

## 3) End-to-end dataflow (the big picture)

```
CFBD API  ─→  /data/raw/cfbd/*.csv  ─→  Feature Engineering  ─→  Models  ─→  Ensemble  ─→  predictions.json  ─→  Web UI
            (daily fetch job)            (stats + lines +      (Calibrated   (Elo ⊕ Stats)      (hourly)             (GitHub Pages)
                                          rest/travel)          Logistic +
                                                                Better Elo)
```

---

## 4) Input formats you control

### 4.1 `docs/input/games.txt`

Accepted patterns (one per line):

* `Away @ Home`
* `Home vs Away`
* `Home, Away`
* `# comments allowed`

Names are normalized via a large alias map plus `aliases.json`. Canonical names are what appear in the schedule CSV (e.g., `Texas Longhorns`).

### 4.2 `docs/input/lines.csv` (optional but powerful)

```
home,away,spread,over_under
Texas Longhorns,Ohio State Buckeyes,-3.5,55.5
```

* **Spread is home-relative**: negative means home is favored.
* If not supplied, the model still works (it just sets spread/total to 0 for predictions).

### 4.3 `docs/input/aliases.json` (optional)

```json
{
  "hawaii": "Hawai'i Rainbow Warriors",
  "miami fl": "Miami (FL) Hurricanes"
}
```

---

## 5) Feature engineering (what the model “sees”)

### 5.1 Box-score stats (per team, per game → numeric → rolling form)

From the long-form stats we extract and numericize:

* `totalYards`
* `netPassingYards`
* `rushingYards`
* `firstDowns`
* `turnovers`
* `sacks`
* `tacklesForLoss`
* `thirdDownEff` → converts `"3-of-9"` to `3/9`
* `fourthDownEff` → converts `"1-of-2"` to `1/2`
* `kickingPoints`

**Rolling form**: we compute **pre-game rolling means over last N=5** but separately for **home** and **away** contexts.

* For the home team: last-5 **home** games (pre-game shifted → no leakage)
* For the away team: last-5 **away** games (pre-game shifted)

Then we form **differentials** used for training/prediction:

```
diff_R5_totalYards = home_R5_totalYards - away_R5_totalYards
... same for each stat ...
```

Why: Home/away split captures venue-specific behaviors (e.g., some teams travel poorly).

### 5.2 Market features (strongest public signal)

* `spread_home` (median across books, sign relative to home)
* `over_under` (median across books)

If you provide `docs/input/lines.csv`, those numbers are used for **your future predictions** even if CFBD hasn’t posted finals yet.

### 5.3 Rest & travel features

* `rest_diff` = (home days since last game) − (away days since last game)
* `shortweek_diff` = (home short week ≤6 days ?1:0) − (away …)
* `bye_diff` = (home bye ≥13 days ?1:0) − (away …)
* `travel_diff_km`:

  * **Neutral site**: distance from each campus → venue; take home − away.
  * **True home game**: home travel = \~0, away travel = campus→home stadium.
* `neutral_site` ∈ {0,1}, `is_postseason` ∈ {0,1}

Why: short rest hurts; long travel + neutral shifts variance.

---

## 6) Models (plural) and how they combine

### 6.1 Better Elo (ratings over time)

Core expectation:

```
E(home beats away) = 1 / (1 + 10^(-(R_home + HFA - R_away)/400))
```

We add:

* **Home-field advantage (HFA)** = 55 Elo points (0 on neutral sites).
* **Margin-of-victory (MOV) scaling**: larger wins → slightly larger updates.
* **Early-season higher K**: Weeks 1–4 use `K = 32` (faster learning), then `K = 20`.
* **Off-season mean reversion**: pull every team `30%` back toward 1500 at season start.
* **Preseason priors**: bump initial rating by `~25 Elo * talent_zscore` (if `cfbd_talent.csv` exists).

Output per game: **Elo win probability** for the home side.

### 6.2 Calibrated logistic regression (on features)

Base model learns weights `w` over your differential/engineered vector `x`:

```
p = sigmoid(w·x + b)
```

We **calibrate** probabilities using the most recent completed season:

* If enough data: **isotonic** calibration (non-parametric).
* Else: **Platt** (sigmoid) calibration.

**Why calibration?** So that a “0.70” actually behaves like \~70% over many games (Brier score improves).

### 6.3 Season-ahead validation (no leakage)

We don’t just random-split; we compute metrics **year by year**:

* Train on seasons ≤ Y−2
* Calibrate on season Y−1
* Test on season Y

Report the **mean** across Y of:

* Accuracy
* AUC (ranking quality)
* Brier (probability calibration; lower is better)

This matches the real timeline: yesterday trains today; yesterday calibrates tomorrow.

### 6.4 The ensemble

Final probability is a **weighted average**:

```
P_home = 0.55 * P_home_from_Elo + 0.45 * P_home_from_CalibratedLogit
```

The weights are tunable constants at the top of `train_and_predict.py`:

```
ELO_WEIGHT  = 0.55
STAT_WEIGHT = 0.45
```

**Pick rule**: if `P_home ≥ 0.50` → pick home; else pick away.

---

## 7) Output format: `docs/data/predictions.json`

Minimal shape:

```json
{
  "generated_at": "2025-08-31T19:20:00Z",
  "model": "ensemble_last5 (Elo 55% + Calibrated stats 45%)",
  "metric": {
    "season_ahead_acc": 0.61,
    "season_ahead_auc": 0.66,
    "season_ahead_brier": 0.218
  },
  "unknown_teams": [],
  "games": [
    {
      "home": "Texas Longhorns",
      "away": "Ohio State Buckeyes",
      "home_prob": 0.5742,
      "away_prob": 0.4258,
      "pick": "Texas Longhorns"
    }
  ]
}
```

* `unknown_teams` lists any names we failed to match. Fix by editing `games.txt`/`aliases.json`.

---

## 8) Front-end (docs/) — what the webpage does

* `app.js` fetches `docs/data/predictions.json`.
* Renders each game as a **card/bubble** with:

  * Team names colored by a **134-team color map** (additions welcome).
  * Predicted winner, probabilities, small badge if neutral/bowl (if exposed).
* Vanilla JS + CSS for speed; nothing fancy required to view on GitHub Pages.

> Rule of thumb: lighter background, team colors inside “pills/bubbles”, big, readable numbers. This is for fast scanning.

---

## 9) GitHub Actions (CI/CD)

File: `.github/workflows/predict.yml`

* **Triggers**

  * `schedule`:

    * `15 9 * * *` → **daily** CFBD fetch (history updates).
    * `0 * * * *` → **hourly** train + predict (writes JSON).
  * `workflow_dispatch`: manual “Run workflow.”
  * `push` to `main` when you change code or inputs.

* **Jobs**

  * `fetch_cfbd`:

    * Requires secret `CFBD_API_KEY`.
    * Writes CSVs into `/data/raw/cfbd/`.
    * Commits only if files changed.
  * `train_predict`:

    * Installs deps.
    * Runs `scripts/train_and_predict.py`.
    * Commits `docs/data/predictions.json` if changed.

* **Concurrency**

  * Ensures a new run cancels an old one (`cancel-in-progress: true`) to avoid piles of queued jobs.

* **Commit strategy**

  * Uses bot identity and `[skip ci]` in messages to prevent loops.

**Common CI gotchas**

* “GitHub Actions is not permitted to create/approve PRs”: we commit directly to `main` to avoid PR permissions issues.
* Merge conflicts in `predictions.json`: concurrency + commit-only-when-changed avoids most. If you still get conflicts, delete the file and re-run; the job will re-write it cleanly.

---

## 10) How training “gets better over time”

* **Daily CFBD fetch** adds new games → both Elo and the training set grow.
* **Hourly re-train** re-fits on all available history (with calibration on the latest finished season).
* If you **add features** (e.g., weather), the model will learn from them on the next run.
* If you **enter manual lines** for upcoming games, predictions update immediately.

---

## 11) Math & metrics (quick but precise)

### 11.1 Logistic regression

* Decision function: `z = w·x + b`
* Probability (pre-calibration): `p = 1 / (1 + exp(-z))`
* Fitted by maximizing conditional log-likelihood.

### 11.2 Calibration

* **Isotonic regression**: non-parametric monotone mapping `p → p'` minimizing squared error on calibration set.
* **Platt scaling**: fit `p' = 1 / (1 + exp(Ap + B))` on calibration set.

### 11.3 Elo

* Expectation: as above.
* Update: `R_new = R_old + K * (score - expected)`
* MOV scaling factor `≈ ln(margin + 1) * 2.2 / ((|ΔR|*0.001) + 2.2)`
* Seasonal mean reversion: `R = 1500 + (R - 1500)*(1 - λ)` with `λ ≈ 0.30`.

### 11.4 Metrics

* **Accuracy**: % correct (sensitive to class balance; baseline ≈ home win rate).
* **AUC**: probability a random home-win game is ranked above a home-loss.
* **Brier**: mean squared error of predicted probabilities; lower is better; perfect=0.

---

## 12) Design choices (why we did it this way)

* **Home/away split rolling form** beats pooled averages when teams are venue-sensitive.
* **Lines** carry sharp crowd wisdom; we use them but still keep a model that can run without them.
* **Calibration** avoids over-confident 60% “coin flips.”
* **Season-ahead validation** mirrors reality and avoids time leakage.
* **Elo + Stats** ensemble: each covers the other’s blind spots (Elo = long-term strength; Stats = matchup form; Lines = market).

---

## 13) Running locally

```bash
# 1) Python env
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Deps
pip install -r scripts/requirements.txt

# 3) Fetch multi-season data (requires CFBD_API_KEY in your env)
export CFBD_API_KEY=YOUR_TOKEN_HERE
python scripts/fetch_cfbd.py

# 4) Train + predict
python scripts/train_and_predict.py

# 5) View site locally
# serve the docs/ folder with any static server (Python example):
python -m http.server --directory docs 8000
# open http://localhost:8000
```

---

## 14) Extending the model (roadmap)

* **Weather**: add wind speed and precipitation (feature engineering: wind × pass rate proxy).
* **QB/coach changes**: flags that reset or dampen rolling form early in season.
* **Depth charts/injuries**: not free, but a few binary features can move the needle.
* **Feature importance**: fit a tree model on same features to get SHAP/importance (for insight, not necessarily for picks).
* **Multi-output**: probability of cover vs win; total over/under lean.
* **Hyper-parameter search**: tune weights (ELO\_WEIGHT/STAT\_WEIGHT) by season-ahead score.

---

## 15) Troubleshooting & FAQs

**“Unknown teams keep showing up.”**
Add to `docs/input/aliases.json`, or fix spelling in `games.txt`. Check the predictions JSON’s `unknown_teams`.

**“Numbers changed a lot overnight.”**
The CFBD fetch pulled new games; Elo & rolling form updated; calibration shifted. That’s expected.

**“Why is my accuracy only \~60%?”**
Random years baseline (home-team bias) can be \~60–64%. Accuracy without lines is hard. Add lines + calibration + last-5 splits to get stable lift; don’t judge off a single 8-game week.

**“Do I need lines?”**
No. But including `spread_home` and `over_under` generally improves ranking and calibration.

**“What if I don’t have a CFBD key?”**
The trainer falls back to a public snapshot (older). You’ll still get predictions, just with less up-to-date data.

**“How often does it retrain?”**
Hourly by default; plus when you push; plus when you press **Run workflow**.

---

## 16) Ethics & compliance

* This is an **educational** project to learn ML, data wrangling, and automation.
* Check your local laws & platform ToS if you use predictions for wagering.
* Don’t over-fit small weekly slates; respect uncertainty (that’s why we calibrate).

---

## 17) Credits

* CollegeFootballData API (awesome community resource).
* GitHub Actions / Pages.
* The many open-source maintainers of pandas, NumPy, scikit-learn.

---

## 18) Appendix — Implementation details (deep-dive bullets)

* **Parsing “3-of-9”** → decimal efficiency before rolling.
* **Pre-game shift**: when computing rolling means, we **`shift(1)`** so a game never uses its own stats.
* **Neutral sites**: Elo HFA = 0; travel computed to venue coordinates.
* **Median lines**: robust to outlier books; if missing, zeros (model still OK).
* **Calibration selection**: isotonic if ≥400 games in calibration season, else Platt.
* **Season-ahead loop**: report mean of per-season metrics (not one giant pool).
* **Ensemble stability**: we weight Elo slightly higher early season; tuneable constants.
* **Git hygiene**: `[skip ci]` on bot commits to avoid recursive triggers; concurrency avoids pile-ups.
* **Front-end perf**: tiny JSON, no frameworks, renders instantly on Pages.
* **Colors**: a JS object maps 134 FBS teams → primary/secondary hex; picks get a tinted background “pill.”

