# CFB Pick'em Model: An Notebook

This project is a complete, automated data and machine learning pipeline for predicting weekly NCAA college football games. It is designed for transparency and reliability, transforming raw historical data into accurate, explainable predictions that are served on a public-facing website.

The entire system is orchestrated via **GitHub Actions** and presents its predictions through a modern, interactive **GitHub Pages** website.



### How it Works
1.  **Data Download (One-Time)**: A manual workflow connects to a sports data API, downloading and caching over 20 years of historical game stats, schedules, and betting lines. This builds the model's vast library of football knowledge.
2.  **Weekly Prediction (Automated)**: When new games are added, an automated workflow kicks off. It uses the cached historical data to build analytical features and train a fresh, up-to-date machine learning model.
3.  **Prediction & Explanation**: The newly trained model predicts the outcome and win probability for the upcoming games. For each prediction, a second tool (**SHAP**) analyzes the model's "brain" to calculate the influence of every single factor. The final picks and their detailed explanations are then published to the website.

---
## The Website: An Interactive Experience
The project's output is an interactive website designed for clarity and trust.
* **Clean Interface**: The site features a modern dark theme, with each game presented in its own "bubble" or card.
* **Click to Expand**: Users can click on any game card to reveal a detailed breakdown of the prediction.
* **Visual Explanations**: Inside each card, a meter-style graph shows every factor the model considered. The color and length of the bar show which team was favored by that factor and by how much, providing full transparency into the model's reasoning.

---
## The Engineering Pipeline
The system is built on two distinct, automated GitHub Actions workflows that work together to ensure the system is both efficient and robust.

### Part 1: The "Fetch and Cache Data" Workflow
This is a manually-triggered workflow that serves one purpose: performing the heavy lifting.
* **What it Does**: It runs the `fetch_cfbd.py` script, which connects to the College Football Data API using a secret key and downloads a massive dataset of historical games.
* **The Smart Part (Caching)**: Instead of storing these large data files in the repository, it saves them to the **GitHub Actions cache**. This is like putting the data in a high-speed storage unit. This process takes about 30 minutes but only needs to be run once, or whenever a full historical refresh is desired.

### Part 2: The "Generate Weekly Predictions" Workflow
This is the main, fully automated workflow that runs weekly.
1.  **Restores Data**: Its first step is to instantly retrieve the raw data from the cache created by the fetch workflow. This skips the 30-minute download, making the process incredibly fast.
2.  **Builds Features**: It processes the raw data, calculating team performance metrics and adding contextual information for every game in history.
3.  **Trains the Model**: Using this complete, clean dataset, it trains a new Gradient Boosting model from scratch, ensuring the model's knowledge is always current.
4.  **Generates Predictions**: It uses the freshly trained model to predict the upcoming games and generates the detailed SHAP explanations.
5.  **Deploys to Website**: The final step is to automatically commit the `predictions.json` file back to the repository, which instantly updates the GitHub Pages website with the new picks.

---
## The Model's Brain: What It Looks For
The model's accuracy comes from a rich set of features that give it a deep understanding of each team and game.

* **Team Performance**: To understand how a team is playing *right now*, the model calculates rolling averages of key stats (like points per play and success rate) over the last 5 games. For the first few weeks of a season, this is blended with the team's final average from the previous season, giving it a stable performance baseline from Week 1.
* **Game Context**: The model considers more than just stats. It analyzes:
    * **Elo Rating**: A long-term power rating for each team.
    * **Betting Market**: The point spread from oddsmakers, which is a powerful signal of public perception.
    * **Travel & Rest**: How far the away team has to travel and whether one team has a rest advantage over the other.
    * **Venue**: Whether the game is at a home stadium or a neutral site.

---
## How the Model Thinks: Training & Explainability
The model is designed not just to be accurate, but to be humble and transparent.

### Training Philosophy
The model is trained with a very high **regularization penalty**.
* **Analogy**: Think of regularization as a "team salary cap." Without it, the model might "spend" all of its predictive power on one superstar feature like Elo. The high penalty forces it to "distribute the salary" more evenly among all the features (team stats, travel, etc.), resulting in a more balanced and robust prediction.

### Prediction Explainability
Every prediction on the website is accompanied by a full breakdown of the model's reasoning.
* **How it Works**: We use a tool called **SHAP** that can precisely calculate the impact of every feature on a single prediction. It answers the question: "How much did knowing the `Rest Advantage` push the final prediction up or down?"
* **The Output**: For each game, we show **all** the factors that had an influence. This list is displayed with visual meters on the website, giving users a complete and honest look inside the model's "brain."

---
## How to Use This Project
The system is designed to be almost completely hands-off.

* **Weekly Predictions**: To get new picks, simply edit the `/docs/input/games.txt` file with the new matchups and push the change. The main workflow will automatically run and update the site.
* **One-Time Setup**: The project requires a `CFBD_API_KEY` to be added to the repository's **Settings > Secrets and variables > Actions** to enable data fetching.
* **Refreshing All Data**: If you ever want to refresh the entire 20+ year historical dataset, you can manually run the **"Manual Fetch and Cache Data"** workflow from the repository's Actions tab.
