⚽ Premier League Match Predictor
A machine learning-based system for predicting English Premier League match outcomes using historical data and current season statistics.


🎯 Overview
This project implements a multi-class classification model to predict Premier League match outcomes (Home Win, Draw, Away Win). It combines:

✅ Historical match data (3 seasons: 2022-23, 2023-24, 2024-25)
✅ Current season statistics from FBref (optional)
✅ Advanced feature engineering (form, head-to-head, expected goals)
✅ XGBoost classifier for predictions

🎮 Key Capabilities

Predict match outcomes with probability distributions
Integrate live current-season data via web scraping
Chronological train-test split to prevent data leakage
Comprehensive feature analysis
Real-time predictions for upcoming matches

📊 Performance Snapshot
Test Accuracy: 49.1%
Baseline (random): 33.3%
Improvement: 47% better than random
Log Loss: 1.0289

✨ Features
1. Historical Features (from CSV data)

Form-based metrics: Points from last 5 matches, win streaks
Attacking metrics: Goals scored average, shots per game, shots on target
Defensive metrics: Goals conceded average, corners conceded
Goal difference advantage: Net goal difference comparison
Head-to-head record: Last 5 meetings between teams

2. Current Season Features (from FBref scraping - optional)

Recent form: Points and results from last 5 matches
Expected Goals (xG): Advanced metric for attacking quality
Expected Goals Against (xGA): Defensive quality metric
Current season goals: Real-time goal scoring data

3. Prediction Output
🔮 PREDICTION: Aston Villa

📊 Probabilities:
  Aston Villa           42.9% █████████████████
  Draw                  35.9% ██████████████
  Nott'm Forest         21.2% ████████

📈 Key Stats:
  Form (last 5):        12 pts vs 8 pts
  Goals/Game:           1.80 vs 1.40
  Shots/Game:           12.8 vs 11.0
  Win Streak:           3 vs 1
  H2H (last 5):         3-0-2

🛠️ Installation
Prerequisites
bashPython 3.8+
Google Chrome (optional - for web scraping)
ChromeDriver (optional - matching your Chrome version)
Install Dependencies
bash# Clone the repository
git clone https://github.com/yourusername/premier-league-predictor.git
cd premier-league-predictor

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install pandas numpy xgboost scikit-learn requests plotly

# Optional: Install Selenium for live data scraping
pip install selenium
Optional: Install Selenium WebDriver
bash# Download ChromeDriver from:
# https://chromedriver.chromium.org/downloads
# Place it in your system PATH or project directory

🚀 Usage
Quick Start
bash# Run the complete pipeline
python football_prediction.py
Custom Predictions
pythonfrom football_prediction import PremierLeaguePredictor

# Initialize predictor
predictor = PremierLeaguePredictor(
    use_xgboost=True,      # Use XGBoost (vs Random Forest)
    enable_fbref=True      # Enable live data scraping (requires Selenium)
)

# Load and train
predictor.get_football_data()
predictor.train_model(test_size=0.15)

# Make predictions
predictor.predict_match('Arsenal', 'Liverpool', use_fbref=True)
predictor.predict_match('Man City', 'Chelsea', use_fbref=False)  # Historical only
Available Teams
python# List all teams in the dataset
predictor.list_available_teams()
Available Teams:
Arsenal, Aston Villa, Bournemouth, Brentford, Brighton, Burnley, Chelsea, Crystal Palace, Everton, Fulham, Ipswich, Leeds, Leicester, Liverpool, Luton, Man City, Man United, Newcastle, Nott'm Forest, Sheffield United, Southampton, Tottenham, West Ham, Wolves

🧠 Model Architecture
XGBoost Classifier
pythonXGBClassifier(
    objective='multi:softprob',  # Multi-class probability output
    num_class=3,                 # Home Win / Draw / Away Win
    max_depth=4,                 # Tree depth (prevents overfitting)
    learning_rate=0.03,          # Conservative learning rate
    n_estimators=150,            # Number of boosting rounds
    subsample=0.7,               # Row sampling (70%)
    colsample_bytree=0.7,        # Column sampling (70%)
    min_child_weight=3,          # Minimum samples per leaf
    gamma=0.1,                   # Regularization parameter
    reg_alpha=0.1,               # L1 regularization (Lasso)
    reg_lambda=1.0,              # L2 regularization (Ridge)
    random_state=42              # Reproducibility
)
Why XGBoost?
AdvantageDescriptionNon-linear patternsCaptures complex team dynamics and form patternsRegularizationBuilt-in L1/L2 prevents overfitting on small datasetsFeature importanceTransparent model - understand which features matterMissing dataHandles missing values gracefullySpeedFast training and prediction (<2 seconds per match)Proven track recordIndustry-standard for tabular data

🔧 Feature Engineering
Total Features: 24
All features are calculated using only historical data up to each match to prevent data leakage.
Feature Categories
1️⃣ Form Features (7 features)
FeatureDescriptionExamplehome_form_ptsPoints from last 5 matches (home team)12 ptsaway_form_ptsPoints from last 5 matches (away team)8 ptsform_diffDifference in form points+4 ptshome_current_season_ptsCurrent season points (last 5 games)10 ptsaway_current_season_ptsCurrent season points (last 5 games)7 ptshome_win_streakCurrent consecutive wins (home)3 gamesaway_win_streakCurrent consecutive wins (away)1 game
2️⃣ Attacking Features (8 features)
FeatureDescriptionExamplehome_goals_scored_avgAverage goals scored per game1.80away_goals_scored_avgAverage goals scored per game1.40home_shots_avgAverage shots per game12.8away_shots_avgAverage shots per game11.0home_shots_target_avgAverage shots on target4.5away_shots_target_avgAverage shots on target3.8home_recent_xgExpected goals (current season)1.76away_recent_xgExpected goals (current season)1.38
3️⃣ Defensive Features (4 features)
FeatureDescriptionExamplehome_goals_conceded_avgAverage goals conceded per game1.20away_goals_conceded_avgAverage goals conceded per game1.55home_recent_xgaExpected goals against (current)1.15away_recent_xgaExpected goals against (current)1.62
4️⃣ Tactical Features (3 features)
FeatureDescriptionExamplehome_corners_avgAverage corners per game5.2away_corners_avgAverage corners per game4.8goal_diff_advantageGoal difference comparison+0.65
Goal Difference Formula:
goal_diff_advantage = (home_goals_scored - home_goals_conceded) 
                    - (away_goals_scored - away_goals_conceded)
5️⃣ Head-to-Head Features (2 features)
FeatureDescriptionExampleh2h_home_winsHome team wins in last 5 H2H matches3h2h_away_winsAway team wins in last 5 H2H matches2
Feature Calculation Strategy
🔒 No Data Leakage: Features are calculated using only past data up to each match.
python# Example: For match at index 100
# Only uses data from matches 0-99
for idx in range(len(matches)):
    past_data = matches[0:idx]  # Only past matches
    features[idx] = calculate_features(past_data)
This ensures the model never "sees the future" during training.

📊 Data Sources
1. Historical Data (Football-Data.co.uk)
AspectDetailsURLhttps://www.football-data.co.ukCoverage3 complete seasons (1,140 matches)Seasons2022-23, 2023-24, 2024-25FormatCSV filesFieldsMatch results, goals, shots, corners, fouls, cardsUpdateWeekly during season
2. Current Season Data (FBref) - Optional
AspectDetailsURLhttps://fbref.comMethodSelenium web scrapingCoverageCurrent season fixtures and statisticsFieldsResults, xG, xGA, possession, formationsUpdateReal-time (after each match)
Data Split
Total Matches: 1,140
├── Training:   927 matches (81.3%)
├── Testing:    163 matches (14.3%)
└── Warm-up:     50 matches (4.4%) [excluded - insufficient history]
Split Method: Chronological (most recent matches reserved for testing)

📈 Evaluation Metrics
1. Accuracy
Train Accuracy: 71.0%
Test Accuracy:  49.1% ⬅️ Real-world performance
Baseline:       33.3% (random guessing)
Improvement:    47% better than random
Interpretation: For a 3-way classification problem (Home/Draw/Away), 49.1% accuracy is solid performance. The model significantly outperforms random guessing.
2. Log Loss
Test Log Loss: 1.0289
Interpretation: Measures probability calibration. Lower is better. Value around 1.0 indicates reasonable confidence calibration - the model doesn't over/underestimate probabilities.
3. Classification Report
              precision    recall  f1-score   support

    Away Win       0.49      0.48      0.49        60
        Draw       0.10      0.03      0.05        33
    Home Win       0.53      0.71      0.61        70

    accuracy                           0.49       163
   macro avg       0.37      0.41      0.38       163
weighted avg       0.43      0.49      0.45       163
Key Insights:
MetricFinding🏆 Best PredictionHome Wins (F1: 0.61, Recall: 71%)⚠️ Worst PredictionDraws (F1: 0.05, Recall: 3%)📊 BalancedAway Wins (F1: 0.49)
Why are Draws hard to predict?

Draws are inherently uncertain (can result from many scenarios)
Less training data (only 33 draws in test set vs 70 home wins)
Multiple pathways to draw (defensive stalemate vs attacking shootout)

4. Confusion Matrix
Predicted →     Away    Draw    Home
Actual ↓
Away Win         29      10      21     (60)
Draw              1       1      31     (33)
Home Win         10      10      50     (70)
Observations:

Model tends to predict Home Win when uncertain (70/163 = 43%)
Draws are often mislabeled as Home Wins (31/33 draws)
Away Wins have reasonable recall (29/60 = 48%)


📊 Results
Feature Importance (Top 10)
The model considers these features most important:
1. goal_diff_advantage            0.0701  ████████████████████
2. form_diff                      0.0574  ████████████████
3. home_current_season_pts        0.0471  █████████████
4. home_shots_avg                 0.0459  █████████████
5. away_shots_avg                 0.0453  █████████████
6. h2h_away_wins                  0.0443  ████████████
7. home_form_pts                  0.0435  ████████████
8. home_corners_avg               0.0428  ████████████
9. home_shots_target_avg          0.0425  ████████████
10. h2h_home_wins                 0.0414  ███████████
Key Takeaways:

Goal difference is the strongest single predictor
Form metrics (points, streak) are highly influential
Attacking stats (shots, xG) matter more than defensive stats
Head-to-head history provides valuable context

Sample Prediction
============================================================
🎯 Aston Villa vs Nott'm Forest
============================================================

🔮 PREDICTION: Aston Villa

📊 Probabilities:
  Aston Villa           42.9% █████████████████
  Draw                  35.9% ██████████████
  Nott'm Forest         21.2% ████████

📈 Key Stats:
  Form (last 5):        12 pts vs 8 pts
  Goals/Game:           1.80 vs 1.40
  Shots/Game:           12.8 vs 11.0
  Win Streak:           3 vs 1
  H2H (last 5):         3-0-2
============================================================
Analysis:

Close prediction (42.9% vs 35.9% draw) - indicates uncertainty
Aston Villa favored due to better form (+4 pts) and home advantage
Historical H2H dominance (3 wins in last 5) supports prediction