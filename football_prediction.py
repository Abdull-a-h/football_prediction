"""
PREMIER LEAGUE MATCH PREDICTOR - COMPLETE UNIFIED VERSION
Combines historical CSV data + current season FBref scraping
All-in-one file - no external dependencies except libraries
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss, classification_report
import requests
import warnings
import plotly.express as px
from io import StringIO
import time

warnings.filterwarnings('ignore')

# Selenium imports (optional - only needed for FBref scraping)
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("⚠️  Selenium not installed. FBref scraping disabled.")
    print("   Install with: pip install selenium")


# ============================================================================
# PART 1: FBREF SCRAPER
# ============================================================================

class FBrefScraper:
    """
    Scrape player-level stats and current season fixtures from FBref
    """
    
    def __init__(self):
        self.driver = None
        self.team_urls = {
            'Aston Villa': 'https://fbref.com/en/squads/8602292d/Aston-Villa-Stats',
            'Nottm Forest': 'https://fbref.com/en/squads/e4a775cb/Nottingham-Forest-Stats',
            'Arsenal': 'https://fbref.com/en/squads/18bb7c10/Arsenal-Stats',
            'Liverpool': 'https://fbref.com/en/squads/822bd0ba/Liverpool-Stats',
            'Man City': 'https://fbref.com/en/squads/b8fd03ef/Manchester-City-Stats',
            'Chelsea': 'https://fbref.com/en/squads/cff3d9bb/Chelsea-Stats',
            'Tottenham': 'https://fbref.com/en/squads/361ca564/Tottenham-Hotspur-Stats',
            'Newcastle': 'https://fbref.com/en/squads/b2b47a98/Newcastle-United-Stats',
            'Man United': 'https://fbref.com/en/squads/19538871/Manchester-United-Stats',
            'Brighton': 'https://fbref.com/en/squads/d07537b9/Brighton-and-Hove-Albion-Stats',
            'West Ham': 'https://fbref.com/en/squads/7c21e445/West-Ham-United-Stats',
            'Wolves': 'https://fbref.com/en/squads/8cec06e1/Wolverhampton-Wanderers-Stats',
            'Fulham': 'https://fbref.com/en/squads/fd962109/Fulham-Stats',
            'Bournemouth': 'https://fbref.com/en/squads/4ba7cbea/Bournemouth-Stats',
            'Brentford': 'https://fbref.com/en/squads/cd051869/Brentford-Stats',
            'Everton': 'https://fbref.com/en/squads/d3fd31cc/Everton-Stats',
            'Crystal Palace': 'https://fbref.com/en/squads/47c64c55/Crystal-Palace-Stats',
            'Leicester City': 'https://fbref.com/en/squads/a2d435b3/Leicester-City-Stats',
            'Ipswich Town': 'https://fbref.com/en/squads/c0fc8580/Ipswich-Town-Stats',
            'Southampton': 'https://fbref.com/en/squads/33c895d4/Southampton-Stats'
        }
    
    def setup_driver(self):
        """Initialize Chrome driver"""
        if not SELENIUM_AVAILABLE:
            return False
        
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
            self.driver = webdriver.Chrome(options=chrome_options)
            return True
        except Exception as e:
            print(f"⚠️  Could not initialize Chrome driver: {e}")
            return False
    
    def close_driver(self):
        """Close the driver"""
        if self.driver:
            self.driver.quit()
    
    def scrape_team_fixtures(self, team_name):
        """Scrape current season fixtures and results for a team"""
        if team_name not in self.team_urls:
            return None
        
        url = self.team_urls[team_name]
        
        try:
            if not self.driver:
                if not self.setup_driver():
                    return None
            
            self.driver.get(url)
            time.sleep(3)
            
            page_source = self.driver.page_source
            tables = pd.read_html(StringIO(page_source))
            
            # Find the fixtures table
            fixtures_df = None
            for table in tables:
                if 'Date' in table.columns and 'Opponent' in table.columns:
                    fixtures_df = table
                    break
            
            if fixtures_df is None:
                return None
            
            fixtures_df = fixtures_df.copy()
            
            # Keep relevant columns
            relevant_cols = ['Date', 'Time', 'Comp', 'Round', 'Venue', 'Result', 
                           'GF', 'GA', 'Opponent', 'xG', 'xGA', 'Poss', 'Formation']
            
            available_cols = [col for col in relevant_cols if col in fixtures_df.columns]
            fixtures_df = fixtures_df[available_cols]
            
            # Convert types
            if 'GF' in fixtures_df.columns:
                fixtures_df['GF'] = pd.to_numeric(fixtures_df['GF'], errors='coerce')
            if 'GA' in fixtures_df.columns:
                fixtures_df['GA'] = pd.to_numeric(fixtures_df['GA'], errors='coerce')
            if 'xG' in fixtures_df.columns:
                fixtures_df['xG'] = pd.to_numeric(fixtures_df['xG'], errors='coerce')
            if 'xGA' in fixtures_df.columns:
                fixtures_df['xGA'] = pd.to_numeric(fixtures_df['xGA'], errors='coerce')
            
            fixtures_df['Team'] = team_name
            
            return fixtures_df
            
        except Exception as e:
            return None
    
    def get_recent_form_features(self, fixtures_df, n_matches=5):
        """Calculate form features from recent fixtures"""
        if fixtures_df is None or len(fixtures_df) == 0:
            return None
        
        completed = fixtures_df[fixtures_df['Result'].notna()].copy()
        
        if len(completed) == 0:
            return None
        
        recent = completed.tail(n_matches)
        
        points = 0
        wins = 0
        draws = 0
        losses = 0
        
        for _, match in recent.iterrows():
            result = match['Result']
            if result == 'W':
                points += 3
                wins += 1
            elif result == 'D':
                points += 1
                draws += 1
            elif result == 'L':
                losses += 1
        
        form_metrics = {
            'recent_points': points,
            'recent_wins': wins,
            'recent_draws': draws,
            'recent_losses': losses,
            'recent_gf': recent['GF'].sum() if 'GF' in recent.columns else 0,
            'recent_ga': recent['GA'].sum() if 'GA' in recent.columns else 0,
            'recent_xg': recent['xG'].mean() if 'xG' in recent.columns else None,
            'recent_xga': recent['xGA'].mean() if 'xGA' in recent.columns else None,
            'recent_poss': recent['Poss'].mean() if 'Poss' in recent.columns else None,
            'matches_analyzed': len(recent)
        }
        
        return form_metrics
    
    def integrate_fbref_features(self, team_name):
        """Get comprehensive FBref features for a team"""
        fixtures = self.scrape_team_fixtures(team_name)
        
        features = {}
        
        if fixtures is not None:
            form = self.get_recent_form_features(fixtures, n_matches=5)
            if form:
                features.update(form)
        
        return features


# ============================================================================
# PART 2: ENHANCED PREDICTOR
# ============================================================================

class PremierLeaguePredictor:
    """
    Complete Premier League match predictor
    - Loads historical CSV data
    - Calculates enhanced features
    - Integrates FBref current season data
    - Makes predictions with XGBoost
    """
    
    def __init__(self, use_xgboost=True, enable_fbref=True):
        """
        Initialize predictor
        
        Args:
            use_xgboost: Use XGBoost (True) or Random Forest (False)
            enable_fbref: Enable FBref scraping for current season data
        """
        if use_xgboost:
            self.model = XGBClassifier(
                objective='multi:softprob',
                num_class=3,
                max_depth=4,
                learning_rate=0.03,
                n_estimators=150,
                subsample=0.7,
                colsample_bytree=0.7,
                min_child_weight=3,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42
            )
        else:
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=10,
                random_state=42
            )
        
        self.match_data = None
        self.feature_columns = None
        self.label_mapping = {'A': 0, 'D': 1, 'H': 2}
        self.reverse_mapping = {0: 'A', 1: 'D', 2: 'H'}
        self.enable_fbref = enable_fbref and SELENIUM_AVAILABLE
        
        if enable_fbref and not SELENIUM_AVAILABLE:
            print("⚠️  FBref integration disabled (Selenium not available)")

    def get_football_data(self):
        """Download historical match data from Football-Data.co.uk"""
        print("\n" + "="*60)
        print("  LOADING HISTORICAL DATA")
        print("="*60)
        
        season_urls = {
            '2022-23': 'https://www.football-data.co.uk/mmz4281/2223/E0.csv',
            '2023-24': 'https://www.football-data.co.uk/mmz4281/2324/E0.csv',
            '2024-25': 'https://www.football-data.co.uk/mmz4281/2425/E0.csv'
        }

        all_matches = []
        for season, url in season_urls.items():
            try:
                print(f"  Downloading {season}...", end=" ")
                response = requests.get(url, timeout=10)
                season_data = pd.read_csv(StringIO(response.text), encoding='latin-1')
                season_data['Season'] = season
                all_matches.append(season_data)
                print(f"✓ {len(season_data)} matches")
            except Exception as e:
                print(f"✗ Error: {e}")

        if all_matches:
            self.match_data = pd.concat(all_matches, ignore_index=True)
            self.match_data['Date'] = pd.to_datetime(self.match_data['Date'], 
                                                     format='%d/%m/%Y', errors='coerce')
            self.match_data = self.match_data.sort_values('Date').reset_index(drop=True)
            print(f"\n✓ Total: {len(self.match_data)} matches loaded\n")
            return True
        return False

    def calculate_enhanced_features(self, data):
        """Calculate comprehensive features from historical data"""
        print("🔧 Calculating features from historical data...")
        df = data.copy()
        
        # Drop incomplete matches
        df = df.dropna(subset=['FTHG', 'FTAG', 'FTR'])
        
        # Initialize feature columns
        feature_cols = [
            'home_form_pts', 'away_form_pts', 'form_diff',
            'home_goals_scored_avg', 'away_goals_scored_avg',
            'home_goals_conceded_avg', 'away_goals_conceded_avg',
            'home_shots_avg', 'away_shots_avg',
            'home_shots_target_avg', 'away_shots_target_avg',
            'home_corners_avg', 'away_corners_avg',
            'home_win_streak', 'away_win_streak',
            'goal_diff_advantage', 'h2h_home_wins', 'h2h_away_wins'
        ]
        
        for col in feature_cols:
            df[col] = 0.0
        
        # Track team statistics
        team_stats = {}
        
        for idx in range(len(df)):
            if idx % 200 == 0:
                print(f"  Processing: {idx}/{len(df)} matches")
            
            row = df.iloc[idx]
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']
            
            # Initialize teams
            for team in [home_team, away_team]:
                if team not in team_stats:
                    team_stats[team] = {
                        'results': [], 'goals_for': [], 'goals_against': [],
                        'shots': [], 'shots_target': [], 'corners': []
                    }
            
            # Calculate features using ONLY past data
            home_stats = self._get_team_form(team_stats[home_team])
            away_stats = self._get_team_form(team_stats[away_team])
            
            # Populate features
            df.loc[idx, 'home_form_pts'] = home_stats['form_points']
            df.loc[idx, 'away_form_pts'] = away_stats['form_points']
            df.loc[idx, 'form_diff'] = home_stats['form_points'] - away_stats['form_points']
            df.loc[idx, 'home_goals_scored_avg'] = home_stats['goals_scored_avg']
            df.loc[idx, 'away_goals_scored_avg'] = away_stats['goals_scored_avg']
            df.loc[idx, 'home_goals_conceded_avg'] = home_stats['goals_conceded_avg']
            df.loc[idx, 'away_goals_conceded_avg'] = away_stats['goals_conceded_avg']
            df.loc[idx, 'home_shots_avg'] = home_stats['shots_avg']
            df.loc[idx, 'away_shots_avg'] = away_stats['shots_avg']
            df.loc[idx, 'home_shots_target_avg'] = home_stats['shots_target_avg']
            df.loc[idx, 'away_shots_target_avg'] = away_stats['shots_target_avg']
            df.loc[idx, 'home_corners_avg'] = home_stats['corners_avg']
            df.loc[idx, 'away_corners_avg'] = away_stats['corners_avg']
            df.loc[idx, 'home_win_streak'] = home_stats['win_streak']
            df.loc[idx, 'away_win_streak'] = away_stats['win_streak']
            
            # Goal difference advantage
            home_gd = home_stats['goals_scored_avg'] - home_stats['goals_conceded_avg']
            away_gd = away_stats['goals_scored_avg'] - away_stats['goals_conceded_avg']
            df.loc[idx, 'goal_diff_advantage'] = home_gd - away_gd
            
            # Head-to-head
            h2h = self._get_h2h_record(df, idx, home_team, away_team)
            df.loc[idx, 'h2h_home_wins'] = h2h['home_wins']
            df.loc[idx, 'h2h_away_wins'] = h2h['away_wins']
            
            # Update team stats AFTER computing features
            self._update_team_stats(team_stats, row, home_team, away_team)
        
        print(f"✓ Features calculated for {len(df)} matches\n")
        return df

    def _get_team_form(self, stats, n_games=5):
        """Calculate team form from last N games"""
        recent_results = stats['results'][-n_games:]
        recent_gf = stats['goals_for'][-n_games:]
        recent_ga = stats['goals_against'][-n_games:]
        recent_shots = stats['shots'][-n_games:]
        recent_st = stats['shots_target'][-n_games:]
        recent_corners = stats['corners'][-n_games:]

        
        win_streak = 0
        for result in reversed(recent_results):
            if result == 3:
                win_streak += 1
            else:
                break
        
        return {
            'form_points': sum(recent_results) if recent_results else 0,
            'goals_scored_avg': np.mean(recent_gf) if recent_gf else 1.5,
            'goals_conceded_avg': np.mean(recent_ga) if recent_ga else 1.5,
            'shots_avg': np.mean(recent_shots) if recent_shots else 12,
            'shots_target_avg': np.mean(recent_st) if recent_st else 4,
            'corners_avg': np.mean(recent_corners) if recent_corners else 5,
            'win_streak': win_streak
        }

    def _update_team_stats(self, team_stats, row, home_team, away_team):
        """Update team statistics after a match"""
        # Results
        if row['FTR'] == 'H':
            team_stats[home_team]['results'].append(3)
            team_stats[away_team]['results'].append(0)
        elif row['FTR'] == 'D':
            team_stats[home_team]['results'].append(1)
            team_stats[away_team]['results'].append(1)
        else:
            team_stats[home_team]['results'].append(0)
            team_stats[away_team]['results'].append(3)
        
        # Goals
        team_stats[home_team]['goals_for'].append(row.get('FTHG', 0))
        team_stats[home_team]['goals_against'].append(row.get('FTAG', 0))
        team_stats[away_team]['goals_for'].append(row.get('FTAG', 0))
        team_stats[away_team]['goals_against'].append(row.get('FTHG', 0))
        
        # Match stats
        team_stats[home_team]['shots'].append(row.get('HS', 12))
        team_stats[away_team]['shots'].append(row.get('AS', 12))
        team_stats[home_team]['shots_target'].append(row.get('HST', 4))
        team_stats[away_team]['shots_target'].append(row.get('AST', 4))
        team_stats[home_team]['corners'].append(row.get('HC', 5))
        team_stats[away_team]['corners'].append(row.get('AC', 5))

    def _get_h2h_record(self, df, current_idx, home_team, away_team, n_games=5):
        """Get head-to-head record"""
        h2h_matches = df.iloc[:current_idx][
            ((df.iloc[:current_idx]['HomeTeam'] == home_team) & 
             (df.iloc[:current_idx]['AwayTeam'] == away_team)) |
            ((df.iloc[:current_idx]['HomeTeam'] == away_team) & 
             (df.iloc[:current_idx]['AwayTeam'] == home_team))
        ].tail(n_games)
        
        home_wins = len(h2h_matches[
            ((h2h_matches['HomeTeam'] == home_team) & (h2h_matches['FTR'] == 'H')) |
            ((h2h_matches['AwayTeam'] == home_team) & (h2h_matches['FTR'] == 'A'))
        ])
        
        away_wins = len(h2h_matches) - home_wins - len(h2h_matches[h2h_matches['FTR'] == 'D'])
        
        return {'home_wins': home_wins, 'away_wins': away_wins}

    def train_model(self, test_size=0.15):
        """Train model with proper chronological split"""
        if self.match_data is None:
            print("❌ No data loaded!")
            return False

        print("="*60)
        print("  TRAINING MODEL")
        print("="*60)
        
        # Calculate features
        df = self.calculate_enhanced_features(self.match_data)
        
        # Remove early matches with insufficient history
        df = df.iloc[50:].reset_index(drop=True)
        
        # Define base features
        base_features = [
            'home_form_pts', 'away_form_pts', 'form_diff',
            'home_goals_scored_avg', 'away_goals_scored_avg',
            'home_goals_conceded_avg', 'away_goals_conceded_avg',
            'home_shots_avg', 'away_shots_avg',
            'home_shots_target_avg', 'away_shots_target_avg',
            'home_corners_avg', 'away_corners_avg',
            'home_win_streak', 'away_win_streak',
            'goal_diff_advantage', 'h2h_home_wins', 'h2h_away_wins'
        ]
        
        # Add proxy FBref features for training
        extended_features = base_features + [
            'home_current_season_pts', 'away_current_season_pts',
            'home_recent_xg', 'away_recent_xg',
            'home_recent_xga', 'away_recent_xga'
        ]
        
        # Create proxy features
        df['home_current_season_pts'] = df['home_form_pts']
        df['away_current_season_pts'] = df['away_form_pts']
        df['home_recent_xg'] = df['home_goals_scored_avg']
        df['away_recent_xg'] = df['away_goals_scored_avg']
        df['home_recent_xga'] = df['home_goals_conceded_avg']
        df['away_recent_xga'] = df['away_goals_conceded_avg']
        
        self.feature_columns = extended_features
        
        # CHRONOLOGICAL SPLIT
        n_test = int(len(df) * test_size)
        train_df = df.iloc[:-n_test]
        test_df = df.iloc[-n_test:]
        
        X_train = train_df[extended_features]
        y_train = train_df['FTR'].map(self.label_mapping)
        X_test = test_df[extended_features]
        y_test = test_df['FTR'].map(self.label_mapping)
        
        print(f"\n📊 Data Split:")
        print(f"  Training:   {len(X_train)} matches")
        print(f"  Testing:    {len(X_test)} matches (most recent)")
        print(f"  Features:   {len(extended_features)}")
        
        # Train
        print(f"\n⚙️  Training XGBoost...")
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        test_pred_proba = self.model.predict_proba(X_test)
        
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        test_logloss = log_loss(y_test, test_pred_proba)
        
        print(f"\n📈 Results:")
        print(f"  Train Accuracy: {train_acc:.1%}")
        print(f"  Test Accuracy:  {test_acc:.1%} ⬅️ Real performance")
        print(f"  Log Loss:       {test_logloss:.4f}")
        
        # Convert back to labels for classification report
        y_test_labels = [self.reverse_mapping[y] for y in y_test]
        test_pred_labels = [self.reverse_mapping[y] for y in test_pred]
        
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test_labels, test_pred_labels, 
                                   target_names=['Away Win', 'Draw', 'Home Win']))
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            self._show_feature_importance()
        
        return True
        
    def _show_feature_importance(self):
        """Display top features"""
        importance_df = pd.DataFrame({
            'Feature': self.feature_columns,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False).head(10)
        
        print(f"\n🏆 Top 10 Most Important Features:")
        for _, row in importance_df.iterrows():
            print(f"  {row['Feature']:<30} {row['Importance']:.4f}")

    def predict_match(self, home_team, away_team, use_fbref=True):
        
        print("\n" + "="*60)
        print(f"🎯 {home_team} vs {away_team}")
        print("="*60)
        
        # Get FBref current season data if enabled
        fbref_data = {}
        if use_fbref and self.enable_fbref:
            try:
                print("\n🌐 Loading FBref data...")
                scraper = FBrefScraper()
                home_fbref = scraper.integrate_fbref_features(home_team)
                away_fbref = scraper.integrate_fbref_features(away_team)
                scraper.close_driver()
                fbref_data = {'home': home_fbref, 'away': away_fbref}
                print("✓ FBref data loaded\n")
            except Exception as e:
                print(f"⚠️  FBref unavailable, using historical data only\n")
        
        # Find exact team names in historical data
        home_team_match = self._find_team_name(home_team)
        away_team_match = self._find_team_name(away_team)
        
        if not home_team_match or not away_team_match:
            print(f"❌ Team not found in historical data")
            print(f"   Available teams:")
            for team in sorted(self.match_data['HomeTeam'].unique()):
                print(f"   - {team}")
            return None, None
        
        print(f"📝 Using: {home_team_match} vs {away_team_match}\n")
        
        # Calculate features from historical data
        print("🔧 Calculating match features...")
        enhanced_data = self.calculate_enhanced_features(self.match_data)
        
        # Get recent matches for both teams
        home_matches = enhanced_data[
            (enhanced_data['HomeTeam'] == home_team_match) |
            (enhanced_data['AwayTeam'] == home_team_match)
        ]
        
        away_matches = enhanced_data[
            (enhanced_data['HomeTeam'] == away_team_match) |
            (enhanced_data['AwayTeam'] == away_team_match)
        ]
        
        if len(home_matches) == 0 or len(away_matches) == 0:
            print("❌ Insufficient historical data for these teams")
            return None, None
        
        # Extract most recent match data
        home_row = home_matches.tail(1).iloc[0]
        away_row = away_matches.tail(1).iloc[0]
        
        # Determine if team was home/away in their last match
        is_home = home_row['HomeTeam'] == home_team_match
        is_away = away_row['AwayTeam'] == away_team_match
        
        # Build feature dictionary
        match_features = {}
        
        # Extract features based on team's position in last match
        for feat in ['form_pts', 'goals_scored_avg', 'goals_conceded_avg',
                    'shots_avg', 'shots_target_avg', 'corners_avg', 'win_streak']:
            # Home team features
            col_prefix = 'home' if is_home else 'away'
            match_features[f'home_{feat}'] = home_row[f'{col_prefix}_{feat}']
            
            # Away team features
            col_prefix = 'away' if is_away else 'home'
            match_features[f'away_{feat}'] = away_row[f'{col_prefix}_{feat}']
        
        # Calculate derived features
        match_features['form_diff'] = match_features['home_form_pts'] - match_features['away_form_pts']
        
        home_gd = match_features['home_goals_scored_avg'] - match_features['home_goals_conceded_avg']
        away_gd = match_features['away_goals_scored_avg'] - match_features['away_goals_conceded_avg']
        match_features['goal_diff_advantage'] = home_gd - away_gd
        
        # Head-to-head record (last 5 matches)
        h2h_matches = enhanced_data[
            ((enhanced_data['HomeTeam'] == home_team_match) & (enhanced_data['AwayTeam'] == away_team_match)) |
            ((enhanced_data['HomeTeam'] == away_team_match) & (enhanced_data['AwayTeam'] == home_team_match))
        ].tail(5)
        
        home_h2h_wins = len(h2h_matches[
            ((h2h_matches['HomeTeam'] == home_team_match) & (h2h_matches['FTR'] == 'H')) |
            ((h2h_matches['AwayTeam'] == home_team_match) & (h2h_matches['FTR'] == 'A'))
        ])
        away_h2h_wins = len(h2h_matches) - home_h2h_wins - len(h2h_matches[h2h_matches['FTR'] == 'D'])
        
        match_features['h2h_home_wins'] = home_h2h_wins
        match_features['h2h_away_wins'] = away_h2h_wins
        
        # Integrate FBref current season features
        if fbref_data:
            home_fb = fbref_data.get('home', {})
            away_fb = fbref_data.get('away', {})
            
            # Use FBref data if available, otherwise use historical averages
            match_features['home_current_season_pts'] = home_fb.get('recent_points', match_features['home_form_pts'])
            match_features['away_current_season_pts'] = away_fb.get('recent_points', match_features['away_form_pts'])
            match_features['home_recent_xg'] = home_fb.get('recent_xg', match_features['home_goals_scored_avg'])
            match_features['away_recent_xg'] = away_fb.get('recent_xg', match_features['away_goals_scored_avg'])
            match_features['home_recent_xga'] = home_fb.get('recent_xga', match_features['home_goals_conceded_avg'])
            match_features['away_recent_xga'] = away_fb.get('recent_xga', match_features['away_goals_conceded_avg'])
        else:
            # Use historical data as proxies
            match_features['home_current_season_pts'] = match_features['home_form_pts']
            match_features['away_current_season_pts'] = match_features['away_form_pts']
            match_features['home_recent_xg'] = match_features['home_goals_scored_avg']
            match_features['away_recent_xg'] = match_features['away_goals_scored_avg']
            match_features['home_recent_xga'] = match_features['home_goals_conceded_avg']
            match_features['away_recent_xga'] = match_features['away_goals_conceded_avg']
        
        # Create DataFrame with features in correct order
        match_df = pd.DataFrame([match_features])[self.feature_columns]
        
        # Make prediction
        prediction_numeric = self.model.predict(match_df)[0]
        prediction = self.reverse_mapping[prediction_numeric]
        probabilities = self.model.predict_proba(match_df)[0]
        
        # Format results
        results = {'H': home_team, 'D': 'Draw', 'A': away_team}
        prob_dict = {'A': probabilities[0], 'D': probabilities[1], 'H': probabilities[2]}
        
        # Display prediction
        print(f"✓ Features calculated\n")
        print(f"🔮 PREDICTION: {results[prediction]}")
        print(f"\n📊 Probabilities:")
        for outcome in ['H', 'D', 'A']:
            prob = prob_dict[outcome]
            bar = "█" * int(prob * 40)
            confidence = "🔥" if prob > 0.6 else "✓" if prob > 0.45 else ""
            print(f"  {results[outcome]:<20} {prob:>6.1%} {bar} {confidence}")
        
        # Display key stats
        print(f"\n📈 Key Stats:")
        print(f"  Form (last 5):        {match_features['home_form_pts']:.0f} pts vs {match_features['away_form_pts']:.0f} pts")
        print(f"  Goals/Game:           {match_features['home_goals_scored_avg']:.2f} vs {match_features['away_goals_scored_avg']:.2f}")
        print(f"  Shots/Game:           {match_features['home_shots_avg']:.1f} vs {match_features['away_shots_avg']:.1f}")
        print(f"  Win Streak:           {match_features['home_win_streak']:.0f} vs {match_features['away_win_streak']:.0f}")
        
        if fbref_data and fbref_data.get('home') and fbref_data.get('away'):
            home_fb = fbref_data['home']
            away_fb = fbref_data['away']
            if home_fb.get('recent_xg') and away_fb.get('recent_xg'):
                print(f"  xG/Game (current):    {home_fb['recent_xg']:.2f} vs {away_fb['recent_xg']:.2f}")
        
        if len(h2h_matches) > 0:
            print(f"  H2H (last {len(h2h_matches)}):         {home_h2h_wins}-{len(h2h_matches[h2h_matches['FTR'] == 'D'])}-{away_h2h_wins}")
        
        print()
        
        return prediction, probabilities
    
    def list_available_teams(self):
        """List all teams in the historical data"""
        if self.match_data is None:
            print("❌ No data loaded")
            return []
        
        teams = sorted(self.match_data['HomeTeam'].unique())
        print("\n📋 Available Teams:")
        for i, team in enumerate(teams, 1):
            print(f"  {i:2}. {team}")
        print()
        return teams
    
    def _find_team_name(self, input_name):
        input_name_clean = input_name.strip().lower().replace("'", "").replace(".", "")
        for team in self.match_data['HomeTeam'].unique():
            team_clean = team.strip().lower().replace("'", "").replace(".", "")
            if input_name_clean == team_clean:
                return team
        return None
    

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*60)
    print("  PREMIER LEAGUE MATCH PREDICTOR")
    print("  With Player Analysis")
    print("="*60)

    predictor = PremierLeaguePredictor(use_xgboost=True, enable_fbref=True)

    if not predictor.get_football_data():
        return None

    if not predictor.train_model(test_size=0.15):
        return None
    
    predictor.list_available_teams()

    # Predict WITH player analysis
    predictor.predict_match(
        'Aston Villa', 
        'Nott\'m Forest', 
        use_fbref=True 
    )

    print("="*60)
    print("✅ COMPLETE!")
    print("="*60)
    
    return predictor


if __name__ == "__main__":
    predictor = main()