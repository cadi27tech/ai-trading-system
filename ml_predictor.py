import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import sqlite3
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

class AdvancedMLPredictor:
    def __init__(self, db_path='market_data.db'):
        """Initialize advanced ML predictor with multiple models"""
        self.db_path = db_path
        
        # Multiple ML models for ensemble
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
            'gradient_boost': GradientBoostingClassifier(n_estimators=50, max_depth=6, random_state=42),
            'logistic': LogisticRegression(max_iter=1000, random_state=42)
        }
        
        # Feature names (must match data_collector)
        self.feature_names = [
            'sma_20', 'sma_50', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'macd_hist',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'atr', 'obv', 'ad_line',
            'price_change', 'volume_change', 'high_low_pct', 'open_close_pct'
        ]
        
        self.scalers = {}
        self.trained_models = {}
        
    def load_training_data(self, symbol):
        """Load comprehensive training data"""
        print(f"📚 Loading training data for {symbol}...")
        
        conn = sqlite3.connect(self.db_path)
        
        # Get all features and target
        feature_cols = ', '.join(self.feature_names)
        query = f'''
            SELECT {feature_cols}, target_direction, date
            FROM daily_data 
            WHERE symbol = ? AND target_direction IS NOT NULL
            ORDER BY date
        '''
        
        df = pd.read_sql_query(query, conn, params=(symbol,))
        conn.close()
        
        if len(df) == 0:
            print(f"❌ No training data for {symbol}")
            return None, None, None
        
        # Prepare features and targets
        X = df[self.feature_names].fillna(0)
        y = df['target_direction']
        dates = pd.to_datetime(df['date'])
        
        print(f"✅ Loaded {len(df)} samples for {symbol}")
        return X, y, dates
    
    def train_symbol_models(self, symbol):
        """Train multiple models for a symbol"""
        print(f"\n🧠 Training models for {symbol}...")
        
        # Load data
        X, y, dates = self.load_training_data(symbol)
        if X is None:
            return None
        
        # Split data temporally (important for financial data)
        split_date = dates.quantile(0.8)  # Use 80% for training
        train_mask = dates <= split_date
        
        X_train, X_test = X[train_mask], X[~train_mask]
        y_train, y_test = y[train_mask], y[~train_mask]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Store scaler
        self.scalers[symbol] = scaler
        
        # Train all models
        model_results = {}
        trained_models = {}
        
        for model_name, model in self.models.items():
            print(f"   🔧 Training {model_name}...")
            
            try:
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Test model
                test_pred = model.predict(X_test_scaled)
                test_accuracy = accuracy_score(y_test, test_pred)
                
                # Cross-validation on training set
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
                
                model_results[model_name] = {
                    'test_accuracy': test_accuracy,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                }
                
                trained_models[model_name] = model
                
                print(f"      ✅ {model_name}: {test_accuracy:.3f} accuracy")
                
            except Exception as e:
                print(f"      ❌ {model_name} failed: {e}")
        
        # Save models
        self.trained_models[symbol] = trained_models
        
        # Save to disk
        model_data = {
            'models': trained_models,
            'scaler': scaler,
            'feature_names': self.feature_names,
            'results': model_results
        }
        
        joblib.dump(model_data, f'models_{symbol.lower()}.pkl')
        
        print(f"💾 {symbol} models saved")
        return model_results
    
    def train_all_models(self, symbols=['SPY', 'QQQ', 'GLD']):
        """Train models for all symbols"""
        print("🚀 Training models for all symbols...")
        
        all_results = {}
        
        for symbol in symbols:
            results = self.train_symbol_models(symbol)
            if results:
                all_results[symbol] = results
        
        print("\n📊 Training Summary:")
        for symbol, results in all_results.items():
            print(f"\n{symbol}:")
            for model_name, metrics in results.items():
                print(f"  {model_name}: {metrics['test_accuracy']:.3f} ± {metrics['cv_std']:.3f}")
        
        return all_results
    
    def load_models(self, symbol):
        """Load trained models for a symbol"""
        model_file = f'models_{symbol.lower()}.pkl'
        
        if os.path.exists(model_file):
            model_data = joblib.load(model_file)
            self.trained_models[symbol] = model_data['models']
            self.scalers[symbol] = model_data['scaler']
            return True
        return False
    
    def predict_ensemble(self, symbol, features):
        """Make ensemble prediction for a symbol"""
        if symbol not in self.trained_models:
            if not self.load_models(symbol):
                return None
        
        # Scale features
        scaler = self.scalers[symbol]
        features_scaled = scaler.transform([features])
        
        # Get predictions from all models
        predictions = {}
        probabilities = {}
        
        for model_name, model in self.trained_models[symbol].items():
            try:
                pred = model.predict(features_scaled)[0]  # ✅ FIXED: Get single value
                prob = model.predict_proba(features_scaled)[0]  # ✅ FIXED: Get single array
                
                predictions[model_name] = pred
                probabilities[model_name] = prob
            except:
                continue
        
        if not predictions:
            return None
        
        # Ensemble prediction (majority vote with confidence weighting)
        ensemble_prob = np.mean(list(probabilities.values()), axis=0)
        ensemble_pred = np.argmax(ensemble_prob)
        confidence = np.max(ensemble_prob) * 100
        
        return {
            'prediction': 'UP' if ensemble_pred == 1 else 'DOWN',
            'confidence': confidence,
            'up_probability': ensemble_prob[1] * 100,  # ✅ FIXED: Was [21]
            'down_probability': ensemble_prob[0] * 100,  # ✅ FIXED: Was just ensemble_prob
            'individual_models': {name: 'UP' if pred == 1 else 'DOWN' for name, pred in predictions.items()}
        }
    
    def predict_all_symbols(self, symbols=['SPY', 'QQQ', 'GLD']):
        """Get predictions for all symbols"""
        from data_collector import AdvancedDataCollector
        
        collector = AdvancedDataCollector()
        all_predictions = {}
        
        for symbol in symbols:
            features = collector.get_latest_features(symbol)
            if features is not None:
                prediction = self.predict_ensemble(symbol, features)
                if prediction:
                    all_predictions[symbol] = prediction
        
        return all_predictions
    
    def get_feature_importance(self, symbol):
        """Get feature importance from Random Forest model"""
        if symbol not in self.trained_models:
            if not self.load_models(symbol):
                return None
        
        if 'random_forest' in self.trained_models[symbol]:
            rf_model = self.trained_models[symbol]['random_forest']
            importance = dict(zip(self.feature_names, rf_model.feature_importances_))
            return sorted(importance.items(), key=lambda x: x[1], reverse=True)  # ✅ FIXED: Was x[21]
        
        return None

# Test the predictor
if __name__ == "__main__":
    predictor = AdvancedMLPredictor()
    results = predictor.train_all_models(['SPY', 'QQQ', 'GLD'])
    
    if results:
        predictions = predictor.predict_all_symbols(['SPY', 'QQQ', 'GLD'])
        print("\n🔮 Current Predictions:")
        for symbol, pred in predictions.items():
            print(f"{symbol}: {pred['prediction']} ({pred['confidence']:.1f}% confidence)")
