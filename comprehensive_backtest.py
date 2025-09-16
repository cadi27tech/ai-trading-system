import yfinance as yf
import pandas as pd
import numpy as np
import sqlite3
import talib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def run_comprehensive_backtest():
    """Run actual backtest on 2-3 years of historical data using your exact system"""
    
    print("🚀 Starting Comprehensive Historical Backtest...")
    
    # 1. Load historical market data (2022-2025)
    symbols = ['SPY', 'QQQ', 'GLD']
    start_date = "2022-01-01"
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"📊 Downloading data from {start_date} to {end_date}")
    
    all_data = {}
    for symbol in symbols:
        print(f"  Downloading {symbol}...")
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)
        if len(df) > 0:
            all_data[symbol] = df
            print(f"    ✅ {len(df)} days downloaded")
        else:
            print(f"    ❌ No data for {symbol}")
    
    # 2. Apply your exact feature engineering (19 indicators)
    def calculate_all_indicators(df):
        """Calculate all 19 technical indicators exactly like your system"""
        
        # Convert to numpy arrays for TA-Lib
        high = df['High'].values.astype(np.float64)
        low = df['Low'].values.astype(np.float64)
        close = df['Close'].values.astype(np.float64)
        volume = df['Volume'].values.astype(np.float64)
        open_price = df['Open'].values.astype(np.float64)
        
        # Trend Indicators
        df['sma_20'] = talib.SMA(close, timeperiod=20)
        df['sma_50'] = talib.SMA(close, timeperiod=50)
        df['sma_200'] = talib.SMA(close, timeperiod=200)
        df['ema_12'] = talib.EMA(close, timeperiod=12)
        df['ema_26'] = talib.EMA(close, timeperiod=26)
        
        # Momentum Indicators
        df['rsi'] = talib.RSI(close, timeperiod=14)
        macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_hist'] = macd_hist
        
        # Volatility Indicators
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        df['bb_width'] = (bb_upper - bb_lower) / bb_middle * 100
        df['atr'] = talib.ATR(high, low, close, timeperiod=14)
        
        # Volume Indicators
        df['obv'] = talib.OBV(close, volume)
        df['ad_line'] = talib.AD(high, low, close, volume)
        
        # Price Action Features
        df['price_change'] = df['Close'].pct_change()
        df['volume_change'] = df['Volume'].pct_change()
        df['high_low_pct'] = (df['High'] - df['Low']) / df['Close'] * 100
        df['open_close_pct'] = (df['Close'] - df['Open']) / df['Open'] * 100
        
        # Target Variables
        df['next_open'] = df['Open'].shift(-1)
        df['next_close'] = df['Close'].shift(-1)
        df['target_direction'] = (df['next_close'] > df['next_open']).astype(int)
        df['next_return'] = (df['next_close'] - df['Close']) / df['Close']
        
        # Clean data
        df = df.dropna()
        return df
    
    # 3. Process all symbols
    processed_data = {}
    feature_names = [
        'sma_20', 'sma_50', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'macd_hist',
        'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'atr', 'obv', 'ad_line',
        'price_change', 'volume_change', 'high_low_pct', 'open_close_pct'
    ]
    
    for symbol in symbols:
        if symbol in all_data:
            print(f"🔧 Processing {symbol} indicators...")
            processed_data[symbol] = calculate_all_indicators(all_data[symbol])
            print(f"    ✅ {len(processed_data[symbol])} complete records")
    
    # 4. Walk-forward backtesting with rolling windows
    print("\n📈 Starting Walk-Forward Backtesting...")
    
    all_predictions = []
    transaction_cost = 0.001  # 0.1% per trade
    
    for symbol in processed_data:
        print(f"\n📊 Backtesting {symbol}...")
        df = processed_data[symbol]
        
        # Prepare features and target
        X = df[feature_names].fillna(0)
        y = df['target_direction']
        dates = df.index
        returns = df['next_return'].fillna(0)
        
        # Walk-forward validation (6-month windows)
        window_size = 126  # ~6 months of trading days
        prediction_window = 21  # ~1 month predictions
        
        predictions = []
        
        for start_idx in range(window_size, len(X) - prediction_window, prediction_window):
            # Training window
            train_start = start_idx - window_size
            train_end = start_idx
            
            # Test window
            test_start = start_idx
            test_end = min(start_idx + prediction_window, len(X))
            
            # Split data
            X_train = X.iloc[train_start:train_end]
            y_train = y.iloc[train_start:train_end]
            X_test = X.iloc[test_start:test_end]
            y_test = y.iloc[test_start:test_end]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train ensemble models (exactly like your system)
            models = {
                'rf': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
                'gb': GradientBoostingClassifier(n_estimators=50, max_depth=6, random_state=42),
                'lr': LogisticRegression(max_iter=1000, random_state=42)
            }
            
            # Train all models
            trained_models = {}
            for model_name, model in models.items():
                model.fit(X_train_scaled, y_train)
                trained_models[model_name] = model
            
            # Generate ensemble predictions
            for i in range(len(X_test)):
                test_sample = X_test_scaled[i:i+1]
                
                # Get probabilities from each model
                rf_prob = trained_models['rf'].predict_proba(test_sample)[0]
                gb_prob = trained_models['gb'].predict_proba(test_sample)[0]
                lr_prob = trained_models['lr'].predict_proba(test_sample)[0]
                
                # Ensemble average
                ensemble_prob = (rf_prob + gb_prob + lr_prob) / 3
                ensemble_pred = np.argmax(ensemble_prob)
                confidence = np.max(ensemble_prob) * 100
                
                # Store prediction
                pred_date = dates[test_start + i]
                actual_direction = y_test.iloc[i]
                actual_return = returns.iloc[test_start + i]
                
                # Calculate strategy return
                if ensemble_pred == 1:  # Predicted UP
                    strategy_return = actual_return - transaction_cost
                else:  # Predicted DOWN (short or avoid)
                    strategy_return = -actual_return - transaction_cost
                
                predictions.append({
                    'date': pred_date,
                    'symbol': symbol,
                    'prediction': 'UP' if ensemble_pred == 1 else 'DOWN',
                    'confidence': confidence,
                    'actual_direction': 'UP' if actual_direction == 1 else 'DOWN',
                    'correct': int(ensemble_pred == actual_direction),
                    'actual_return': actual_return,
                    'strategy_return': strategy_return
                })
        
        all_predictions.extend(predictions)
        print(f"    ✅ Generated {len(predictions)} predictions")
    
    # 5. Calculate comprehensive performance metrics
    print("\n📊 Calculating Performance Metrics...")
    
    results_df = pd.DataFrame(all_predictions)
    
    # Overall performance
    performance_summary = {}
    
    for symbol in symbols:
        symbol_data = results_df[results_df['symbol'] == symbol]
        
        if len(symbol_data) > 0:
            # Basic metrics
            total_predictions = len(symbol_data)
            correct_predictions = symbol_data['correct'].sum()
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            
            # Return metrics
            strategy_returns = symbol_data['strategy_return'].values
            cumulative_return = (1 + strategy_returns).prod() - 1
            annualized_return = (1 + cumulative_return) ** (252 / len(strategy_returns)) - 1
            
            # Risk metrics
            volatility = np.std(strategy_returns) * np.sqrt(252)
            sharpe_ratio = (annualized_return - 0.02) / volatility if volatility > 0 else 0
            
            # Drawdown
            cumulative_curve = (1 + strategy_returns).cumprod()
            rolling_max = np.maximum.accumulate(cumulative_curve)
            drawdown = (cumulative_curve - rolling_max) / rolling_max
            max_drawdown = np.min(drawdown)
            
            # Win rate
            win_rate = accuracy * 100
            
            performance_summary[symbol] = {
                'Total Predictions': total_predictions,
                'Accuracy %': f"{accuracy*100:.1f}%",
                'Win Rate %': f"{win_rate:.1f}%",
                'Total Return %': f"{cumulative_return*100:.1f}%",
                'Annual Return %': f"{annualized_return*100:.1f}%",
                'Volatility %': f"{volatility*100:.1f}%",
                'Sharpe Ratio': f"{sharpe_ratio:.2f}",
                'Max Drawdown %': f"{max_drawdown*100:.1f}%"
            }
    
    # Save results to CSV
    results_df.to_csv('backtest_results.csv', index=False)
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(performance_summary).T
    
    print("\n🎉 Backtest Complete!")
    print("\n📊 REAL PERFORMANCE RESULTS:")
    print("=" * 50)
    print(summary_df.to_string())
    print("\n💾 Detailed results saved to 'backtest_results.csv'")
    
    return summary_df, results_df

if __name__ == "__main__":
    # Run the backtest
    summary, detailed_results = run_comprehensive_backtest()
