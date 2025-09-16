 import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class AdvancedBacktester:
    def __init__(self, db_path='market_data.db', initial_capital=100000):
        """Initialize comprehensive backtesting engine"""
        self.db_path = db_path
        self.initial_capital = initial_capital
        
        # Trading parameters
        self.transaction_cost = 0.001  # 0.1% per trade
        self.min_confidence = 0.55     # Minimum confidence to trade
        
    def load_historical_data(self, symbol, start_date=None):
        """Load historical data for backtesting"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT date, open, high, low, close, volume, target_direction, target_return
            FROM daily_data 
            WHERE symbol = ?
        '''
        params = [symbol]
        
        if start_date:
            query += ' AND date >= ?'
            params.append(start_date)
            
        query += ' ORDER BY date'
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        if len(df) > 0:
            df['date'] = pd.to_datetime(df['date'])
            return df.set_index('date')
        
        return None
    
    def generate_signals(self, symbol, data, model):
        """Generate trading signals using the trained model"""
        from data_collector import AdvancedDataCollector
        
        collector = AdvancedDataCollector()
        signals = []
        
        print(f"🔄 Generating signals for {symbol}...")
        
        # Get features for each day
        conn = sqlite3.connect(self.db_path)
        
        feature_query = '''
            SELECT date, sma_20, sma_50, ema_12, ema_26, rsi, macd, macd_signal, macd_hist,
                   bb_upper, bb_middle, bb_lower, bb_width, atr, obv, ad_line,
                   price_change, volume_change, high_low_pct, open_close_pct
            FROM daily_data 
            WHERE symbol = ?
            ORDER BY date
        '''
        
        features_df = pd.read_sql_query(feature_query, conn, params=(symbol,))
        conn.close()
        
        if len(features_df) == 0:
            return []
        
        feature_names = [
            'sma_20', 'sma_50', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'macd_hist',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'atr', 'obv', 'ad_line',
            'price_change', 'volume_change', 'high_low_pct', 'open_close_pct'
        ]
        
        for idx, row in features_df.iterrows():
            try:
                features = row[feature_names].fillna(0).values
                
                if len(features) == len(feature_names):
                    prediction = model.predict_ensemble(symbol, features)
                    
                    if prediction and prediction['confidence'] >= self.min_confidence * 100:
                        signals.append({
                            'date': pd.to_datetime(row['date']),
                            'signal': 1 if prediction['prediction'] == 'UP' else -1,
                            'confidence': prediction['confidence'] / 100
                        })
                    else:
                        signals.append({
                            'date': pd.to_datetime(row['date']),
                            'signal': 0,  # No trade
                            'confidence': prediction['confidence'] / 100 if prediction else 0
                        })
                        
            except Exception as e:
                signals.append({
                    'date': pd.to_datetime(row['date']),
                    'signal': 0,
                    'confidence': 0
                })
        
        return pd.DataFrame(signals).set_index('date') if signals else pd.DataFrame()
    
    def run_backtest(self, symbol, start_date='2022-01-01'):
        """Run comprehensive backtest for a symbol"""
        print(f"\n📈 Running backtest for {symbol}...")
        
        # Load model
        from ml_predictor import AdvancedMLPredictor
        model = AdvancedMLPredictor()
        
        if not model.load_models(symbol):
            print(f"❌ No model found for {symbol}")
            return None
        
        # Load historical data
        data = self.load_historical_data(symbol, start_date)
        if data is None or len(data) == 0:
            print(f"❌ No data found for {symbol}")
            return None
        
        # Generate signals
        signals = self.generate_signals(symbol, data, model)
        if len(signals) == 0:
            print(f"❌ No signals generated for {symbol}")
            return None
        
        # Merge data and signals
        combined = data.join(signals, how='inner')
        combined = combined.dropna()
        
        if len(combined) == 0:
            print(f"❌ No combined data for {symbol}")
            return None
        
        # Initialize backtest variables
        capital = self.initial_capital
        position = 0  # 0 = no position, 1 = long, -1 = short
        trades = []
        daily_returns = []
        daily_capital = []
        
        print(f"📊 Processing {len(combined)} trading days...")
        
        for i, (date, row) in enumerate(combined.iterrows()):
            daily_start_capital = capital
            
            # Get current signal
            signal = int(row['signal'])
            confidence = row['confidence']
            
            # Calculate returns
            if i > 0:
                prev_close = combined.iloc[i-1]['close']
                current_return = (row['close'] - prev_close) / prev_close
                
                # Strategy return based on position
                if position == 1:  # Long position
                    strategy_return = current_return
                    capital = capital * (1 + strategy_return - self.transaction_cost)
                elif position == -1:  # Short position
                    strategy_return = -current_return
                    capital = capital * (1 + strategy_return - self.transaction_cost)
                else:
                    strategy_return = 0
                
                daily_returns.append(strategy_return)
            else:
                daily_returns.append(0)
            
            # Trading logic
            if signal != 0 and position == 0:
                # Enter position
                position = signal
                trades.append({
                    'entry_date': date,
                    'entry_price': row['close'],
                    'signal': signal,
                    'confidence': confidence
                })
                
            elif signal == 0 and position != 0:
                # Exit position
                if trades and 'exit_date' not in trades[-1]:
                    trades[-1]['exit_date'] = date
                    trades[-1]['exit_price'] = row['close']
                    
                    # Calculate trade return
                    entry_price = trades[-1]['entry_price']
                    exit_price = row['close']
                    
                    if position == 1:  # Long
                        trade_return = (exit_price - entry_price) / entry_price
                    else:  # Short
                        trade_return = (entry_price - exit_price) / entry_price
                    
                    trades[-1]['return'] = trade_return
                    trades[-1]['profit'] = (daily_start_capital * trade_return) - (daily_start_capital * self.transaction_cost * 2)
                
                position = 0
            
            daily_capital.append(capital)
        
        # Close any open position
        if position != 0 and trades and 'exit_date' not in trades[-1]:
            last_date = combined.index[-1]
            last_price = combined.iloc[-1]['close']
            
            trades[-1]['exit_date'] = last_date
            trades[-1]['exit_price'] = last_price
            
            entry_price = trades[-1]['entry_price']
            if position == 1:
                trade_return = (last_price - entry_price) / entry_price
            else:
                trade_return = (entry_price - last_price) / entry_price
                
            trades[-1]['return'] = trade_return
            trades[-1]['profit'] = (daily_start_capital * trade_return) - (daily_start_capital * self.transaction_cost * 2)
        
        # Calculate performance metrics
        trades_df = pd.DataFrame(trades)
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'date': combined.index,
            'price': combined['close'].values,
            'signal': combined['signal'].values,
            'daily_return': daily_returns,
            'capital': daily_capital
        })
        
        results_df['strategy_cumulative'] = (1 + pd.Series(daily_returns)).cumprod()
        results_df['buy_hold_cumulative'] = combined['close'] / combined['close'].iloc[0]
        
        # Calculate benchmark returns
        benchmark_returns = combined['close'].pct_change().fillna(0)
        
        performance = self.calculate_performance_metrics(
            daily_returns, benchmark_returns.values, trades_df
        )
        
        print(f"✅ Backtest complete: {performance['total_return']:.2%} return, {performance['sharpe_ratio']:.2f} Sharpe")
        
        return {
            'symbol': symbol,
            'results': results_df,
            'trades': trades_df,
            'performance': performance,
            'start_date': start_date,
            'end_date': combined.index[-1].strftime('%Y-%m-%d')
        }
    
    def calculate_performance_metrics(self, strategy_returns, benchmark_returns, trades_df):
        """Calculate comprehensive performance metrics"""
        strategy_returns = pd.Series(strategy_returns).fillna(0)
        benchmark_returns = pd.Series(benchmark_returns).fillna(0)
        
        # Basic returns
        total_return = (1 + strategy_returns).prod() - 1
        benchmark_total_return = (1 + benchmark_returns).prod() - 1
        
        # Annualized metrics (assuming 252 trading days per year)
        days = len(strategy_returns)
        years = days / 252
        
        annualized_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        volatility = strategy_returns.std() * np.sqrt(252)
        
        # Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        excess_return = annualized_return - risk_free_rate
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + strategy_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Trade statistics
        if len(trades_df) > 0:
            winning_trades = len(trades_df[trades_df['return'] > 0])
            total_trades = len(trades_df)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            avg_win = trades_df[trades_df['return'] > 0]['return'].mean() if winning_trades > 0 else 0
            avg_loss = trades_df[trades_df['return'] < 0]['return'].mean() if total_trades > winning_trades else 0
            
            profit_factor = abs(avg_win * winning_trades / (avg_loss * (total_trades - winning_trades))) if avg_loss != 0 else np.inf
        else:
            total_trades = 0
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'benchmark_return': benchmark_total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'calmar_ratio': annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        }
    
    def run_multi_asset_backtest(self, symbols=['SPY', 'QQQ', 'GLD'], start_date='2022-01-01'):
        """Run backtest for all assets"""
        print("🚀 Running multi-asset backtest...")
        
        all_results = {}
        
        for symbol in symbols:
            result = self.run_backtest(symbol, start_date)
            if result:
                all_results[symbol] = result
        
        return all_results
    
    def save_backtest_results(self, results):
        """Save backtest results to database"""
        conn = sqlite3.connect(self.db_path)
        
        for symbol, result in results.items():
            performance_data = result['performance']
            
            # Save daily performance
            for idx, row in result['results'].iterrows():
                conn.execute('''
                    INSERT OR REPLACE INTO performance_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    row['date'].strftime('%Y-%m-%d'), symbol,
                    row['daily_return'], row['price'] / result['results']['price'].iloc[0] - 1,
                    row['strategy_cumulative'], row['buy_hold_cumulative'],
                    0,  # drawdown calculation would go here
                    performance_data['total_trades'], performance_data['win_rate'],
                    performance_data['sharpe_ratio']
                ))
        
        conn.commit()
        conn.close()
        print("💾 Backtest results saved to database")

# Test the backtester
if __name__ == "__main__":
    backtester = AdvancedBacktester()
    results = backtester.run_multi_asset_backtest(['SPY', 'QQQ', 'GLD'])
    
    if results:
        print("\n📊 Backtest Summary:")
        for symbol, result in results.items():
            perf = result['performance']
            print(f"{symbol}: {perf['total_return']:.2%} return, {perf['sharpe_ratio']:.2f} Sharpe, {perf['win_rate']:.2%} win rate")
        
        backtester.save_backtest_results(results)

