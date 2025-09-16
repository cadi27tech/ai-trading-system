import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import talib
import os

class AdvancedDataCollector:
    def __init__(self, alpha_vantage_key=None):
        """Initialize the advanced data collector with all indicators"""
        self.av_key = alpha_vantage_key
        self.db_path = 'market_data.db'
        self.setup_database()
        
        # Symbol definitions
        self.symbols_info = {
            'SPY': {'name': 'S&P 500', 'type': 'Stock Index'},
            'QQQ': {'name': 'NASDAQ', 'type': 'Tech Index'}, 
            'GLD': {'name': 'Gold', 'type': 'Commodity'}
        }
    
    def setup_database(self):
        """Create comprehensive database schema"""
        conn = sqlite3.connect(self.db_path)
        
        # Main data table with all indicators
        conn.execute('''
            CREATE TABLE IF NOT EXISTS daily_data (
                date TEXT,
                symbol TEXT,
                open REAL, high REAL, low REAL, close REAL, volume INTEGER,
                sma_20 REAL, sma_50 REAL, sma_200 REAL,
                ema_12 REAL, ema_26 REAL,
                rsi REAL, macd REAL, macd_signal REAL, macd_hist REAL,
                bb_upper REAL, bb_middle REAL, bb_lower REAL, bb_width REAL,
                atr REAL,
                obv REAL, ad_line REAL,
                price_change REAL, volume_change REAL,
                high_low_pct REAL, open_close_pct REAL,
                target_direction INTEGER, target_return REAL,
                PRIMARY KEY (date, symbol)
            )
        ''')
        
        # Predictions table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                date TEXT,
                symbol TEXT,
                prediction TEXT,
                confidence REAL,
                up_prob REAL, down_prob REAL,
                actual_direction TEXT,
                actual_return REAL,
                correct INTEGER,
                PRIMARY KEY (date, symbol)
            )
        ''')
        
        # Performance metrics table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                date TEXT,
                symbol TEXT,
                strategy_return REAL,
                benchmark_return REAL,
                cumulative_strategy REAL,
                cumulative_benchmark REAL,
                drawdown REAL,
                trades_count INTEGER,
                win_rate REAL,
                sharpe_ratio REAL,
                PRIMARY KEY (date, symbol)
            )
        ''')
        
        conn.commit()
        conn.close()
        print("✅ Database schema created")
    
    def download_stock_data(self, symbols=['SPY', 'QQQ', 'GLD'], period='3y'):
        """Download data for multiple symbols"""
        all_data = {}
        
        for symbol in symbols:
            print(f"📊 Downloading {symbol} ({self.symbols_info[symbol]['name']})...")
            try:
                stock = yf.Ticker(symbol)
                data = stock.history(period=period)
                if len(data) > 0:
                    all_data[symbol] = data
                    print(f"   ✅ {len(data)} days downloaded")
                else:
                    print(f"   ❌ No data for {symbol}")
            except Exception as e:
                print(f"   ❌ Error downloading {symbol}: {e}")
        
        return all_data
    
    def calculate_all_indicators(self, df):
        """Calculate comprehensive technical indicators"""
        print("🔧 Calculating advanced technical indicators...")
        
        # Convert ALL arrays to float64 for TA-Lib compatibility
        high = df['High'].values.astype(np.float64)
        low = df['Low'].values.astype(np.float64)
        close = df['Close'].values.astype(np.float64)
        volume = df['Volume'].values.astype(np.float64)
        open_price = df['Open'].values.astype(np.float64)
        
        # Moving Averages
        df['SMA_20'] = talib.SMA(close, timeperiod=20)
        df['SMA_50'] = talib.SMA(close, timeperiod=50)
        df['SMA_200'] = talib.SMA(close, timeperiod=200)
        df['EMA_12'] = talib.EMA(close, timeperiod=12)
        df['EMA_26'] = talib.EMA(close, timeperiod=26)
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        df['MACD'] = macd
        df['MACD_Signal'] = macd_signal
        df['MACD_Hist'] = macd_hist
        
        # RSI
        df['RSI'] = talib.RSI(close, timeperiod=14)
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        df['BB_Upper'] = bb_upper
        df['BB_Middle'] = bb_middle
        df['BB_Lower'] = bb_lower
        df['BB_Width'] = (bb_upper - bb_lower) / bb_middle * 100
        
        # ATR (Average True Range)
        df['ATR'] = talib.ATR(high, low, close, timeperiod=14)
        
        # Volume Indicators - NOW FIXED!
        df['OBV'] = talib.OBV(close, volume)
        df['AD_Line'] = talib.AD(high, low, close, volume)
        
        # Price Action Features
        df['Price_Change'] = df['Close'].pct_change()
        df['Volume_Change'] = df['Volume'].pct_change()
        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close'] * 100
        df['Open_Close_Pct'] = (df['Close'] - df['Open']) / df['Open'] * 100
        
        # Target Variables for ML
        df['Next_Open'] = df['Open'].shift(-1)
        df['Next_Close'] = df['Close'].shift(-1)
        
        # Direction: 1 if next day closes higher than it opens, 0 otherwise
        df['Target_Direction'] = (df['Next_Close'] > df['Next_Open']).astype(int)
        
        # Return: next day's return
        df['Target_Return'] = (df['Next_Close'] - df['Close']) / df['Close']
        
        # Clean data
        df = df.dropna()
        print(f"   ✅ {len(df)} complete records with all indicators")
        return df
    
    def save_to_database(self, df, symbol):
        """Save comprehensive data to database"""
        conn = sqlite3.connect(self.db_path)
        
        for index, row in df.iterrows():
            conn.execute('''
                INSERT OR REPLACE INTO daily_data VALUES (
                    ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?, ?,
                    ?, ?,
                    ?, ?, ?, ?,
                    ?, ?
                )
            ''', (
                index.strftime('%Y-%m-%d'), symbol,
                row['Open'], row['High'], row['Low'], row['Close'], row['Volume'],
                row['SMA_20'], row['SMA_50'], row['SMA_200'], row['EMA_12'], row['EMA_26'],
                row['RSI'], row['MACD'], row['MACD_Signal'], row['MACD_Hist'],
                row['BB_Upper'], row['BB_Middle'], row['BB_Lower'], row['BB_Width'], row['ATR'],
                row['OBV'], row['AD_Line'],
                row['Price_Change'], row['Volume_Change'], row['High_Low_Pct'], row['Open_Close_Pct'],
                row['Target_Direction'], row['Target_Return']
            ))
        
        conn.commit()
        conn.close()
        print(f"   💾 {symbol} data saved to database")
    
    def collect_all_data(self, symbols=['SPY', 'QQQ', 'GLD']):
        """Complete data collection pipeline"""
        print("🚀 Starting complete data collection...")
        
        # Download raw data
        raw_data = self.download_stock_data(symbols)
        
        # Process each symbol
        for symbol, data in raw_data.items():
            print(f"\n📈 Processing {symbol}...")
            processed_data = self.calculate_all_indicators(data)
            self.save_to_database(processed_data, symbol)
        
        print("\n🎉 Complete data collection finished!")
        return len(raw_data)
    
    def get_latest_features(self, symbol):
        """Get latest features for prediction"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT sma_20, sma_50, ema_12, ema_26, rsi, macd, macd_signal, macd_hist,
                   bb_upper, bb_middle, bb_lower, bb_width, atr, obv, ad_line,
                   price_change, volume_change, high_low_pct, open_close_pct
            FROM daily_data 
            WHERE symbol = ? 
            ORDER BY date DESC 
            LIMIT 1
        '''
        
        result = pd.read_sql_query(query, conn, params=(symbol,))
        conn.close()
        
        if len(result) > 0:
            return result.iloc[0].values
        return None

# Test the advanced data collector
if __name__ == "__main__":
    collector = AdvancedDataCollector()
    collector.collect_all_data(['SPY', 'QQQ', 'GLD'])
