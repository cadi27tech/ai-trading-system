 import pandas as pd
import numpy as np
import sqlite3
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class PerformanceAnalytics:
    def __init__(self, db_path='market_data.db'):
        """Initialize performance analytics system"""
        self.db_path = db_path
        
    def load_performance_data(self, symbols=['SPY', 'QQQ', 'GLD']):
        """Load performance data from database"""
        conn = sqlite3.connect(self.db_path)
        
        all_data = {}
        
        for symbol in symbols:
            query = '''
                SELECT date, strategy_return, benchmark_return, 
                       cumulative_strategy, cumulative_benchmark, 
                       win_rate, sharpe_ratio
                FROM performance_metrics 
                WHERE symbol = ?
                ORDER BY date
            '''
            
            df = pd.read_sql_query(query, conn, params=(symbol,))
            if len(df) > 0:
                df['date'] = pd.to_datetime(df['date'])
                all_data[symbol] = df.set_index('date')
        
        conn.close()
        return all_data
    
    def calculate_portfolio_metrics(self, performance_data):
        """Calculate portfolio-level metrics"""
        if not performance_data:
            return {}
        
        # Equal weight portfolio
        portfolio_returns = []
        dates = None
        
        for symbol, data in performance_data.items():
            if dates is None:
                dates = data.index
            else:
                dates = dates.intersection(data.index)
        
        if len(dates) == 0:
            return {}
        
        # Calculate equal-weighted portfolio returns
        portfolio_daily_returns = pd.Series(0, index=dates)
        
        for symbol, data in performance_data.items():
            symbol_data = data.loc[dates]
            portfolio_daily_returns += symbol_data['strategy_return'] / len(performance_data)
        
        # Portfolio metrics
        total_return = (1 + portfolio_daily_returns).prod() - 1
        volatility = portfolio_daily_returns.std() * np.sqrt(252)
        sharpe_ratio = (portfolio_daily_returns.mean() * 252 - 0.02) / volatility if volatility > 0 else 0
        
        # Max drawdown
        cumulative = (1 + portfolio_daily_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        return {
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'daily_returns': portfolio_daily_returns,
            'cumulative_returns': cumulative
        }
    
    def create_performance_dashboard(self, performance_data):
        """Create comprehensive performance dashboard"""
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Cumulative Returns Comparison',
                'Daily Returns Distribution', 
                'Rolling Sharpe Ratio',
                'Drawdown Analysis',
                'Monthly Returns Heatmap',
                'Risk-Return Scatter'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        colors = {'SPY': 'blue', 'QQQ': 'green', 'GLD': 'gold'}
        
        # 1. Cumulative Returns
        for symbol, data in performance_data.items():
            if len(data) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['cumulative_strategy'],
                        name=f'{symbol} Strategy',
                        line=dict(color=colors.get(symbol, 'black')),
                        legendgroup=symbol
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['cumulative_benchmark'],
                        name=f'{symbol} Buy&Hold',
                        line=dict(color=colors.get(symbol, 'gray'), dash='dash'),
                        legendgroup=symbol,
                        showlegend=False
                    ),
                    row=1, col=1
                )
        
        # 2. Daily Returns Distribution
        for symbol, data in performance_data.items():
            if len(data) > 0:
                fig.add_trace(
                    go.Histogram(
                        x=data['strategy_return'],
                        name=f'{symbol} Returns',
                        opacity=0.7,
                        nbinsx=50
                    ),
                    row=1, col=2
                )
        
        # 3. Rolling Sharpe Ratio (30-day)
        for symbol, data in performance_data.items():
            if len(data) > 30:
                rolling_sharpe = data['strategy_return'].rolling(30).apply(
                    lambda x: (x.mean() * 252 - 0.02) / (x.std() * np.sqrt(252)) if x.std() > 0 else 0
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=rolling_sharpe,
                        name=f'{symbol} Sharpe',
                        line=dict(color=colors.get(symbol, 'black'))
                    ),
                    row=2, col=1
                )
        
        # 4. Drawdown Analysis
        for symbol, data in performance_data.items():
            if len(data) > 0:
                cumulative = data['cumulative_strategy']
                rolling_max = cumulative.expanding().max()
                drawdown = (cumulative - rolling_max) / rolling_max
                
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=drawdown,
                        fill='tonexty',
                        name=f'{symbol} Drawdown',
                        line=dict(color=colors.get(symbol, 'red'))
                    ),
                    row=2, col=2
                )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title="Comprehensive Performance Analysis",
            showlegend=True
        )
        
        return fig
    
    def create_risk_return_analysis(self, performance_data):
        """Create risk-return analysis chart"""
        
        metrics = []
        
        for symbol, data in performance_data.items():
            if len(data) > 0:
                returns = data['strategy_return']
                total_return = (1 + returns).prod() - 1
                volatility = returns.std() * np.sqrt(252)
                sharpe = (returns.mean() * 252 - 0.02) / volatility if volatility > 0 else 0
                
                metrics.append({
                    'Symbol': symbol,
                    'Total Return': total_return,
                    'Volatility': volatility,
                    'Sharpe Ratio': sharpe
                })
        
        if not metrics:
            return None
        
        metrics_df = pd.DataFrame(metrics)
        
        # Create scatter plot
        fig = px.scatter(
            metrics_df,
            x='Volatility',
            y='Total Return',
            color='Sharpe Ratio',
            size='Sharpe Ratio',
            hover_data=['Symbol'],
            title='Risk-Return Analysis',
            labels={
                'Volatility': 'Annualized Volatility',
                'Total Return': 'Total Return',
                'Sharpe Ratio': 'Sharpe Ratio'
            }
        )
        
        # Add text annotations
        for i, row in metrics_df.iterrows():
            fig.add_annotation(
                x=row['Volatility'],
                y=row['Total Return'],
                text=row['Symbol'],
                showarrow=True,
                arrowhead=2
            )
        
        return fig
    
    def generate_performance_report(self, symbols=['SPY', 'QQQ', 'GLD']):
        """Generate comprehensive performance report"""
        
        # Load data
        performance_data = self.load_performance_data(symbols)
        
        if not performance_data:
            return None
        
        # Calculate individual metrics
        individual_metrics = {}
        
        for symbol, data in performance_data.items():
            if len(data) > 0:
                returns = data['strategy_return']
                
                metrics = {
                    'Total Return': (1 + returns).prod() - 1,
                    'Annualized Return': (1 + returns).prod() ** (252/len(returns)) - 1,
                    'Volatility': returns.std() * np.sqrt(252),
                    'Sharpe Ratio': (returns.mean() * 252 - 0.02) / (returns.std() * np.sqrt(252)),
                    'Max Drawdown': ((1 + returns).cumprod() / (1 + returns).cumprod().expanding().max() - 1).min(),
                    'Win Rate': len(returns[returns > 0]) / len(returns),
                    'Total Days': len(returns),
                    'Best Day': returns.max(),
                    'Worst Day': returns.min()
                }
                
                individual_metrics[symbol] = metrics
        
        # Portfolio metrics
        portfolio_metrics = self.calculate_portfolio_metrics(performance_data)
        
        return {
            'individual_metrics': individual_metrics,
            'portfolio_metrics': portfolio_metrics,
            'performance_data': performance_data,
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def create_monthly_returns_heatmap(self, returns_series, title="Monthly Returns"):
        """Create monthly returns heatmap"""
        
        # Resample to monthly returns
        monthly_returns = returns_series.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # Create pivot table for heatmap
        monthly_df = monthly_returns.to_frame('returns')
        monthly_df['year'] = monthly_df.index.year
        monthly_df['month'] = monthly_df.index.month
        
        pivot_table = monthly_df.pivot_table(
            values='returns', 
            index='year', 
            columns='month', 
            fill_value=0
        )
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=pivot_table.values,
            x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
            y=pivot_table.index,
            colorscale='RdYlGn',
            zmid=0,
            text=np.round(pivot_table.values * 100, 1),
            texttemplate='%{text}%',
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Month",
            yaxis_title="Year"
        )
        
        return fig

# Test performance analytics
if __name__ == "__main__":
    analytics = PerformanceAnalytics()
    report = analytics.generate_performance_report(['SPY', 'QQQ', 'GLD'])
    
    if report:
        print("📊 Performance Report Generated:")
        for symbol, metrics in report['individual_metrics'].items():
            print(f"\n{symbol}:")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    if 'Rate' in metric or 'Return' in metric:
                        print(f"  {metric}: {value:.2%}")
                    else:
                        print(f"  {metric}: {value:.3f}")
                else:
                    print(f"  {metric}: {value}")

