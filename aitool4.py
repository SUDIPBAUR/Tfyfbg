import pandas as pd
import numpy as np
import yfinance as yf
import ccxt
from datetime import datetime, timedelta
from scipy.stats import zscore
from sklearn.ensemble import RandomForestClassifier

class AlgoTrader:
    def __init__(self, capital=1000000):
        self.capital = capital
        self.positions = {}
        self.portfolio_value = []
        self.trade_log = []
        
    class RiskManager:
        def __init__(self, max_position_size=0.05, stop_loss=0.03):
            self.max_position_size = max_position_size
            self.stop_loss = stop_loss
            self.risk_free_rate = 0.02  # Assume 2% risk-free rate
            
        def calculate_position_size(self, current_price, volatility):
            """Kelly Criterion-based position sizing"""
            position_size = min(self.max_position_size * self.capital, 
                              (self.capital * volatility) / current_price)
            return int(position_size)

    class ExecutionEngine:
        def __init__(self, slippage=0.0005, commission=0.0002):
            self.slippage = slippage
            self.commission = commission
            
        def execute_order(self, symbol, quantity, price, side):
            # Simulate market execution with slippage and commissions
            executed_price = price * (1 + self.slippage) if side == 'BUY' else price * (1 - self.slippage)
            total_cost = quantity * executed_price * (1 + self.commission)
            return executed_price, total_cost

    def fetch_market_data(self, symbols, start_date, end_date, timeframe='1d'):
        """Multi-asset data fetching (stocks and crypto)"""
        data = {}
        
        # Fetch stock data
        stock_data = yf.download(list(symbols['stocks']), 
                               start=start_date, 
                               end=end_date,
                               interval=timeframe)['Adj Close']
        
        # Fetch crypto data
        exchange = ccxt.binance()
        crypto_data = {}
        for symbol in symbols['crypto']:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=exchange.parse8601(start_date))
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            crypto_data[symbol] = df.set_index('timestamp')['close']
        
        # Combine datasets
        data = pd.concat([stock_data, pd.DataFrame(crypto_data)], axis=1)
        return data.dropna()

    def create_features(self, data):
        """Feature engineering for multi-asset strategy"""
        features = pd.DataFrame(index=data.index)
        
        # Momentum features
        features['RSI'] = data.pct_change().apply(lambda x: x.rolling(14).apply(
            lambda y: 100 - (100 / (1 + (y[y > 0].mean() / abs(y[y < 0].mean())))))
        
        # Mean reversion features
        features['ZScore'] = data.apply(lambda x: zscore(x - x.rolling(20).mean()))
        
        # Volatility features
        features['ATR'] = data.rolling(14).apply(lambda x: x.max() - x.min())
        
        # Macro features
        features['VIX'] = yf.download('^VIX', start=data.index[0], end=data.index[-1])['Adj Close']
        
        return features.dropna()

    class Strategy:
        def __init__(self, model=RandomForestClassifier(n_estimators=100)):
            self.model = model
            self.threshold = 0.6  # Confidence threshold
            
        def train(self, X, y):
            self.model.fit(X, y)
            
        def predict(self, X_current):
            proba = self.model.predict_proba(X_current.reshape(1, -1))[0][1]
            return 'BUY' if proba > self.threshold else 'SELL' if proba < (1 - self.threshold) else 'HOLD'

    def backtest(self, symbols, start_date, end_date):
        # Initialize components
        risk_manager = self.RiskManager()
        execution = self.ExecutionEngine()
        strategy = self.Strategy()
        
        # Get data
        data = self.fetch_market_data(symbols, start_date, end_date)
        features = self.create_features(data)
        returns = data.pct_change().shift(-1)  # Next period returns
        
        # Train ML model
        strategy.train(features.iloc[:-100], (returns.iloc[:-100] > 0).astype(int))
        
        # Backtesting loop
        for i in range(len(features)-100, len(features)):
            current_prices = data.iloc[i]
            current_features = features.iloc[i]
            
            # Get predictions
            signal = strategy.predict(current_features.values)
            
            # Execute trades
            if signal != 'HOLD':
                for symbol in symbols:
                    position_size = risk_manager.calculate_position_size(
                        current_prices[symbol], 
                        features['ATR'][symbol].iloc[i]
                    )
                    
                    executed_price, cost = execution.execute_order(
                        symbol,
                        position_size,
                        current_prices[symbol],
                        signal
                    )
                    
                    # Update positions and capital
                    if signal == 'BUY':
                        self.capital -= cost
                        self.positions[symbol] = position_size
                    else:
                        self.capital += cost
                        del self.positions[symbol]
                        
                    # Log trade
                    self.trade_log.append({
                        'timestamp': data.index[i],
                        'symbol': symbol,
                        'action': signal,
                        'quantity': position_size,
                        'price': executed_price
                    })
            
            # Update portfolio value
            self.portfolio_value.append(self.capital + sum(
                data[symbol].iloc[i] * qty for symbol, qty in self.positions.items()
            ))
            
        return pd.Series(self.portfolio_value, index=data.index[-100:])

    def analyze_performance(self):
        returns = pd.Series(self.portfolio_value).pct_change().dropna()
        
        print(f"Total Return: {returns.sum()*100:.2f}%")
        print(f"Sharpe Ratio: {(returns.mean() - 0.02)/returns.std()*np.sqrt(252):.2f}")
        print(f"Max Drawdown: {(returns.cumsum().expanding().max() - returns.cumsum()).max()*100:.2f}%")
        
        # Plot performance
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12,6))
        plt.plot(self.portfolio_value)
        plt.title('Portfolio Performance')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.show()

# Example Usage
if __name__ == "__main__":
    trader = AlgoTrader(capital=1000000)
    
    symbols = {
        'stocks': ['SPY', 'QQQ', 'IWM'],
        'crypto': ['BTC/USDT', 'ETH/USDT']
    }
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*3)
    
    # Run backtest
    performance = trader.backtest(symbols, start_date, end_date)
    
    # Analyze results
    trader.analyze_performance()