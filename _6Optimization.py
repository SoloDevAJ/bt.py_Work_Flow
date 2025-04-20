from backtesting import Backtest, Strategy
import pandas as pd
from backtesting.lib import crossover, plot_heatmaps, resample_apply
import seaborn as sns
import matplotlib.pyplot as plt
import mpld3
import numpy as np
import time
from talib import CDLENGULFING, ADX, CCI, ATR
from DataPaths import data_paths
import plotly.express as px


# Strategy: Simple Engulfing Pattern with Risk Management
from talib import CDLENGULFING

class SimpleEngulfingStrategy(Strategy):
    """
    Strategy: Simple Engulfing Pattern
    - Buy on every bullish engulfing pattern.
    - Sell on every bearish engulfing pattern.
    - Only one trade is allowed at a time.
    - Maximum trade hold time is 1 day.
    - Risk management: Fixed stop-loss and take-profit.
    - Excludes trades on Mondays and Fridays.
    """

    # Class-level variables
    SL = 50  # Fixed stop-loss in pips/handles
    TP_R = 4  # Risk-to-reward multiplier for take-profit
    days_held = 1
    
    MAX_HOLD_TIME = pd.Timedelta(days=days_held)  # Maximum trade hold time

    def init(self):
        # Indicators
        self.engulfing = self.I(CDLENGULFING, self.data.Open, self.data.High, self.data.Low, self.data.Close)

        # Instance-level variables
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None
        self.entry_time = None  # Timestamp when the trade was opened

    def next(self):
        engulf = self.engulfing[-1]
        price = self.data.Close[-1]
        current_time = self.data.index[-1]  # Current timestamp

        # Skip trades on Mondays (0) and Fridays (4)
        if current_time.weekday() in [0, 3]:
            return

        # If no position is open, look for entry signals
        if not self.position:
            # Long Entry: Bullish engulfing pattern
            if engulf == 100:
                self.buy()
                self.entry_price = price
                self.stop_loss = price - self.SL
                self.take_profit = price + self.SL * self.TP_R
                self.entry_time = current_time  # Record the entry time

            # Short Entry: Bearish engulfing pattern
            elif engulf == -100:
                self.sell()
                self.entry_price = price
                self.stop_loss = price + self.SL
                self.take_profit = price - self.SL * self.TP_R
                self.entry_time = current_time  # Record the entry time

        # If a position is open, check exit conditions
        if self.position:
            if self.position.is_long:
                # Exit if price hits stop-loss, take-profit, or exceeds max hold time
                if (
                    price <= self.stop_loss
                    or price >= self.take_profit
                    or (current_time - self.entry_time) > self.MAX_HOLD_TIME
                ):
                    self.position.close()
                    self.entry_price = None
                    self.stop_loss = None
                    self.take_profit = None
                    self.entry_time = None

            elif self.position.is_short:
                # Exit if price hits stop-loss, take-profit, or exceeds max hold time
                if (
                    price >= self.stop_loss
                    or price <= self.take_profit
                    or (current_time - self.entry_time) > self.MAX_HOLD_TIME
                ):
                    self.position.close()
                    self.entry_price = None
                    self.stop_loss = None
                    self.take_profit = None
                    self.entry_time = None

# Ensure the data is loaded and prepared correctly
def load_and_prepare_data(file_path, start_date, end_date):
    # Load CSV
    data = pd.read_csv(file_path, parse_dates=['Time'], index_col='Time')
    # Ensure index is datetime
    data.index = pd.to_datetime(data.index)

    # Print column names to verify
    print("Columns in the CSV file:", data.columns)

    # Select required columns
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

    # Remove duplicate indexes
    if data.index.duplicated().any():
        print("Duplicate indexes found. Removing duplicates.")
        data = data[~data.index.duplicated(keep='first')]

    # Check for NaN values
    print("Checking for NaN values in the data:")
    print(data.isna().sum())

    # Drop NaN values
    data = data.dropna()

    # Reduce the number of data points to a specific date range
    data = data.loc[start_date:end_date]

    print(f"Number of data points after reduction: {len(data)}")

    return data


if __name__ == '__main__':

    file_path = data_paths["NAS100"]["H1"]
    
    start_date = '2020-01-01'
    end_date = '2021-01-01'

    data = load_and_prepare_data(file_path, start_date, end_date)

    # Stopwatch: Start measuring time
    start_time = time.time()
    
    print("loading backtest results.....")

    bt = Backtest(data, SimpleEngulfingStrategy, cash=1_000_000)

    heatmap = None  # Initialize heatmap variable

    # '''def custom_maximize(stats):
    # if stats['# Trades'] < 80:
    #     return -1
    
    # #     if stats['Max. Trade Duration'] > 3: # NO SWING TRADES TRADES ALLOWED OVER 3 DAYS 
    # #         return -1
    # #     # # Convert Timedelta to number of days
    # #     # drawdown_duration_days = stats['Max. Drawdown Duration'].days
    # #     # if drawdown_duration_days > 30:
    # #         return -1

    # #     # Combine the metrics you want to maximize
    #     return stats['Return [%]']  # * stats['Win Rate [%]']'''

    def custom_maximize(stats):
        # if stats['# Trades'] < 70 :
        #     return -1  # Reject strategies with too few trades

        # Ensure max trade duration is within limit
        # max_trade_duration = stats['Max. Trade Duration']
        # if isinstance(max_trade_duration, pd.Timedelta):  
        #     max_trade_duration = max_trade_duration.days  

        # if max_trade_duration > 3:  # No swing trades over x days
        #     return -1  

        # Convert avg trade duration to minutes for better control
        # avg_trade_duration = stats['Avg. Trade Duration']
        # if isinstance(avg_trade_duration, pd.Timedelta):  
        #     avg_trade_duration = avg_trade_duration.total_seconds() / 60  # Convert to minutes  

        # Compute optimization score: prioritize equity while minimizing avg trade duration
        return stats['Equity Final [$]'] #- (avg_trade_duration / 100)  # Small penalty for longer trades
    
    try:
        stats, heatmap = bt.optimize(
            SL = range(25, 500, 25),  # Fixed stop-loss in pips/handles
            TP_R = range(1, 10, 1),  # Risk-to-reward multiplier for take-profit
            days_held = range(1, 5, 1),

            maximize=custom_maximize,          # Custom optimization function
            # max_tries=10_000,                     # Maximum number of optimization attempts
            return_heatmap=True                # Return heatmap for visualization
        )

        # Stopwatch: End measuring time
        end_time = time.time()
        elapsed_time = end_time - start_time
        print('-' * 20)
        print(file_path)
        print(f"\tBacktest completed in {elapsed_time / 60:.2f} MIN")

        # Print optimized results
        print("Optimized Stats:\n", stats)

        # Plot optimized results
        bt.plot()

        # Plot heatmap
        if heatmap is not None:
            fig, ax = plt.subplots()
            plot_heatmaps(heatmap, agg="mean")
            heatmap_html = mpld3.fig_to_html(fig)
            print(heatmap)

    except Exception as e:
        print(f"An error occurred during optimization: {e}")
