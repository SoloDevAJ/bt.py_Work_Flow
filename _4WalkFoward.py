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
from talib import CDLENGULFING
import pickle
import os

class SimpleEngulfingStrategy(Strategy):
    SL = 50
    TP_R = 4
    days_held = 1
    MAX_HOLD_TIME = pd.Timedelta(days=days_held)

    def init(self):
        self.engulfing = self.I(CDLENGULFING, self.data.Open, self.data.High, self.data.Low, self.data.Close)
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None
        self.entry_time = None

    def next(self):
        engulf = self.engulfing[-1]
        price = self.data.Close[-1]
        current_time = self.data.index[-1]

        if current_time.weekday() in [0, 3]:
            return

        if not self.position:
            if engulf == 100:
                self.buy()
                self.entry_price = price
                self.stop_loss = price - self.SL
                self.take_profit = price + self.SL * self.TP_R
                self.entry_time = current_time

            elif engulf == -100:
                self.sell()
                self.entry_price = price
                self.stop_loss = price + self.SL
                self.take_profit = price - self.SL * self.TP_R
                self.entry_time = current_time

        if self.position:
            if self.position.is_long:
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

def load_and_prepare_data(file_path, start_date, end_date):
    data = pd.read_csv(file_path)

    print("Columns in the CSV file:", data.columns)

    if 'Time' in data.columns:
        if data['Time'].dtype in ['int64', 'float64'] and data['Time'].max() > 1e12:
            print("Detected UNIX millisecond timestamps. Converting...")
            data['Time'] = pd.to_datetime(data['Time'], unit='ms')
        else:
            print("Assuming standard datetime format. Parsing...")
            data['Time'] = pd.to_datetime(data['Time'])
    else:
        raise ValueError("No 'Time' column found in CSV.")

    data.set_index('Time', inplace=True)
    data.index = pd.to_datetime(data.index)
    data.sort_index(inplace=True)

    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    data = data[required_columns].copy()

    if data.index.duplicated().any():
        print("Duplicate indexes found. Removing duplicates.")
        data = data[~data.index.duplicated(keep='first')]

    print("Checking for NaN values in the data:")
    print(data.isna().sum())
    data = data.dropna()
    data = data.loc[start_date:end_date]

    print(f"Number of data points after reduction: {len(data)}")
    print("Datetime index type:", data.index.dtype)
    print("Data index is monotonic:", data.index.is_monotonic_increasing)

    return data

def walk_forward(
    # Default Values
    strategy,
    data_full,
    warmup_bars,
    lookback_bars=31 * 96,  # 28 days * 96 bars per day
    validation_bars=14 * 96,  # 7 days * 96 bars per day
    cash=10_000_000,
    commission=0.002,
):
    stats_master = []

    for i in range(lookback_bars, len(data_full) - validation_bars, validation_bars):
        print(f"Running walk-forward iteration at index: {i}")

        sample_data = data_full.iloc[i - lookback_bars : i]

        bt_training = Backtest(sample_data, strategy, cash=cash, commission=commission)
        stats_training = bt_training.optimize(
            SL=range(25, 500, 25),
            TP_R=range(1, 10, 1),
            days_held=range(1, 5, 1),
            maximize="Equity Final [$]",
        )

        SL = stats_training._strategy.SL
        TP_R = stats_training._strategy.TP_R
        days_held = stats_training._strategy.days_held

        validation_data = data_full.iloc[i - warmup_bars : i + validation_bars]
        bt_validation = Backtest(validation_data, strategy, cash=cash, commission=commission)
        stats_validation = bt_validation.run(
            SL=SL,
            TP_R=TP_R,
            days_held=days_held,
        )

        # Print the stats for this validation window
        print(f"Validation stats for window ending at index {i}:")
        print(stats_validation)

        stats_master.append(stats_validation)

    return stats_master

if __name__ == '__main__':
    file_path = data_paths["NAS100"]["M15"]
    
    # Note: To avoid assertion error & Plot equity curve (Need at least 6months) 
    start_date = "2024-01-01"
    end_date = "2024-07-01"
    data = load_and_prepare_data(file_path, start_date, end_date)

    # Walk-forward validation
    print("\nStarting walk-forward optimization...\n")
    results = walk_forward(
        # Overridden Values
        strategy=SimpleEngulfingStrategy,
        data_full=data,
        warmup_bars=500,
        lookback_bars=31 * 96,  # shorter lookback for faster test
        validation_bars=14 * 96,
        cash=1_000_000,
    )

    print(f"Completed {len(results)} walk-forward validation windows.")

    # Define global constants for lookback and validation bars
    LOOKBACK_BARS = 31 * 96  # 31 days * 96 bars per day
    VALIDATION_BARS = 14 * 96  # 14 days * 96 bars per day
    WARMUP_BARS = 500  # Warmup bars remain the same

    # RUN WALK FORWARD

    if os.path.exists("stats.pickle"):
        with open("stats.pickle", "rb") as f:
            stats = pickle.load(f)
    else:
        stats = walk_forward(SimpleEngulfingStrategy, data, warmup_bars=WARMUP_BARS)
        with open("stats.pickle", "wb") as f:
            pickle.dump(stats, f)

    def plot_stats(data, stats):
        equity_curve = stats._equity_curve
        aligned_data = data.reindex(equity_curve.index).dropna()  # Drop rows with NaN values

        # Ensure indices match exactly
        if not aligned_data.index.equals(equity_curve.index):
            print("Warning: Indices do not match. Aligning indices...")
            # Align indices by keeping only the common timestamps
            common_index = aligned_data.index.intersection(equity_curve.index)
            aligned_data = aligned_data.loc[common_index]
            equity_curve = equity_curve.loc[common_index]

        missing_timestamps = equity_curve.index.difference(data.index)
        print("Missing Timestamps:", missing_timestamps)

        bt = Backtest(aligned_data, SimpleEngulfingStrategy, cash=10_000_000, commission=0.002)
        print(stats)
        bt.plot(results=stats)

    for i, stat in enumerate(stats):
        print(f"Plotting statistics for window {i}...")
        plot_stats(data, stat)

    def plot_full_equity_curve(data, stats_list, overlay_price=True):
        equity_curves = [x["_equity_curve"].iloc[WARMUP_BARS:] for x in stats_list]

        combined = pd.Series()
        for curve in equity_curves:
            # Need to normalize each equity curve to connect them up
            if len(combined) == 0:
                combined = curve["Equity"] / 1e7
            else:
                combined = pd.concat([combined, (curve["Equity"] / 1e7) * combined.iloc[-1]])

        last_date = combined.index[-1]
        aligned_price_data = data[data.index <= last_date].iloc[LOOKBACK_BARS:]

        plt.style.use('fivethirtyeight')
        fig, ax1 = plt.subplots()
        # Get rid of grid on graph
        ax1.grid(False)
        equity_line, = ax1.plot(combined.index, combined, color="orange", label="Equity")

        if overlay_price:
            ax2 = ax1.twinx()
            ax2.grid(False)
            price_line, = ax2.plot(aligned_price_data.index, aligned_price_data.Close, label=file_path)
            ax1.legend(handles=[equity_line, price_line])
        else:
            ax1.legend(handles=[equity_line])

        plt.show()

    plot_full_equity_curve(data, stats, overlay_price=True)

    def plot_split_graph(data, anchor=False):
        """
        Plot the flow diagram of the training vs test data
        """
        fig, ax = plt.subplots()
        fig.set_figwidth(12)

        ranges = list(range(LOOKBACK_BARS, len(data) - VALIDATION_BARS, VALIDATION_BARS))

        for i in range(len(ranges)):
            # To do anchored walk-forward, just set the first slice here to 0
            if anchor:
                sample_data = data.iloc[0: ranges[i]]
            else:
                sample_data = data.iloc[ranges[i] - LOOKBACK_BARS: ranges[i]]

            validation_data = data.iloc[ranges[i]:ranges[i] + VALIDATION_BARS]

            plt.fill_between(sample_data.index,
                             [len(ranges) - i - 0.5] * len(sample_data.index),
                             [len(ranges) - i + 0.5] * len(sample_data.index),
                             color="blue")
            plt.fill_between(validation_data.index,
                             [len(ranges) - i - 0.5] * len(validation_data.index),
                             [len(ranges) - i + 0.5] * len(validation_data.index),
                             color="orange")

        plt.show()

    plot_split_graph(data, anchor=False)