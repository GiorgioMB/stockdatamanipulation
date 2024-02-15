#%%
"""
This module contains the Transform class which is designed to do multiple manipulation with financial data and statements
"""
from requests.exceptions import RequestException
import yfinance as yf
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import numpy as np
import requests
import pandas_datareader.data as web
from datafetcher import Fetcher
class Transform:
    def __init__(self, ticker:str = None, dataframe:pd.DataFrame = None):
        if ticker:
            self.fetch = Fetcher(ticker)
            self.df, self.yf_stock = self.fetch.df, self.fetch.yf_stock
            self.income_statement = self.fetch.income_statement 
            self.balance_sheet = self.fetch.balance_sheet
            self.cashflow = self.fetch.cashflow
        elif dataframe:
            self.df = dataframe
    
    def exponential_weights(self, span:int, df:pd.DataFrame = None) -> pd.DataFrame:
        if df is None:
            df = self.df
        weights = np.exp(-np.log(2)*np.arange(len(df))/span)[::-1]
        weighted_close = df['Close'] * weights / weights.max()
        df['Close'] = weighted_close
        return df
    
    def adjust_inflation(self, df:pd.DataFrame = None) -> pd.DataFrame:
        if df is None:
            df = self.df.copy()
        df.index = df.index.tz_localize(None)
        first_date = df.index.min()
        last_date = df.index.max()
        cpi_series_id = 'CPIAUCSL'
        cpi_data = web.DataReader(cpi_series_id, 'fred', first_date, last_date)
        cpi_data.rename(columns = {'CPIAUCSL':'CPI'}, inplace = True)
        base_cpi = cpi_data['CPI'].iloc[-1] 
        cpi_all_data = cpi_data.reindex(df.index, method = 'ffill')
        cpi_all_data = cpi_all_data.fillna(method = 'bfill')
        refactor = base_cpi / cpi_all_data['CPI']
        df['Refactored'] = df['Close'] * refactor
        return df

    def calculate_returns(self, df:pd.DataFrame = None) -> pd.DataFrame:
        if df is None:
            df = self.df
        df['Return'] = df['Close'].pct_change()
        return df
    
    def calculate_RSI(self, df:pd.DataFrame = None, period:int = 14) -> pd.DataFrame:
        if df is None:
            df = self.df
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        RS = gain / loss
        RSI = 100 - (100 / (1 + RS))
        df['RSI'] = RSI
        return df
    
    def calculate_MACD(self, df:pd.DataFrame = None, short_window:int = 12, long_window:int = 26) -> pd.DataFrame:
        if df is None:
            df = self.df
        short_ema = df['Close'].ewm(span = short_window, adjust = False).mean()
        long_ema = df['Close'].ewm(span = long_window, adjust = False).mean()
        df['MACD'] = short_ema - long_ema
        df['Signal'] = df['MACD'].ewm(span = 9, adjust = False).mean()
        return df
    
    def calculate_bollinger_bands(self, df:pd.DataFrame = None, window:int = 20) -> pd.DataFrame:
        if df is None:
            df = self.df
        rolling_mean = df['Close'].rolling(window).mean()
        rolling_std = df['Close'].rolling(window).std()
        df['Upper'] = rolling_mean + (rolling_std * 2)
        df['Lower'] = rolling_mean - (rolling_std * 2)
        return df

    def calculate_EMA(self, df:pd.DataFrame = None, window:int = 20) -> pd.DataFrame:
        if df is None:
            df = self.df
        df['EMA'] = df['Close'].ewm(span = window, adjust = False).mean()
        return df
    
    def calculate_SMA(self, df:pd.DataFrame = None, window:int = 20) -> pd.DataFrame:
        if df is None:
            df = self.df
        df['SMA'] = df['Close'].rolling(window).mean()
        return df
    
    def calculate_KAMA(self, df:pd.DataFrame = None, window:int = 20, fast:int=2, slow: int =30) -> pd.DataFrame:
    # Step 0: Ensure the DataFrame contains a 'Close' column
        if df is None:
            df = self.df
        # Step 1: Calculate the Efficiency Ratio (ER)
        change = df['Close'].diff(window).abs()
        volatility = df['Close'].diff().abs().rolling(window=window).sum()
        ER = change / volatility

        # Calculate the Smoothing Constant (SC)
        fastSC = 2 / (fast + 1)
        slowSC = 2 / (slow + 1)
        SC = (ER * (fastSC - slowSC) + slowSC) ** 2

        # Initialize KAMA with first value being the same as the first close price
        KAMA = pd.Series(index=df.index, data=np.NaN)
        KAMA.iloc[window] = df['Close'].iloc[window]

        # Calculate KAMA
        for i in range(window + 1, len(df)):
            KAMA.iloc[i] = KAMA.iloc[i - 1] + SC.iloc[i] * (df['Close'].iloc[i] - KAMA.iloc[i - 1])

        df['KAMA'] = KAMA
        return df
    
    def calculate_on_balance_volume(self, df:pd.DataFrame = None) -> pd.DataFrame:
        if df is None:
            df = self.df
        df['OBV'] = np.where(df['Close'] > df['Close'].shift(1), df['Volume'], np.where(df['Close'] < df['Close'].shift(1), -df['Volume'], 0)).cumsum()
        return df

        
    def calculate_ATR(self, df:pd.DataFrame = None, window:int = 14) -> pd.DataFrame:
        if df is None:
            df = self.df
        df['H-L'] = abs(df['High'] - df['Low'])
        df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
        df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
        df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis = 1, skipna = False)
        df['ATR'] = df['TR'].rolling(window).mean()
        return df
    
    def calculate_ADI(self, df: pd.DataFrame = None) -> pd.DataFrame:
        if df is None:
            df = self.df
        clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) 
        range_span = (df['High'] - df['Low'])
        df['MFM'] = np.where(range_span == 0, 0, clv / range_span)  
        df['MFMV'] = df['MFM'] * df['Volume']
        df['ADI'] = df['MFMV'].cumsum()
        
        return df


    def calculate_parabolic_sar(self, df, acceleration=0.02, maximum=0.2) -> pd.DataFrame:
        if df is None:
            df = self.df
        # Initialize columns for SAR, EP (Extreme Point), and AF (Acceleration Factor)
        df['SAR'] = 0.0
        df['EP'] = 0.0
        df['AF'] = acceleration

        # Determine the initial trend
        rising_trend = df['Close'].iloc[1] > df['Close'].iloc[0]

        # Initialize SAR, EP, and the first trend
        if rising_trend:
            df['SAR'].iloc[0] = df['Low'].iloc[0]
            df['EP'].iloc[0] = df['High'].iloc[0]
        else:
            df['SAR'].iloc[0] = df['High'].iloc[0]
            df['EP'].iloc[0] = df['Low'].iloc[0]

        for i in range(1, len(df)):
            # Adjust AF
            if (rising_trend and df['High'].iloc[i] > df['EP'].iloc[i-1]) or (not rising_trend and df['Low'].iloc[i] < df['EP'].iloc[i-1]):
                df['AF'].iloc[i] = min(df['AF'].iloc[i-1] + acceleration, maximum)
            else:
                df['AF'].iloc[i] = df['AF'].iloc[i-1]

            # Calculate SAR
            df['SAR'].iloc[i] = df['SAR'].iloc[i-1] + df['AF'].iloc[i-1] * (df['EP'].iloc[i-1] - df['SAR'].iloc[i-1])

            # Check for trend reversal
            if rising_trend:
                if df['Low'].iloc[i] < df['SAR'].iloc[i]:
                    rising_trend = False
                    df['SAR'].iloc[i] = df['EP'].iloc[i-1]
                    df['EP'].iloc[i] = df['Low'].iloc[i]
                    df['AF'].iloc[i] = acceleration
                else:
                    if df['High'].iloc[i] > df['EP'].iloc[i-1]:
                        df['EP'].iloc[i] = df['High'].iloc[i]
            else:
                if df['High'].iloc[i] > df['SAR'].iloc[i]:
                    rising_trend = True
                    df['SAR'].iloc[i] = df['EP'].iloc[i-1]
                    df['EP'].iloc[i] = df['High'].iloc[i]
                    df['AF'].iloc[i] = acceleration
                else:
                    if df['Low'].iloc[i] < df['EP'].iloc[i-1]:
                        df['EP'].iloc[i] = df['Low'].iloc[i]

            if rising_trend:
                df['SAR'].iloc[i] = min(df['SAR'].iloc[i], df['Low'].iloc[i-1], df['Low'].iloc[i-2] if i >= 2 else df['SAR'].iloc[i])
            else:
                df['SAR'].iloc[i] = max(df['SAR'].iloc[i], df['High'].iloc[i-1], df['High'].iloc[i-2] if i >= 2 else df['SAR'].iloc[i])

        return df
    
    def calculate_stochastic_oscillator(self, df:pd.DataFrame = None, window:int = 14) -> pd.DataFrame:
        if df is None:
            df = self.df
        df['L14'] = df['Low'].rolling(window).min()
        df['H14'] = df['High'].rolling(window).max()
        df['%K'] = 100 * ((df['Close'] - df['L14']) / (df['H14'] - df['L14']))
        df['%D'] = df['%K'].rolling(window).mean()
        return df
    
    def calculate_momentum(self, df:pd.DataFrame = None, window:int = 10) -> pd.DataFrame:
        if df is None:
            df = self.df
        df['Momentum'] = df['Close'] - df['Close'].shift(window)
        return df
    
    def calculate_williams_R(self, df:pd.DataFrame = None, window:int = 14) -> pd.DataFrame:
        if df is None:
            df = self.df
        df['H14'] = df['High'].rolling(window).max()
        df['L14'] = df['Low'].rolling(window).min()
        df['Williams %R'] = -100 * ((df['H14'] - df['Close']) / (df['H14'] - df['L14']))
        return df
    
    def calculate_fibonacci_retracement(self, df:pd.DataFrame = None, low:int = 0, high:int = 0) -> pd.DataFrame:
        if df is None:
            df = self.df
        df['Low'] = low
        df['High'] = high
        df['Retracement'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        return df
    
    def calculate_aroon_oscillator(self, df:pd.DataFrame = None, window:int = 25) -> pd.DataFrame:
        if df is None:
            df = self.df
        df['Up'] = df['High'].rolling(window).apply(lambda x: x.argmax(), raw = True) / window * 100
        df['Down'] = df['Low'].rolling(window).apply(lambda x: x.argmin(), raw = True) / window * 100
        df['Aroon Oscillator'] = df['Up'] - df['Down']
        return df
    
    def calculate_ADX(self, df:pd.DataFrame = None, window:int = 14) -> pd.DataFrame:
        if df is None:
            df = self.df
        df['H-L'] = abs(df['High'] - df['Low'])
        df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
        df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
        df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis = 1, skipna = False)
        df['DMplus'] = np.where((df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']), df['High'] - df['High'].shift(1), 0)
        df['DMminus'] = np.where((df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)), df['Low'].shift(1) - df['Low'], 0)
        TRn = df['TR'].rolling(window).mean()
        DMplusN = df['DMplus'].rolling(window).mean()
        DMminusN = df['DMminus'].rolling(window).mean()
        DIplus = 100 * (DMplusN / TRn)
        DIminus = 100 * (DMminusN / TRn)
        df['DX'] = 100 * abs((DIplus - DIminus) / (DIplus + DIminus))
        ADX = df['DX'].rolling(window).mean()
        df['ADX'] = ADX
        return df

    def calculate_VWAP(self, df:pd.DataFrame = None) -> pd.DataFrame:
        if df is None:
            df = self.df
        df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['Traded'] = df['TP'] * df['Volume']
        df['Cumulative Traded'] = df['Traded'].cumsum()
        df['Cumulative Volume'] = df['Volume'].cumsum()
        df['VWAP'] = df['Cumulative Traded'] / df['Cumulative Volume']
        return df        
    
    def plot_fibonacci_retracement(self, df:pd.DataFrame = None, low:int = 0, high:int = 0):
        if df is None:
            df = self.df
        df = self.calculate_fibonacci_retracement(df, low, high)
        plt.figure(figsize=(12,6))
        plt.plot(df['Retracement'], label = 'Retracement')
        plt.title('Fibonacci Retracement')
        plt.legend()
        plt.show()
    
    def plot_stochastic_oscillator(self, df:pd.DataFrame = None, window:int = 14):
        if df is None:
            df = self.df
        df = self.calculate_stochastic_oscillator(df, window)
        plt.figure(figsize=(12,6))
        plt.plot(df['%D'], label = '%D')
        plt.title('Stochastic Oscillator')
        plt.legend()
        plt.show()

    def plot_momentum(self, df:pd.DataFrame = None, window:int = 10):
        if df is None:
            df = self.df
        df = self.calculate_momentum(df, window)
        plt.figure(figsize=(12,6))
        plt.plot(df['Momentum'], label = 'Momentum')
        plt.title('Momentum')
        plt.legend()
        plt.show()

    def plot_williams_R(self, df:pd.DataFrame = None, window:int = 14):
        if df is None:
            df = self.df
        df = self.calculate_williams_R(df, window)
        plt.figure(figsize=(12,6))
        plt.plot(df['Williams %R'], label = 'Williams %R')
        plt.title('Williams %R')
        plt.legend()
        plt.show()

    def plot_parabolic_sar(self, df:pd.DataFrame = None, acceleration = 0.02, maximum = 0.2):
        if df is None:
            df = self.df
        df = self.calculate_parabolic_sar(df, acceleration, maximum)
        plt.figure(figsize=(12,6))
        plt.plot(df['SAR'], label = 'SAR')
        plt.title('Parabolic SAR')
        plt.legend()
        plt.show()     

    def plot_inflation_adjustment(self, df:pd.DataFrame = None):
        if df is None:
            df = self.df
        df = self.adjust_inflation(df)
        plt.figure(figsize=(12,6))
        plt.plot(df['Close'], label = 'Inflation Adjusted Close')
        plt.title('Inflation Adjustment')
        plt.legend()
        plt.show()   
    
    def plot_ADI(self, df:pd.DataFrame = None):
        if df is None:
            df = self.df
        df = self.calculate_ADI(df)
        plt.figure(figsize=(12,6))
        plt.plot(df['ADI'], label = 'ADI')
        plt.title('ADI')
        plt.legend()
        plt.show()

    def plot_EMA(self, df:pd.DataFrame = None, window:int = 20):
        if df is None:
            df = self.df
        df = self.calculate_EMA(df, window)
        plt.figure(figsize=(12,6))
        plt.plot(df['EMA'], label = 'EMA')
        plt.title('EMA')
        plt.legend()
        plt.show()
    
    def plot_SMA(self, df:pd.DataFrame = None, window:int = 20):
        if df is None:
            df = self.df
        df = self.calculate_SMA(df, window)
        plt.figure(figsize=(12,6))
        plt.plot(df['SMA'], label = 'SMA')
        plt.title('SMA')
        plt.legend()
        plt.show()
    
    def plot_ATR(self, df:pd.DataFrame = None, window:int = 14):
        if df is None:
            df = self.df
        df = self.calculate_ATR(df, window)
        plt.figure(figsize=(12,6))
        plt.plot(df['ATR'], label = 'ATR')
        plt.title('ATR')
        plt.legend()
        plt.show()
    
    def plot_KAMA(self, df:pd.DataFrame = None, window:int = 20):
        if df is None:
            df = self.df
        df = self.calculate_KAMA(df, window)
        plt.figure(figsize=(12,6))
        plt.plot(df['KAMA'], label = 'KAMA')
        plt.title('KAMA')
        plt.legend()
        plt.show()

    def plot_bollinger_bands(self, df:pd.DataFrame = None, window:int = 20):
        if df is None:
            df = self.df
        df = self.calculate_bollinger_bands(df, window)
        plt.figure(figsize=(12,6))
        plt.plot(df['Close'], label = 'Close')
        plt.plot(df['Upper'], label = 'Upper')
        plt.plot(df['Lower'], label = 'Lower')
        plt.title('Bollinger Bands')
        plt.legend()
        plt.show()

    def plot_RSI(self, df:pd.DataFrame = None, period:int = 14):
        if df is None:
            df = self.df
        df = self.calculate_RSI(df, period)
        plt.figure(figsize=(12,6))
        plt.plot(df['RSI'], label = 'RSI')
        plt.title('RSI')
        plt.axhline(0, linestyle='--', alpha=0.1)
        plt.axhline(20, linestyle='--', alpha=0.5)
        plt.axhline(30, linestyle='--')
        plt.axhline(70, linestyle='--')
        plt.axhline(80, linestyle='--', alpha=0.5)
        plt.axhline(100, linestyle='--', alpha=0.1)
        plt.show()
    
    def plot_MACD(self, df:pd.DataFrame = None, short_window:int = 12, long_window:int = 26):
        if df is None:
            df = self.df
        df = self.calculate_MACD(df, short_window, long_window)
        plt.figure(figsize=(12,6))
        plt.plot(df['MACD'], label = 'MACD')
        plt.title('MACD')
        plt.legend()
        plt.show()
    
    def plot_returns(self, df:pd.DataFrame = None):
        if df is None:
            df = self.df
        df = self.calculate_returns(df)
        plt.figure(figsize=(12,6))
        plt.plot(df['Return'], label = 'Return')
        plt.title('Returns')
        plt.show()

a = Transform('AAPL').plot_williams_R()
# %%
