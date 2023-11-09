import pandas as pd

class caculator():
    def __init__(self, window, num_std_dev, short_period, long_period, signal_period):
        self.window = window
        self.num_std_dev = num_std_dev
        self.short_period = short_period
        self.long_period = long_period
        self.signal_period = signal_period

    def calculate_ema(self, data):
        data['EMA50'] = data['Close'].ewm(span=50, adjust=False).mean()
        data['EMA100'] = data['Close'].ewm(span=100, adjust=False).mean()
        data['EMA200'] = data['Close'].ewm(span=200, adjust=False).mean()
        return data

    # Calculate SMA and Bollinger Bands
    def calculate_sma(self,data):
        data['SMA'] = data['Close'].rolling(window=self.window).mean()
        return data

    def calculate_bollinger_bands(self,data):
        data['SMA'] = data['Close'].rolling(window=self.window).mean()
        data['Std'] = data['Close'].rolling(window=self.window).std()
        data['Upper'] = data['SMA'] + (data['Std'] * 2)
        data['Lower'] = data['SMA'] - (data['Std'] * 2)
        return data
    
    def calculate_macd(self,data):
        data['EMA12'] = data['Close'].ewm(span=self.short_period).mean()
        data['EMA26'] = data['Close'].ewm(span=self.long_period).mean()
        data['MACD'] = data['EMA12'] - data['EMA26']
        data['Signal Line'] = data['MACD'].ewm(span=self.signal_period).mean()
        return data
    








