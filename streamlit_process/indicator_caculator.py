import pandas as pd

class caculator():
    def __init__(self, window, num_std_dev):
        self.window = window
        self.num_std_dev = num_std_dev

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
        data['STD'] = data['Close'].rolling(window=self.window).std()
        data['Upper'] = data['SMA'] + (data['STD'] * self.num_std_dev)
        data['Lower'] = data['SMA'] - (data['STD'] * self.num_std_dev)
        return data

