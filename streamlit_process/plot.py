import streamlit as st
import mplfinance as mpf
from . import indicator_caculator as ic

class draw():
    def __init__(self, stock_symbol, window, num_std_dev, short_period, long_period, signal_period):
        self.stock_symbol = stock_symbol
        self.window = int(window)
        self.num_std_dev = int(num_std_dev)
        self.short_period = short_period
        self.long_period = long_period
        self.signal_period = signal_period
        self.ic = ic.caculator(self.window, self.num_std_dev, self.short_period,self.long_period, self.signal_period)

    def __call__(self, data):
        # Select the strategy
        selected_strategy = st.selectbox('Select a strategy', ['Candlestick', 'EMA', 'SMA', 'Bollinger Bands','MACD'])
        # Display the chart based on the selected strategy
        if selected_strategy == 'Candlestick':
            fig, axes = mpf.plot(data, type='candle', style='yahoo', title=f'Stock Chart for {self.stock_symbol}',
                                ylabel='Stock Price', returnfig=True)
            st.pyplot(fig)
        elif selected_strategy == 'EMA':
            data = self.ic.calculate_ema(data)
            fig, axes = mpf.plot(data, type='candle', style='yahoo', title=f'Stock Chart with EMA for {self.stock_symbol}',
                                ylabel='Stock Price', addplot=[mpf.make_addplot(data['EMA50'], color='blue'),
                                                            mpf.make_addplot(data['EMA100'], color='green'),
                                                            mpf.make_addplot(data['EMA200'], color='red')],
                                returnfig=True)
            st.pyplot(fig)
        elif selected_strategy == 'SMA':
            data = self.ic.calculate_sma(data)
            fig, axes = mpf.plot(data, type='candle', style='yahoo', title=f'Stock Chart with SMA for {self.stock_symbol}',
                                ylabel='Stock Price', ylabel_lower='SMA', mav=(self.window,),
                                hlines=dict(hlines=[data['SMA'].mean()], colors='red', linestyle='dashed'),
                                returnfig=True)
            st.pyplot(fig)
        elif selected_strategy == 'Bollinger Bands':
            data = self.ic.calculate_bollinger_bands(data)
            fig, axes = mpf.plot(data, type='candle', style='yahoo', title=f'Stock Chart with Bollinger Bands for {self.stock_symbol}',
                                ylabel='Stock Price', ylabel_lower='Bollinger Bands', mav=(self.window,),
                                addplot=[mpf.make_addplot(data['Upper'], color='purple', secondary_y=False),
                                        mpf.make_addplot(data['Lower'], color='purple', secondary_y=False)],
                                returnfig=True)
            st.pyplot(fig)
        elif selected_strategy == 'MACD':
            data = self.ic.calculate_macd(data)
            fig, axes = mpf.plot(data, type='candle', style='yahoo', title=f'Stock Chart with MACD for {self.stock_symbol}',
                                ylabel='Stock Price', addplot=[mpf.make_addplot(data['MACD'], color='blue', secondary_y=True),
                                                            mpf.make_addplot(data['Signal Line'], color='red', secondary_y=True)],
                                returnfig=True)
            st.pyplot(fig)
