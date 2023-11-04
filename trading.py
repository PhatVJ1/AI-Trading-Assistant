import streamlit as st
import yfinance as yf
import mplfinance as mpf
import pandas as pd

# Lấy danh sách các mã chứng khoán từ yfinance
stock_info = yf.Tickers('Meta')

# Create a Streamlit app
st.title('Stock Chart')

# List of stock symbols
stock_symbols = [ticker.info['symbol'] for ticker in stock_info.tickers]

# Customize the stock symbol
stock_symbol = st.selectbox('Select a stock symbol', stock_symbols)

# Customize the start date
start_date = st.date_input('Start Date', pd.to_datetime("2018-11-01"))

# Customize the end date
end_date = st.date_input('End Date', pd.to_datetime("2023-10-04"))

# Download data from yfinance based on the selected stock symbol, start date, and end date
data_frame = yf.download(stock_symbol, start=start_date, end=end_date, interval="1d")

# Create a copy of the DataFrame to work with the data
data = data_frame.copy()

# Calculate EMA 50, EMA 100, and EMA 200 for the 'Close' column
data['EMA50'] = data['Close'].ewm(span=50, adjust=False).mean()
data['EMA100'] = data['Close'].ewm(span=100, adjust=False).mean()
data['EMA200'] = data['Close'].ewm(span=200, adjust=False).mean()

# Calculate SMA and Bollinger Bands
def calculate_sma(data, window):
    data['SMA'] = data['Close'].rolling(window=window).mean()
    return data

def calculate_bollinger_bands(data, window, num_std_dev):
    data['SMA'] = data['Close'].rolling(window=window).mean()
    data['STD'] = data['Close'].rolling(window=window).std()
    data['Upper'] = data['SMA'] + (data['STD'] * num_std_dev)
    data['Lower'] = data['SMA'] - (data['STD'] * num_std_dev)
    return data

# Parameters for SMA and Bollinger Bands
sma_window = 20
bb_window = 20
num_std_dev = 2


# Function to compute the zigzag line
def compute_zigzag_line(data, threshold):
    zigzag_x = []
    zigzag_y = []
    current_zigzag_x = [data.index[0]]
    current_zigzag_y = [data['Close'][0]]
    direction = 1

    for i in range(1, len(data)):
        if direction == 1:
            if data['Close'][i] - data['Close'][i - 1] > threshold:
                current_zigzag_x.append(data.index[i])
                current_zigzag_y.append(data['Close'][i])
            else:
                if len(current_zigzag_x) > 1:  # Only add if there is more than one point in the zigzag
                    zigzag_x += current_zigzag_x
                    zigzag_y += current_zigzag_y
                current_zigzag_x = [data.index[i]]
                current_zigzag_y = [data['Close'][i]]
                direction = -1
        else:
            if data['Close'][i - 1] - data['Close'][i] > threshold:
                current_zigzag_x.append(data.index[i])
                current_zigzag_y.append(data['Close'][i])
            else:
                if len(current_zigzag_x) > 1:  # Only add if there is more than one point in the zigzag
                    zigzag_x += current_zigzag_x
                    zigzag_y += current_zigzag_y
                current_zigzag_x = [data.index[i]]
                current_zigzag_y = [data['Close'][i]]
                direction = 1

    zigzag = pd.DataFrame({'Date': zigzag_x, 'Zigzag': zigzag_y})
    zigzag.set_index('Date', inplace=True)
    return zigzag

threshold = 0.1  # You can adjust this threshold value
zigzag_data = compute_zigzag_line(data, threshold)

# Select the strategy
selected_strategy = st.selectbox('Select a strategy', ['Candlestick', 'EMA', 'SMA', 'Bollinger Bands'])

# Display the chart based on the selected strategy
if selected_strategy == 'Candlestick Chart':
    fig, axes = mpf.plot(data, type='candle', style='yahoo', title=f'Stock Chart for {stock_symbol}',
                        ylabel='Stock Price', returnfig=True)
    st.pyplot(fig)
elif selected_strategy == 'EMA':
    fig, axes = mpf.plot(data, type='candle', style='yahoo', title=f'Stock Chart with EMA for {stock_symbol}',
                        ylabel='Stock Price', addplot=[mpf.make_addplot(data['EMA50'], color='blue'),
                                                    mpf.make_addplot(data['EMA100'], color='green'),
                                                    mpf.make_addplot(data['EMA200'], color='red')],
                        returnfig=True)
    st.pyplot(fig)
elif selected_strategy == 'SMA':
    data = calculate_sma(data, sma_window)
    fig, axes = mpf.plot(data, type='candle', style='yahoo', title=f'Stock Chart with SMA for {stock_symbol}',
                        ylabel='Stock Price', ylabel_lower='SMA', mav=(sma_window,),
                        hlines=dict(hlines=[data['SMA'].mean()], colors='red', linestyle='dashed'),
                        returnfig=True)
    st.pyplot(fig)
elif selected_strategy == 'Bollinger Bands':
    data = calculate_bollinger_bands(data, bb_window, num_std_dev)
    fig, axes = mpf.plot(data, type='candle', style='yahoo', title=f'Stock Chart with Bollinger Bands for {stock_symbol}',
                        ylabel='Stock Price', ylabel_lower='Bollinger Bands', mav=(sma_window,),
                        addplot=[mpf.make_addplot(data['Upper'], color='purple', secondary_y=False),
                                mpf.make_addplot(data['Lower'], color='purple', secondary_y=False)],
                        returnfig=True)
    st.pyplot(fig)
elif selected_strategy == 'ZigZag':
    fig, axes = mpf.plot(data, type='candle', style='yahoo', title=f'Candlestick Chart with Zigzag Line for {stock_symbol}',
                         volume=True)
    st.pyplot(fig)

# # Plot the zigzag line
# plt.plot(zigzag_data.index, zigzag_data['Zigzag'], label='Zigzag Line', color='r', linewidth=2)
# plt.xlabel('Date')
# plt.ylabel('Zigzag Line')
# plt.legend()

