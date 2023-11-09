import streamlit as st
import yfinance as yf
import mplfinance as mpf
import pandas as pd
from datetime import date, timedelta
from streamlit_process import plot as plt 
from AI_Trading_Assistant_Model.Channel_Aligned_Robust_Dual_Transformer import CARD
import tensorflow as tf
import plotly.graph_objects as go

# Tạo ứng dụng Streamlit
st.title('Biểu đồ nến và các chiến thuật')

# Danh sách mã cổ phiếu
stock_symbols = ['META', 'AAPL', 'MSFT', 'TSLA', 'AMZN', 'ADA-USD',
                 'XRP-USD', 'ETH-USD', 'CLDX', 'XPEV',
                 'NVDA', 'GOOGL']

stock_symbol = st.selectbox('Chọn mã cổ phiếu', stock_symbols)

# Trường nhập liệu số tháng muốn lùi lại
months_to_subtract = st.number_input('Nhập số tháng muốn lùi lại', value=6)

# Tự động cập nhật ngày hiện tại
current_date = date.today()

# Tự động tính ngày bắt đầu và kết thúc dựa trên số tháng và ngày hiện tại
end_date = current_date
start_date = end_date - pd.DateOffset(months=months_to_subtract)

# Hiển thị ngày bắt đầu và ngày kết thúc
st.write(f"Ngày bắt đầu: {start_date}")
st.write(f"Ngày kết thúc: {end_date}")

# Download data from yfinance based on the selected stock symbol, start date, and end date
data_frame = yf.download(stock_symbol, start=start_date, end=end_date, interval="1d")

# Create a copy of the DataFrame to work with the data
data = data_frame.copy()


plot = plt.draw(stock_symbol,20,20,12,26,9)
plot(data)

data_for_pred = tf.expand_dims(tf.convert_to_tensor(data[['Open', 'Close']].tail(50)), axis=0)

def model_run(input, model, model_name):
    model.load_weights("AI_Trading_Assistant_Model/model_weights/" + model_name + ".h5")
    return tf.squeeze(model.predict(input))

if st.button("Chạy Model"):
    
    model = CARD.Transformer(50, 3, 2, 3, 1, 2048, 8, 64, 0.9, 3)
    CARD.model_builder(model, data_for_pred[1:])

    result = model_run(data_for_pred, model, stock_symbol)  # Gọi hàm chạy model

    # Tạo một danh sách (list) để lưu trữ các ngày sau khi thêm 3 ngày
    date_list = [end_date]

    # Thêm 3 ngày vào danh sách
    for i in range(1, 4):
        new_date = end_date + timedelta(days=i)
        date_list.append(new_date)

    fig = go.Figure(data=[go.Candlestick(x=date_list,
                    open=result[0, :],
                    high=result[0, :],
                    low=result[1, :],
                    close=result[1, :])])

    fig.update_layout(title='Candlestick Chart Predict')
    # Hiển thị biểu đồ trong Streamlit
    st.plotly_chart(fig)

    if result[1,0] > result[1,1] and result[1,1] > result[1,2]:
        st.header("Xu hướng sắp tới sẽ giảm")
    if result[1,0] < result[1,1] and result[1,1] < result[1,2]:
        st.header("Xu hướng sắp tới sẽ tăng")
    if result[1,0] > result[1,1] and result[1,1] < result[1,2] or result[1,0] < result[1,1] and result[1,1] > result[1,2]:
        st.header("Xu hướng thị trường đang tích lũy")

