import streamlit as st
import yfinance as yf
import mplfinance as mpf
import pandas as pd
from datetime import date, timedelta
from streamlit_process import plot as plt 
from AI_Trading_Assistant_Model.Channel_Aligned_Robust_Dual_Transformer import CARD
from AI_Trading_Assistant_Model.input_created import data_generator as dg
import tensorflow as tf


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


plot = plt.draw(stock_symbol,20,20)
plot(data)

data_for_pred = data[['Open', 'Close']].tail(60)
data_for_pred, _ = dg(data=data_for_pred, seq_train=50, seq_label=0)
model = None
if "model_check" not in st.session_state:
    st.session_state.model_check = False


def model_run(input, model, model_name):
    model.load_weights("AI-Trading-Assistant/AI_Trading_Assistant_Model/model_weights/" + model_name + ".h5")
    return model.predict(input)

if st.button("Chạy Model"):
    if st.session_state.model_check == False:
        model = CARD.Transformer(50, 3, 2, 3, 1, 2048, 8, 64, 0.9, 3)
        CARD.model_builder(model, data_for_pred[1:])
        st.session_state.model_check = True
    result = model_run(data_for_pred, model, stock_symbol)  # Gọi hàm chạy model
    st.write("Kết quả: ", result)  # Hiển thị kết quả


