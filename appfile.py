import streamlit as st
from datetime import date
from plotly import graph_objs as go
from yahoofinancials import YahooFinancials
from nsetools import Nse
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import math
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from yahoofinancials import YahooFinancials

import warnings
warnings.filterwarnings('ignore')

st.title("Stock Price Prediction using LSTM")
st.write("Since the models are built from scratch to arrive at the prediction, the application will take some time to execute")
st.subheader("Enter the ticker(s) of stock(s) for which close price has to be predicted")
st.write("Only enter the tickers of stocks which are listed in NSE")
st.write("If multiple stocks are entered, separate the tickers by commas")
stocks = st.text_area(label="Stock(s)")
a=stocks.split(',')
stock_ticker = [str(x)+".NS" for x in a]


    
@st.cache
def load_data(tickers):
    maindf=pd.DataFrame()
    start=(date.today()-datetime.timedelta(days=4*365)).strftime('%Y-%m-%d')
    end=date.today().strftime('%Y-%m-%d')
    raw_data = YahooFinancials(tickers).get_historical_price_data(start,end,"daily")
    for i in tickers:
        data=pd.DataFrame(raw_data[i]["prices"])[['formatted_date','close']]
        data=data.dropna()
        data.rename(columns={'formatted_date':'Date','close':'Close'},inplace=True)
        data.set_index('Date',inplace=True)
        maindf[i] = data['Close']
    return maindf

def plot_raw_data(data):
    fig=go.Figure()
    for i in data:
        fig.add_trace(go.Scatter(x=data.index,y=data[i],name=i[:-3]))
        fig.layout.update(title_text="Close Price of the Stock(s) over a period of time",xaxis_rangeslider_visible=True)
        fig.update_layout(
        title="Recent Close Prices of stocks",
        xaxis_title="Date",
        yaxis_title="Close Price",
        legend_title="Company",
        font=dict(
            family="Courier New, monospace",
            size=15,
            color="RebeccaPurple"
        )
        )
    
    st.plotly_chart(fig)    

@st.cache
def predictions(tickers,data):
  @st.cache  
  def sliding_window(records, time_step):
    dataX, dataY = [], []
    for i in range(len(records)-time_step):
      a = records[i:(i+time_step), 0]   
      dataX.append(a)
      dataY.append(records[i + time_step, 0])
    return np.array(dataX), np.array(dataY)
  results = pd.DataFrame()
  rmse_percentage=[]

  for i in tickers:
      df=data[i]
      scaler = MinMaxScaler(feature_range=(0,1))
      df = scaler.fit_transform(np.array(df).reshape(-1,1))
      train_data = df[:data.shape[0]-5]
      test_data=df[data.shape[0]-6:]
      time_step = 5
      X_train, y_train = sliding_window(train_data, time_step)
      X_test, y_test = sliding_window(test_data, time_step)
      X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] , 1)
      X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] , 1)
      lstmmodel = Sequential()
      lstmmodel.add(LSTM(256, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))    
      lstmmodel.add(Dropout(0.3))
      lstmmodel.add(LSTM(256))
      lstmmodel.add(Dropout(0.3))
      lstmmodel.add(Dense(1))                                              
      lstmmodel.compile(loss='mean_squared_error', optimizer='adam')
      history = lstmmodel.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=128, verbose=False)
      y_pred = lstmmodel.predict(X_test)
      y_pred1 = scaler.inverse_transform(y_pred)
      y_test1 = scaler.inverse_transform(y_test.reshape(-1, 1))
      rmse=round(math.sqrt(mean_squared_error(y_test1, y_pred1)), 4)
      rmse_per = rmse/np.mean(y_test1)*100
      rmse_percentage.append(rmse_per)
      results[i] = y_pred1.flatten()
  results.index=["Prediction of Close Price"]
  results=results.T
  results["Prediction of Close Price"] = results["Prediction of Close Price"].apply(lambda x:round(x,2))
  return results

if st.button("Forecast the Close Price of the Stock(s)"):
    data = load_data(stock_ticker)
    st.subheader('Recent Prices of the Stock(s)')
    st.write(data.tail())
    plot_raw_data(data)
    result = predictions(stock_ticker,data)
    st.subheader('Forecasted Value(s)')
    st.write("The training data is available until ",(pd.to_datetime(data.index[-1]).strftime('%d %B, %Y')),". The prediction of close price at the end of next trading day is as follows")
    st.write(result)
    

    

 
