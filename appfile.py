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
no_stocks=(1,2,3,4,5,6,7,8,9,10)
selected_stocks = st.selectbox("Enter the number of stocks",no_stocks) 
l=[]     
for i in range(1,selected_stocks+1):
    row1_1, row1_2,row1_3,row1_4 = st.columns((1,1,1,1))

    with row1_1:
        ticker=st.text_input(label="Ticker "+str(i))
    
    with row1_2:
        price=st.text_input(label="Purchase Price "+str(i))
        
    with row1_3:
        nou=st.text_input(label="Number of Units "+str(i))
    
    with row1_4:
        er=st.text_input(label="Expected Return "+str(i))
    l.append([ticker,price,nou,er])   
    
d=pd.DataFrame(l,columns=['Ticker','Price','No of units','Expected Return'])
d['Expected Return'] = d['Expected Return'].astype('float64')
d['No of units'] = d['No of units'].astype('float64')
d['Price'] = d['Price'].astype('float64')
d.set_index('Ticker',inplace=True)
d.index = [str(x)+'.NS' for x in list(d.index).copy()]

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
    return  maindf

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

stock_ticker=list(d.index)
if st.checkbox("Forecast the Close Price of the Stock(s)"):
    data = load_data(stock_ticker)
    st.subheader('Recent Prices of the Stock(s)')
    st.dataframe(data.tail().style.format("{:.2f}"))
    plot_raw_data(data)
    result = predictions(stock_ticker,data)
    final=pd.concat([d,result],axis=1)
    final['Bought Price'] = (final['Price'].astype('float'))*final['No of units'].astype('int')
    final['Expected Price'] = (final['Prediction of Close Price'].astype('float'))*(final['No of units'].astype('int'))
    final['Predicted Return'] = (((final['Expected Price'] - final['Bought Price'])/final['Bought Price'])*100)
    final['+/-'] = final['Predicted Return'] >= final['Expected Return']
    final['+/-'].replace({True:"+",False:"-"},inplace=True)
    pred_port=((final['Expected Price'].sum() - final['Bought Price'].sum() )/ final['Bought Price'].sum())*100
    st.subheader('Forecasted Value(s)')
    st.write("The training data is available until ",(pd.to_datetime(data.index[-1]).strftime('%d %B, %Y')),". The prediction of close price at the end of next trading day is as follows")
    st.dataframe(pd.DataFrame(final['Prediction of Close Price']).style.format(({"Prediction of Close Price": "{:.2f}"})))
    st.subheader('Comparison of Returns')
    st.dataframe(final[['Expected Return','Predicted Return','+/-']].style.format(({"Expected Return": "{:.2f}","Predicted Return": "{:.1f}"})))
    if st.checkbox(label="Check here if you consider the stocks entered as portfolio"):
        exp_ret=st.number_input(label="Enter the expected rate of return of the portfolio")
        if exp_ret!=0.00 :
            st.write("Predicted rate of return of the portfolio is " + str(np.round(pred_port,2)))
            if pred_port < float(exp_ret):
                st.error('Portfolio is not performing upto expectation')
            else:
                st.success('Portfolio is performing upto expectation')
        else:
            st.error("Enter expected rate of return of portfolio")
