import numpy as np

"""input section"""
"""stockinput can be either an upper case or lower case string. stock will uppercase the input string since stock ticker symbols are capitalized"""
stockinput = (input("Enter Stock Ticker Symbol: "))
stock = stockinput.upper()

"""Dates Section"""
"""Date Libraries"""
import datetime 
from datetime import date

today = date.today()

#get day from one year ago
start = datetime.datetime.now() - datetime.timedelta(days=365)
end = datetime.datetime(today.year, today.month, today.day)

"""Datagathering section"""
"""Pandas libraries for data gathering"""
import pandas as pd 
import pandas_datareader.data as web
from pandas import Series, DataFrame
from pandas_datareader._utils import RemoteDataError
from pandas_datareader.data import Options

"""Using Dataframe Constructor for data gathering"""
df = web.DataReader(stock, 'yahoo', start, end)

"""write csv data"""
csvurl1 = stock+'_shortterm.csv'
outputdata1 = df.to_csv(csvurl1)

"""Read csv to show end of the csv data"""
inputdata1 = pd.read_csv(csvurl1)

"""Section where moving average function is calculated"""
"""Will be using a numpy libary"""
import numpy as np
import csv

listdata = []

listdata = df[['Adj Close']]

# Calculating the short-window simple moving average
inputdata1['MA20'] = inputdata1['Close'].rolling(20).mean()
inputdata1['MA100'] = inputdata1['Close'].rolling(100).mean()
#inputdata1 = inputdata1.dropna()
#inputdata1.head()

"""Section where Moving Averages graphs are made"""
"""Libraries to plot data"""
import plotly.graph_objects as go
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

inputdata1['Date'] = pd.to_datetime(inputdata1['Date'])

price_date = inputdata1['Date']


#Create y-data points
y1 = inputdata1['Adj Close']
y2 = inputdata1['MA20']

#plot the short rolling data
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.plot(price_date, y1, color ='tab:green', label='Price')
ax.plot(price_date, y2, color = 'tab:red', label='MA20')

#set vairables for x-axis
plt.gcf().autofmt_xdate()

#set labels
ax.legend()
plt.ylabel('Stock Price in USD')
plt.xlabel('Date')

#set title
ax.set_title(stock+' 20 Day Moving Average Analysis')

#display the plot
plt.show()

#plot long rolling data
#set y-axis
y3 = inputdata1['MA100']

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.plot(price_date, y1, color = 'tab:green', label='Price')
ax.plot(price_date, y3, color = "tab:orange", label='MA100')
plt.gcf().autofmt_xdate()

#set labels
ax.legend()
plt.ylabel('Stock Price in USD')
plt.xlabel('Date')

#set title
ax.set_title(stock+' 100 Day Moving Average Analysis')

#display the plot
plt.show()

#Plot All Data
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.plot(price_date, y1, color = 'tab:green', label='Price')
ax.plot(price_date, y2, color = 'tab:red', label='MA20')
ax.plot(price_date, y3, color = "tab:orange", label='MA100')
plt.gcf().autofmt_xdate()

#set labels
ax.legend()
plt.ylabel('Stock Price in USD')
plt.xlabel('Date')

#set title
ax.set_title(stock+' 1-Year Moving Average Analysis')

#display the plot
plt.show()

"""Show price and trading volume"""
fig = go.Figure([go.Scatter(x = inputdata1['Date'], y = inputdata1['Adj Close'])])
fig.update_layout(title=stock+' Price Analysis', plot_bgcolor='rgb(230,230,230)', showlegend=False)
fig.show()

fig = go.Figure([go.Scatter(x = inputdata1['Date'], y = inputdata1['Volume'])])
fig.update_layout(title=stock+' Trading Volume Analysis', plot_bgcolor='rgb(230,230,230)', showlegend=False)
fig.show()

"""Plot a Graph To Show Profit/Loss"""
#Add a new column "Shares", if MA10>MA50, denote as 1 (long share of stock), otherwise denoted as 0 (do nothing)
inputdata1['Shares'] = [1 if inputdata1.loc[ei, 'MA20']>inputdata1.loc[ei, 'MA100'] else 0 for ei in inputdata1.index]

inputdata1['Close1'] = inputdata1['Close'].shift(-1)
inputdata1['Profit'] = [inputdata1.loc[ei, 'Close1'] - inputdata1.loc[ei, 'Close'] if inputdata1.loc[ei, 'Shares']==1 else 0 for ei in inputdata1.index]

plt.plot(price_date, inputdata1['Profit'])

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=60))

plt.gcf().autofmt_xdate()

plt.axhline(y=0, color='red')
plt.show()

"""Plot A Graph to show Gains"""
#use .cumsum() to calculate the accumulated wealth over the period
inputdata1['wealth'] = inputdata1['Profit'].cumsum()

#plot the wealth to show the grapth of profit over the period
plt.title('Total money you win is {}'.format(inputdata1.loc[inputdata1.index[-2], 'wealth']))

plt.gcf().autofmt_xdate()

plt.plot(price_date, inputdata1['wealth'])

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=60))

plt.title(stock+' Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
