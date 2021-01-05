# inspiration was from - https://medium.com/swlh/creating-a-trading-strategy-from-scratch-in-python-fe047cb8f12
# Looking to find a good indicator to find stocks that have already made their move to sell calls/puts
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import ta
import yfinance as yf
import mplfinance as mpf
import plotly.graph_objs as go
import plotly.io as pio
pio.renderers.default = "browser"
from numpy import zeros
from pandas import DataFrame
# ETFs that we do not need compliance approval
tickers = ['DIA', 'IYY', 'QQQ', 'PSQ', 'QQQE', 'SPSM', 'VTWO', 'IWM', 'RWM', 'IWV', 'SPTM', 'VTHR', 'OEF', 'SH', 'SPXB',
           'SPY', 'IVV', 'VOO', 'VXXB', 'SPDN', 'RSP', 'PBP', 'MIDU', 'MYY', 'MZZ', 'IEV', 'MIDD', 'EWU', 'EWH', 'TYBS',
           'TYNS', 'TBT', 'SHY', 'TLT', 'IEF', 'STIP', 'GVI', 'TLH', 'FXA', 'FXB', 'FXC', 'FXCH', 'FXE', 'FXY', 'FXSG',
           'FXSG', 'FXF', 'DBV', 'UDN', 'UUP', 'DBA', 'DBB', 'DBC', 'DBE', 'DGL', 'DBO', 'DBP', 'DBS', 'PDBC', 'SGOL',
           'PPLT', 'GLTR', 'SIVR', 'PALL', 'BCI', 'AOIL', 'DJP', 'OIL', 'JO', 'BCM', 'GSP', 'SGG', 'NIB', 'JJG',
           'JJC', 'COW', 'BAL', 'DGZ', 'OILK', 'IAU', 'CMDY']

data = yf.download('SPY', period='6mo')
df = pd.DataFrame(data)

df = pd.DataFrame(df[['Open', 'High', 'Low', 'Close', 'Volume']])
print(df.isnull().sum())# check to see if any zeroes
print(df.head())
plt.plot(df['Close'], linewidth=0.75, label='', color = 'blue')
plt.show() #look normal?

#MFN = ((close - low) - (high - close) / (high - low)) *100
df["mfm_cml"] = df["Close"] - df["Low"]
df['mfm_hmc'] = df["High"] - df["Close"]
df["mfn_hml"] = df["High"] - df["Low"]
df["mfmB"] = ((df["mfm_cml"] - df['mfm_hmc']) / df["mfn_hml"])
df["mfm"] = ((df["mfm_cml"] - df['mfm_hmc']) / df["mfn_hml"])*100
df.head()
print(df.isnull().sum())

plt.plot(df["mfm_cml"], linewidth=0.75, label='1', color = 'blue')
plt.plot(df['mfm_hmc'], linewidth=0.75, label='2', color = 'r')
plt.plot(df["mfn_hml"], linewidth=0.75, label='2', color = 'g')
plt.plot(df["mfmB"], linewidth=0.95, label='2', color = 'black')
plt.show() #just a visual check

#Charts https://github.com/matplotlib/mplfinance
mfn =pd.DataFrame(df["mfm"])
lower_b = -90
high_b = 90


fig = mpf.figure(figsize=(12,9))

ax1 = fig.add_subplot(2,2,1,style='blueskies')
ax2 = fig.add_subplot(2,2,2,style='yahoo')
s   = mpf.make_mpf_style(base_mpl_style='fast', base_mpf_style='nightclouds')
ax3 = fig.add_subplot(2,2,3,style=s)
ax4 = fig.add_subplot(2,2,4,style='starsandstripes')

mpf.plot(df,ax=ax1,axtitle='Renko', type='renko', xrotation=15)
mpf.plot(df,type='pnf',ax=ax2,axtitle='PNF', xrotation=15)
mpf.plot(df,ax=ax3,type='candle',axtitle='nightclouds', mav=(3, 6, 9))
mpf.plot(df,type='candle',ax=ax4,axtitle='starsandstripes')

mpf.show()

mpf.plot(df, type='renko')
mpf.plot(df, type='pnf')
mpf.plot(df, type='candle', mav=(3, 6, 9))
mpf.plot(df)

price = df["Close"]
def mfn_below90(mfn,price):
    import numpy as np
    signal   = []
    previous = -1.0
    for date,value in mfn.iteritems():
        if value < -90 and previous >= -90:
            signal.append(price[date]*0.99)
        else:
            signal.append(np.nan)
        previous = value
    return signal
def mfn_above90(mfn,price):
    import numpy as np
    signal   = []
    previous = -1.0
    for date,value in mfn.iteritems():
        if value < 90 and previous >= 90:
            signal.append(price[date]*1.01)
        else:
            signal.append(np.nan)
        previous = value
    return signal

low_signal = mfn_below90(df['mfm'], df['Close'])
high_signal = mfn_above90(df['mfm'], df['Close'])

apd =[ mpf.make_addplot(low_signal, type='scatter', markersize=100, marker='^'),
       mpf.make_addplot(high_signal, type='scatter', markersize=100, marker='v'),
       mpf.make_addplot((df['mfm']), panel=1, color='g')
     ]
mpf.plot(df, type='candle', addplot=apd)


def signal(df, price, buy, sell):
    for i in range(len(df)):

        if df[i, price] < low_signal and df[i - 1, price] > low_signal and df[i - 2, price] > low_signal:
            df[i, buy] = 1

        if df[i, price] > high_signal and df[i - 1, price] < high_signal and df[i - 2, price] < high_signal:
            df[i, sell] = -1