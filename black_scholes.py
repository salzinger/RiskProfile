import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

import pandas_datareader.data as web
import pandas as pd
import datetime as dt



N = norm.cdf

def BS_CALL(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * N(d1) - K * np.exp(-r*T)* N(d2)

def BS_delta(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    #d2 = d1 - sigma * np.sqrt(T)
    return N(d1)

def BS_PUT(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma* np.sqrt(T)
    return K*np.exp(-r*T)*N(-d2) - S*N(-d1)
fig, ax = plt.subplots(2, 2, figsize=(10, 10))


K = 100
r = 0.1
T = 1
sigma = 0.3

S = np.arange(60, 140, 0.1)

calls = [BS_CALL(s, K, T, r, sigma) for s in S]
puts = [BS_PUT(s, K, T, r, sigma) for s in S]
delta = [BS_delta(s, K, T, r, sigma) for s in S]

ax[0, 0].plot(S, calls, label='Call Value', color="g")
ax[0, 0].plot(S, puts, label='Put Value', color="r")
ax[1, 1].plot(S, delta, label='delta', color="r")

ax[0, 0].set_xlabel('$S_0$')
ax[0, 0].set_ylabel(' Value')
ax[0, 0].legend()

K = 100
r = 0.1
T = 1
Sigmas = np.arange(0.1, 1.5, 0.01)
S = 100

calls = [BS_CALL(S, K, T, r, sig) for sig in Sigmas]
puts = [BS_PUT(S, K, T, r, sig) for sig in Sigmas]
ax[0, 1].plot(Sigmas, calls, label='Call Value', color="g")
ax[0, 1].plot(Sigmas, puts, label='Put Value', color="r")
ax[0, 1].set_xlabel('$\sigma$')
ax[0, 1].set_ylabel(' Value')
ax[0, 1].legend()


K = 100
r = 0.05
T = np.arange(0, 2, 0.01)
sigma = 0.3
S = 100

calls = [BS_CALL(S, K, t, r, sigma) for t in T]
puts = [BS_PUT(S, K, t, r, sigma) for t in T]
ax[1, 0].plot(T, calls, label='Call Value', color="g")
ax[1, 0].plot(T, puts, label='Put Value', color="r")
ax[1, 0].set_xlabel('$T$ in years')
ax[1, 0].set_ylabel(' Value')
ax[1, 0].legend()




plt.show()







'''
start = dt.datetime(2010,1,1)
end =dt.datetime(2020,10,1)
symbol = 'AAPL' ###using Apple as an example
source = 'yahoo'
data = web.DataReader(symbol, source, start, end)
data['change'] = data['Adj Close'].pct_change()
data['rolling_sigma'] = data['change'].rolling(20).std() * np.sqrt(255)


data.rolling_sigma.plot()
plt.ylabel('$\sigma$')
plt.title('AAPL Rolling Volatility')
'''
















































