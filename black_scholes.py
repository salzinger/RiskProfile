import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

import pandas_datareader.data as web
import pandas as pd
import datetime as dt



N = norm.cdf

phi=norm.pdf

def BS_CALL(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * N(d1) - K * np.exp(-r*T) * N(d2)

def BS_CALL_delta(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    #d2 = d1 - sigma * np.sqrt(T)
    return N(d1)

def BS_PUT_delta(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    #d2 = d1 - sigma * np.sqrt(T)
    return N(d1)-1

def BS_gamma(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    #d2 = d1 - sigma * np.sqrt(T)
    return phi(d1)/(S*sigma*np.sqrt(T))

def BS_vega(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    #d2 = d1 - sigma * np.sqrt(T)
    return phi(d1)*(S*np.sqrt(T))



def BS_PUT(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma* np.sqrt(T)
    return K*np.exp(-r*T)*N(-d2) - S*N(-d1)



fig, ax = plt.subplots(3, 3, figsize=(10, 10))


K = 100
r = 0.1
T = 1
sigma = 0.3

S = np.arange(60, 140, 0.1)

calls = [BS_CALL(s, K, T, r, sigma) for s in S]
puts = [BS_PUT(s, K, T, r, sigma) for s in S]
cdelta = [BS_CALL_delta(s, K, T, r, sigma) for s in S]
pdelta = [BS_PUT_delta(s, K, T, r, sigma) for s in S]
gamma = [BS_gamma(s, K, T, r, sigma) for s in S]



ax[0, 0].plot(S, calls, label='Call Value', color="g")
ax[0, 0].plot(S, puts, label='Put Value', color="r")



ax[0, 0].set_xlabel('Underlying Stock Value $S_0$')
ax[0, 0].set_ylabel('Option Value')
ax[0, 0].legend()

ax[1, 0].plot(S, cdelta, label='Call $\Delta$', color="g")
ax[1, 0].plot(S, pdelta, label='Put $\Delta$', color="r")
ax[1, 0].set_xlabel('Underlying Stock Value $S_0$')
ax[1, 0].set_ylabel('Option $\Delta$')
ax[1, 0].legend()

ax[2, 0].plot(S, gamma, label='$\Gamma$', color="black")
ax[2, 0].set_xlabel('Underlying Stock Value $S_0$')
ax[2, 0].set_ylabel('Option $\Gamma$')
ax[2, 0].legend()


K = 100
r = 0.1
T = 1
Sigmas = np.arange(0.1, 1.5, 0.01)
S = 100

calls = [BS_CALL(S, K, T, r, sig) for sig in Sigmas]
puts = [BS_PUT(S, K, T, r, sig) for sig in Sigmas]

vega = [BS_vega(S, K, T, r, sig) for sig in Sigmas]

ax[0, 1].plot(Sigmas, calls, label='Call Value', color="g")
ax[0, 1].plot(Sigmas, puts, label='Put Value', color="r")
ax[0, 1].set_xlabel('Volatility $\sigma$')
ax[0, 1].set_ylabel('Option Value')
ax[0, 1].legend()

ax[1, 1].plot(Sigmas, vega, label='$\Lambda$', color="black")
#ax[1, 1].plot(Sigmas, puts, label='Put Value', color="r")
ax[1, 1].set_xlabel('Volatility $\sigma$')
ax[1, 1].set_ylabel('Option Vega $\Lambda$')
ax[1, 1].legend()


K = 100
r = 0.05
T = np.arange(0, 2, 0.01)
sigma = 0.3
S = 100

calls = [BS_CALL(S, K, t, r, sigma) for t in T]
puts = [BS_PUT(S, K, t, r, sigma) for t in T]
ax[0, 2].plot(T, calls, label='Call Value', color="g")
ax[0, 2].plot(T, puts, label='Put Value', color="r")
ax[0, 2].set_xlabel('Time until expiration in years')
ax[0, 2].set_ylabel('Option Value')
ax[0, 2].legend()









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


import mibian
import quandl

quandl.ApiConfig.api_key = "GtsKPdfA8r4Ry8egnoSv"

c = mibian.GK([1.4565, 1.45, 1, 2, 30], volatility=20)

print(c.callPrice)

def Brownian(seed, N):
    np.random.seed(seed)
    dt = 1. / N  # time step
    b = np.random.normal(0., 1., int(N)) * np.sqrt(dt)  # brownian increments
    W = np.cumsum(b)  # brownian path
    return W, b

def daily_return(adj_close,start,end):
    returns = []
    for i in range(start, end):
        today = adj_close[i+1]
        yesterday = adj_close[i]
        daily_return = (today - yesterday)/yesterday
        returns.append(daily_return)
    return returns

def GBM(So, mu, sigma, W, T, N):
    t = np.linspace(0., 1., N+1)
    S = []
    S.append(So)
    for i in range(1, int(N+1)):
        drift = (mu - 0.5 * sigma**2) * t[i]
        diffusion = sigma * W[i-1]
        S_temp = So*np.exp(drift + diffusion)
        S.append(S_temp)
    return S, t

seed = 5
N = 99  # increments

# brownian increments
b = Brownian(seed, N)[1]

# brownian motion
W = Brownian(seed, N)[0]
W = np.insert(W, 0, 0.)  # W_0 = 0. for brownian motion

# brownian increments


start = "2016-01-01"
end = "2021-01-11"

df = quandl.get("WIKI/AMZN", start_date=start, end_date=end)
print(df)
adj_close = df['Adj. Close']
time = np.linspace(1, len(adj_close), len(adj_close))


ax[2, 1].set_ylabel('Stock Price, $')
ax[2, 1].set_xlabel('Trading Days')

So = adj_close[460]
mu = 0.6
sigma = 0.1
seed = 22
N = 99
W = Brownian(seed, N)[0]
T = 100
soln = GBM(So, mu, sigma, W, T, N)[0]    # Exact solution
t = GBM(So, mu, sigma, W, T, N)[1]       # time increments for  plotting
ax[2, 2].plot(t, soln, label="GBM $\mu$="+str(mu)+" $\sigma$="+str(sigma)+" and seed "+str(seed))

mu = 0.15
sigma=0.3
soln = GBM(So, mu, sigma, W, T, N)[0]    # Exact solution
t = GBM(So, mu, sigma, W, T, N)[1]       # time increments for  plotting
ax[2, 2].plot(t, soln, label="GBM $\mu$="+str(mu)+" $\sigma$="+str(sigma)+" and seed "+str(seed))



ax[2, 2].plot(t, adj_close[460:len(adj_close)], label="Actual Stock", color="black")
ax[2, 2].set_xlabel('Trading Days')
ax[2, 2].set_ylabel('Stock Price')
ax[2, 2].legend(loc="upper left")

time = np.linspace(1, len(adj_close), len(adj_close))


returns = daily_return(adj_close, 0, 460)
mu = np.mean(returns)*460.           # drift coefficient
sig = np.std(returns)*np.sqrt(460.)  # diffusion coefficient
print(mu, sig)
seed = 22
W = Brownian(seed, N)[0]
soln2 = GBM(So, mu, sig, W, T, N)[0]    # Exact solution
t = GBM(So, mu, sig, W, T, N)[1]       # time increments for  plotting
ax[2, 1].plot(time[460:len(adj_close)], soln2, label="GBM fitted $\mu$="+str(round(mu, 2))+", $\sigma$="+str(round(sig, 2))+" and seed "+str(seed))
ax[2, 1].plot(time, adj_close, label="Actual Stock", color="black")
ax[2, 1].set_xlabel('Trading Days')
ax[2, 1].set_ylabel('Stock Price')
ax[2, 1].legend(loc="upper left")









def call_value(underlying, call_strike_price, call_break_even, call_initial_invest, call_premium_percentage):
    return np.heaviside(underlying - call_strike_price, 1) * (underlying - call_strike_price) * \
           call_initial_invest * (1 + call_premium_percentage) / (call_break_even - call_strike_price) - call_initial_invest * (1 + call_premium_percentage)

def put_value(underlying, put_strike_price, put_break_even, put_initial_invest, put_premium_percentage):
    return np.heaviside(put_strike_price - underlying, 1) * (underlying - put_strike_price) * \
           put_initial_invest * (1 + put_premium_percentage) /(put_break_even - put_strike_price) - put_initial_invest * (1 + put_premium_percentage)



#Tesla
current_underlying = 811.19
#https://wertpapiere.ing.de/Investieren/Derivat/DE000JC57SA4
put_strike_price = 1015
put_break_even = 774.3
put_initial_invest = 750
put_premium_percentage = 7.5/100


#https://finance.yahoo.com/quote/BABA/?guccounter=1&guce_referrer=aHR0cHM6Ly93d3cuc3RhcnRwYWdlLmNvbS8&guce_referrer_sig=AQAAAJxltTxoz6hR3t-DqbMrBx7jfhWa5Q5z3TEGKqGUvOn0szniJ1CiQn1OGR_xwQwUrJm7eeOmLtY0tplGpmYs_fJBlsHAVYLM0oepAPMYkTsY3EWfCo-KUJRfQUaiMnKuFUkMKSAuebS9mXp0_dar6mPjwpesnRwuvGAy85OuJ0cs
#https://wertpapiere.ing.de/Investieren/Derivat/DE000JC7HPS6
call_strike_price = 464
call_break_even = 838.4
call_initial_invest = (2000-put_initial_invest)
call_premium_percentage = -3/100
#https://finance.yahoo.com/quote/TSLA/?guccounter=1&guce_referrer=aHR0cHM6Ly93d3cuc3RhcnRwYWdlLmNvbS8&guce_referrer_sig=AQAAAKuCgqTIHdmbgRIe0_qbYWjJRu_Mqzq-PEr8nFpS1gLGDfZ8pdByXN8GI8_8SVE78OpffhzQ66qM-4F_q9M7SkBj6jN84sNo5EdIdYijIo6_YDIm52b9M04Fz49Xb9ZHPYGz0v3V19H7kkCHto91QypDGUVAev_bO-KBQ1dktMt_
stock_initial_invest = 0



underlying = np.linspace(put_break_even * 0, call_break_even * 2, 100)

combined = call_value(underlying, call_strike_price, call_break_even, call_initial_invest, call_premium_percentage) + put_value(underlying, put_strike_price, put_break_even, put_initial_invest, put_premium_percentage) + (underlying/current_underlying-1)*stock_initial_invest


ax[1, 2].plot(underlying, call_value(underlying, call_strike_price, call_break_even, call_initial_invest, call_premium_percentage), label="Call", marker='', linestyle='-', markersize='2', color="g")
ax[1, 2].plot(underlying, put_value(underlying, put_strike_price, put_break_even, put_initial_invest, put_premium_percentage), label="Put", marker='', linestyle='-', markersize='2', color="r")
ax[1, 2].plot(underlying, combined, label="Combined", marker='', linestyle='-', markersize='2', color="black")
ax[1, 2].plot(underlying, np.zeros(underlying.shape) , marker='', linestyle='--', markersize='2', color="grey")
#ax[0, 0].plot(underlying, (underlying/current_underlying-1)*stock_initial_invest, label="Stocks", marker='', linestyle='-', markersize='2')

ax[1, 2].set_xlabel('Underlying')
ax[1, 2].set_ylabel('Return')
ax[1, 2].legend(loc="lower right")

plt.show()
















































