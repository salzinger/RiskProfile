import matplotlib.pyplot as plt
import numpy as np
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

def call_value(underlying, call_strike_price, call_break_even, call_initial_invest, call_premium_percentage):
    return np.heaviside(underlying - call_strike_price, 1) * (underlying - call_strike_price) * \
           call_initial_invest * (1 + call_premium_percentage) / (call_break_even - call_strike_price) - call_initial_invest * (1 + call_premium_percentage)

def put_value(underlying, put_strike_price, put_break_even, put_initial_invest, put_premium_percentage):
    return np.heaviside(put_strike_price - underlying, 1) * (underlying - put_strike_price) * \
           put_initial_invest * (1 + put_premium_percentage) /(put_break_even - put_strike_price) - put_initial_invest * (1 + put_premium_percentage)

t = np.linspace(0, 14, 8)

#EURO Stoxx 50
current_underlying = 3540
#https://wertpapiere.ing.de/Investieren/Derivat/DE000VP7XK01
put_strike_price = 4200
put_break_even = 3157
put_initial_invest = 3000
put_premium_percentage = 11.5/100 *0
#https://wertpapiere.ing.de/Investieren/Derivat/CH0540336895
call_strike_price = 2200
call_break_even = 3421
call_initial_invest = 7000-put_initial_invest
call_premium_percentage = -4/100 *0

#NVIDIA
current_underlying = 533.3
#https://wertpapiere.ing.de/Investieren/Derivat/DE000HZ92279
put_strike_price = 200
put_break_even = 199.98
put_initial_invest = 1
put_premium_percentage = 62/100
#https://wertpapiere.ing.de/Investieren/Derivat/DE000HZ92279
call_strike_price = 275
call_break_even = 536.5
call_initial_invest = (4000-put_initial_invest)*0
call_premium_percentage = 0/100

#Alibaba
current_underlying = 227.85
#https://wertpapiere.ing.de/Investieren/Derivat/DE000JC57SA4
put_strike_price = 298
put_break_even = 231
put_initial_invest = 500
put_premium_percentage = 0.5/100
#https://finance.yahoo.com/quote/BABA/?guccounter=1&guce_referrer=aHR0cHM6Ly93d3cuc3RhcnRwYWdlLmNvbS8&guce_referrer_sig=AQAAAJxltTxoz6hR3t-DqbMrBx7jfhWa5Q5z3TEGKqGUvOn0szniJ1CiQn1OGR_xwQwUrJm7eeOmLtY0tplGpmYs_fJBlsHAVYLM0oepAPMYkTsY3EWfCo-KUJRfQUaiMnKuFUkMKSAuebS9mXp0_dar6mPjwpesnRwuvGAy85OuJ0cs
stock_initial_invest = 2000

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

double_bagger = combined - call_initial_invest * (1 + call_premium_percentage) - put_initial_invest * (1 + put_premium_percentage)

zero_crossings = np.where(np.diff(np.sign(combined)))[0]
bagger_crossings = np.where(np.diff(np.sign(double_bagger)))[0]
loss = 0
for loss_values in range(zero_crossings[0], zero_crossings[1]):
    loss += underlying[loss_values]

print('Loss Area: ', loss/(call_initial_invest * (1 + call_premium_percentage) + put_initial_invest * (1 + put_premium_percentage)))
#print("Double for %.2f: " % underlying[bagger_crossings[0]], "at %.2f percent" % ((underlying[bagger_crossings[0]]/current_underlying-1)*100) )
print("Break for %.2f: " % underlying[zero_crossings[0]], "at %.2f percent" % ((underlying[zero_crossings[0]]/current_underlying-1)*100) )
print("Max Loss: %.2f" % np.min(combined))
print("Break for %.2f: " % underlying[zero_crossings[1]], "at %.2f percent" % ((underlying[zero_crossings[1]]/current_underlying-1)*100) )
#print("Double for %.2f: " % underlying[bagger_crossings[1]], "at %.2f percent" % ((underlying[bagger_crossings[1]]/current_underlying-1)*100) )

fig, ax = plt.subplots(2, 2, figsize=(10, 10))

ax[0, 0].plot(underlying, call_value(underlying, call_strike_price, call_break_even, call_initial_invest, call_premium_percentage), label="Call", marker='', linestyle='-', markersize='2')
ax[0, 0].plot(underlying, put_value(underlying, put_strike_price, put_break_even, put_initial_invest, put_premium_percentage), label="Put", marker='', linestyle='-', markersize='2')
ax[0, 0].plot(underlying, combined, label="Combined", marker='', linestyle='-', markersize='2')
ax[0, 0].plot(underlying, np.zeros(underlying.shape) , marker='', linestyle='-', markersize='2')
#ax[0, 0].plot(underlying, (underlying/current_underlying-1)*stock_initial_invest, label="Stocks", marker='', linestyle='-', markersize='2')

ax[0, 0].set_xlabel('Underlying')
ax[0, 0].set_ylabel('Return')
ax[0, 0].legend(loc="lower right")


seed = 5
N = 99  # increments

# brownian increments
b = Brownian(seed, N)[1]

# brownian motion
W = Brownian(seed, N)[0]
W = np.insert(W, 0, 0.)  # W_0 = 0. for brownian motion

# brownian increments

plt.rcParams['figure.figsize'] = (10, 8)
xb = np.linspace(1, len(b), len(b))
ax[0, 1].plot(xb, b, label='Brownian Increments')

# brownian motion
xw = np.linspace(1, len(W), len(W))
ax[0, 1].plot(xw, W, label='Brownian Motion')

ax[0, 1].legend(loc="lower right")

start = "2016-01-01"
end = "2021-01-11"

df = quandl.get("WIKI/AMZN", start_date=start, end_date=end)
print(df)
adj_close = df['Adj. Close']
time = np.linspace(1, len(adj_close), len(adj_close))


ax[1, 0].set_ylabel('Stock Price, $')
ax[1, 0].set_xlabel('Trading Days')

# GBM Exact Solution
# Parameters
#
# So:     initial stock price
# mu:     returns (drift coefficient)
# sigma:  volatility (diffusion coefficient)
# W:      brownian motion
# T:      time period
# N:      number of increments

So = adj_close[460]
mu = 0.15
sigma = 0.4
seed = 5
N = 99
W = Brownian(seed, N)[0]
T = 100
soln = GBM(So, mu, sigma, W, T, N)[0]    # Exact solution
t = GBM(So, mu, sigma, W, T, N)[1]       # time increments for  plotting
ax[1, 1].plot(t, soln, label="GBM mu "+str(mu)+" sigma "+str(sigma)+" and seed "+str(seed))
ax[1, 1].set_ylabel('Stock Price, $')

mu = 0.15
sigma = 0.4
seed = 22
W = Brownian(seed, N)[0]
soln1 = GBM(So, mu, sigma, W, T, N)[0]    # Exact solution
t = GBM(So, mu, sigma, W, T, N)[1]       # time increments for  plotting
ax[1, 1].plot(t, soln1, label="GBM mu "+str(mu)+" sigma "+str(sigma)+" and seed "+str(seed))

returns = daily_return(adj_close, 0, 460)
mu = np.mean(returns)*460.           # drift coefficient
sig = np.std(returns)*np.sqrt(460.)  # diffusion coefficient
print(mu, sig)
seed = 22
W = Brownian(seed, N)[0]
soln2 = GBM(So, mu, sigma, W, T, N)[0]    # Exact solution
t = GBM(So, mu, sigma, W, T, N)[1]       # time increments for  plotting
ax[1, 1].plot(t, soln2, label="GBM,adjusted mu "+str(round(mu, 2))+" sigma "+str(sigma)+" and seed "+str(seed))
ax[1, 1].plot(t, adj_close[460:len(adj_close)], label="Actual Stock")
ax[1, 1].set_xlabel('Time')
ax[1, 1].set_ylabel('Stock Prediction, $')
ax[1, 1].legend(loc="upper left")

mu = 0.15
sigma = 0.4
seed = 5
ax[1, 0].plot(time[460:len(adj_close)], soln1, label="GBM mu "+str(mu)+" sigma "+str(sigma)+" and seed "+str(seed))

mu = 0.15
sigma = 0.4
seed = 22
ax[1, 0].plot(time[460:len(adj_close)], soln, label="GBM mu "+str(mu)+" sigma "+str(sigma)+" and seed "+str(seed))
ax[1, 0].set_ylabel('Stock Price, $')
ax[1, 0].set_xlabel('Trading Days')


mu = np.mean(returns)*460.           # drift coefficient
sigma = np.std(returns)*np.sqrt(460.)  # diffusion coefficient
seed=22
ax[1, 0].plot(time[460:len(adj_close)], soln2, label="GBM,adjusted mu "+str(round(mu, 2))+" sigma "+str(sigma)+" and seed "+str(seed))

ax[1, 0].plot(time, adj_close, label='Actual stock')
ax[1, 0].legend(loc="upper left")
plt.show()







