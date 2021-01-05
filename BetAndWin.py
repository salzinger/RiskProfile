import matplotlib.pyplot as plt
import numpy as np
import mibian

c = mibian.GK([1.4565, 1.45, 1, 2, 30], volatility=20)

#print(c.callPrice)


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
call_initial_invest = 4000-put_initial_invest
call_premium_percentage = 0.8/100


#Alibaba
current_underlying = 227.85
#https://wertpapiere.ing.de/Investieren/Derivat/DE000JC57SA4
put_strike_price = 298
put_break_even = 231
put_initial_invest = 500
put_premium_percentage = 0.5/100

#https://finance.yahoo.com/quote/BABA/?guccounter=1&guce_referrer=aHR0cHM6Ly93d3cuc3RhcnRwYWdlLmNvbS8&guce_referrer_sig=AQAAAJxltTxoz6hR3t-DqbMrBx7jfhWa5Q5z3TEGKqGUvOn0szniJ1CiQn1OGR_xwQwUrJm7eeOmLtY0tplGpmYs_fJBlsHAVYLM0oepAPMYkTsY3EWfCo-KUJRfQUaiMnKuFUkMKSAuebS9mXp0_dar6mPjwpesnRwuvGAy85OuJ0cs

stock_initial_invest = 2000-put_initial_invest






underlying = np.linspace(put_break_even * 0, call_break_even * 2, 100)

combined = call_value(underlying, call_strike_price, call_break_even, call_initial_invest, call_premium_percentage) + put_value(underlying, put_strike_price, put_break_even, put_initial_invest, put_premium_percentage) + (underlying/current_underlying-1)*stock_initial_invest

double_bagger = combined - call_initial_invest * (1 + call_premium_percentage) - put_initial_invest * (1 + put_premium_percentage)

zero_crossings = np.where(np.diff(np.sign(combined)))[0]
bagger_crossings = np.where(np.diff(np.sign(double_bagger)))[0]
loss = 0
#for loss_values in range(zero_crossings[0], zero_crossings[1]):
#    loss += underlying[loss_values]

#print('Loss Area: ', loss/(call_initial_invest * (1 + call_premium_percentage) + put_initial_invest * (1 + put_premium_percentage)))
#print("Double for %.2f: " % underlying[bagger_crossings[0]], "at %.2f percent" % ((underlying[bagger_crossings[0]]/current_underlying-1)*100) )
#print("Break for %.2f: " % underlying[zero_crossings[0]], "at %.2f percent" % ((underlying[zero_crossings[0]]/current_underlying-1)*100) )
#print("Max Loss: %.2f" % np.min(combined))
#print("Break for %.2f: " % underlying[zero_crossings[1]], "at %.2f percent" % ((underlying[zero_crossings[1]]/current_underlying-1)*100) )
#print("Double for %.2f: " % underlying[bagger_crossings[1]], "at %.2f percent" % ((underlying[bagger_crossings[1]]/current_underlying-1)*100) )

fig, ax = plt.subplots()

ax.plot(underlying, call_value(underlying, call_strike_price, call_break_even, call_initial_invest, call_premium_percentage), label="Call", marker='', linestyle='-', markersize='2')
ax.plot(underlying, put_value(underlying, put_strike_price, put_break_even, put_initial_invest, put_premium_percentage), label="Put", marker='', linestyle='-', markersize='2')
ax.plot(underlying, combined, label="Combined", marker='', linestyle='-', markersize='2')
ax.plot(underlying, np.zeros(underlying.shape) , marker='', linestyle='-', markersize='2')
ax.plot(underlying, (underlying/current_underlying-1)*stock_initial_invest, label="Stocks", marker='', linestyle='-', markersize='2')

# ax.plot(times, result.expect[1],label="MagnetizationZ",linestyle='--',marker='o',markersize='2');
# ax.plot(times, result.expect[2],label="Exp(SigmaZ,0)");
# ax.plot(times, result.expect[3],label="Exp(SigmaX,0)",linestyle='--');
# ax.plot(times, np.abs(ups),label="Tr(rho_0,uu)",linestyle='--');
# ax.plot(times, np.abs(downs),label="Tr(rho_0,dd)",linestyle='-');
ax.set_xlabel('Underlying')
ax.set_ylabel('Return')
ax.legend(loc="lower right")

plt.show()
