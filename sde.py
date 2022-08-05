import sdepy
from sdepy import wiener_source, kfunc
import numpy as np
import matplotlib.pyplot as plt

dw = kfunc(wiener_source)
my_instance = dw(paths=100, dtype=np.float32)
x = my_instance(t=0, dt=1)

def my_process(t, x, theta=1., k=1., sigma=1.):
    return {'dt': k*(theta - x), 'dw': sigma}

#myp = kfunc(my_process)
coarse_timeline = (0., 0.25, 0.5, 0.75, 1.0)
timeline = np.linspace(0., 1., 500)
#x = my_instance(x0=1, paths=100*1000, steps=100)(coarse_timeline)

#gr = plt.plot(timeline, x[:, :30])


from numpy import exp, sin, sqrt
from scipy.special import erf
from scipy.integrate import quad

np.random.seed(1)
k = .5
x0, x1 = 0, 10;
t0, t1 = 0, 1
lb, hb = 4, 6

def green_exact(y, s, x, t):
     return exp(-(x - y)**2/(4*k*(t - s)))/sqrt(4*np.pi*k*(t - s))

def u1_exact(x, t):
    return (erf((x - lb)/2/sqrt(k*(t - t0))) - erf((x - hb)/2/sqrt(k*(t - t0))))/2

def u2_exact(x, t):
    return exp(-k*(t - t0))*sin(x)

xgrid = np.linspace(x0, x1, 51)
tgrid = np.linspace(t0, t1, 5)
xp = sdepy.wiener_process(
    paths=10000, steps=100,
    sigma=np.sqrt(2*k),
    vshape=xgrid.shape, x0=xgrid[..., np.newaxis],
    i0=-1,
    )(timeline=tgrid)

a = sdepy.montecarlo(xp, bins=100)

def green(y, i, j):
    """green function from (y=y, s=tgrid[i]) to (x=xgrid[j], t=t1)"""
    return a[i, j].pdf(y)

u1, u2 = np.empty(51), np.empty(51)
for j in range(51):
    u1[j] = quad(lambda y: green(y, 0, j), lb, hb)[0]
    u2[j] = quad(lambda y: sin(y)*green(y, 0, j), -np.inf, np.inf)[0]

y = np.linspace(x0, x1, 500)
for i, j in ((1, 20), (2, 30), (3, 40)):
    gr = plt.plot(y, green(y, i, j),
                  y, green_exact(y, tgrid[i], xgrid[j], t1), ':')
plt.show()

gr = plt.plot(xgrid, u1, y, u1_exact(y, t1), ':')
gr = plt.plot(xgrid, u2, y, u2_exact(y, t1), ':')
plt.show()

print('u1 error: {:.2e}\nu2 error: {:.2e}'.format(
    np.abs(u1 - u1_exact(xgrid, t1)).mean(),
    np.abs(u2 - u2_exact(xgrid, t1)).mean()))



plt.show()
