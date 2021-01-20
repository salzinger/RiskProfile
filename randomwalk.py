#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 10:49:23 2020

@author: Oliver Drozdowski
"""


import numpy as np
from matplotlib import pyplot as plt

## standard random walk
##################################
N = 10000
n = 5000
## Random numbers give direction
random_numbers = np.random.randint(0, 4, (N,n))


steps = np.zeros((N,n+1,2))

## Depending on the value of the random number go in one direction
steps[:,1:n+1][random_numbers==0,:] = [-1,0]
steps[:,1:n+1][random_numbers==1,:] = [1,0]
steps[:,1:n+1][random_numbers==2,:] = [0,-1]
steps[:,1:n+1][random_numbers==3,:] = [0,1]

## Add up all direction sums
standard_walks = np.cumsum(steps, axis=1)


## self-avoiding random walks
##################################

N = 700000
n = 20
## Random numbers give direction
random_numbers = np.random.randint(0, 4, (N,n))


steps = np.zeros((N,n+1,2))

## Depending on the value of the random number go in one direction
steps[:,1:n+1][random_numbers==0,:] = [-1,0]
steps[:,1:n+1][random_numbers==1,:] = [1,0]
steps[:,1:n+1][random_numbers==2,:] = [0,-1]
steps[:,1:n+1][random_numbers==3,:] = [0,1]

## Add up all direction sums
self_avoiding_walks = np.cumsum(steps, axis=1)

## Now kick out walks with duplicates
## Create array of tuples
self_avoiding_walks_in_sets = [[tuple(entry) for entry in walk] for walk in self_avoiding_walks]
## If one position is duplicate in a walk, the set contains the position only once
self_avoiding_walks_without_duplicates = np.array([walk for walk in self_avoiding_walks_in_sets if len(set(walk)) == len(walk)])

## Plotting
##################################

fig = plt.figure(figsize=(5,6))
ax1 = fig.add_subplot(321)
ax1.scatter(standard_walks[:3,:,0].T, standard_walks[:3,:,1].T,s=0.05);
ax1.plot(standard_walks[:3,:,0].T, standard_walks[:3,:,1].T,lw=0.5,ls="-");
ax1.set_title("Standard RW")
ax1.set_ylabel(r"$y$")
ax1.set_xlabel(r"$x$")

ax2 = fig.add_subplot(322)
ax2.scatter(self_avoiding_walks_without_duplicates[:4,:,0].T, self_avoiding_walks_without_duplicates[:4,:,1].T,s=0.05);
ax2.plot(self_avoiding_walks_without_duplicates[:4,:,0].T, self_avoiding_walks_without_duplicates[:4,:,1].T,lw=0.5,ls="-");
ax2.set_title("Self-avoiding RW")
ax2.set_ylabel(r"$y$")
ax2.set_xlabel(r"$x$")

## Calculate squared displacement
standard_walks_means = np.sum(standard_walks**2, axis=2)
self_avoiding_walks_without_duplicates_means = np.sum(self_avoiding_walks_without_duplicates**2, axis=2)

## Calculate means
standard_walks_means = np.mean(standard_walks_means, axis=0)
self_avoiding_walks_without_duplicates_means = np.mean(self_avoiding_walks_without_duplicates_means, axis=0)

rw_n = np.arange(standard_walks_means.shape[0])
saw_n = np.arange(self_avoiding_walks_without_duplicates_means.shape[0])

ax3 = fig.add_subplot(323)
ax3.plot(rw_n, standard_walks_means, c="black");
ax3.set_ylabel(r"$<r^2>$")
ax3.set_xlabel(r"$n$")
ax3.set_title("N=" + str(standard_walks.shape[0]))

ax4 = fig.add_subplot(324)
ax4.plot(saw_n, self_avoiding_walks_without_duplicates_means, c="black");
ax4.set_ylabel(r"$<r^2>$")
ax4.set_xlabel(r"$n$")
ax4.set_title("N=" + str(self_avoiding_walks_without_duplicates.shape[0]))

ax5 = fig.add_subplot(325)
ax5.loglog(rw_n, rw_n**(3.0/2.0), label=r"$n^{3/2}$", c="grey", lw=2.0, ls=":")
ax5.loglog(rw_n, rw_n, label=r"$n$", c="red", lw=2.0, ls=":")
ax5.loglog(rw_n, standard_walks_means, c="black", label="simulation");
ax5.set_ylabel(r"$<r^2>$")
ax5.set_xlabel(r"$n$")
ax5.legend(loc="upper left", fontsize=6)

ax6 = fig.add_subplot(326)
ax6.loglog(saw_n, saw_n**(3.0/2.0), label=r"$n^{3/2}$", c="grey", lw=2.0, ls=":")
ax6.loglog(saw_n, saw_n, label=r"$n$", c="red", lw=2.0, ls=":")
ax6.loglog(saw_n, self_avoiding_walks_without_duplicates_means, c="black", label="simulation");
ax6.set_ylabel(r"$<r^2>$")
ax6.set_xlabel(r"$n$")
ax6.legend(loc="upper left", fontsize=6)

plt.tight_layout()
plt.show()
fig.savefig("bonus_exercise.png", dpi=400)
