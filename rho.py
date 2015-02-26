# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 13:40:55 2015

@author: hubert
"""

data1 = np.loadtxt('/home/hubert/Documents/MAP565/kalman/traj1.dat')
data2 = np.loadtxt('/home/hubert/Documents/MAP565/kalman/traj2.dat')
 
V = data1 - data2
 
Cov = np.Cov(V.T)

rho = Cov.trace()/2