#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import math
from scipy.stats import norm
import matplotlib.pyplot as plt

def MCsim(S,T,r,vol,N):
    dt = T/N
    global St
    St[0] = S
    for i in range(N):
        St[i+1] = St[i]*math.exp((r-0.5*vol*vol)*dt+np.random.normal()*vol*math.sqrt(dt))
    return St

def BLSprice(S,L,T,r,vol):
    d1 = (math.log(S/L)+(r+0.5*vol*vol)*T)/(vol*math.sqrt(T))
    d2 = d1-vol*math.sqrt(T)
    C = S*norm.cdf(d1)-L*math.exp(-r*T)*norm.cdf(d2)
    return C

def BTcall(S,T,r0,vol,N,L):
    dt = T/N
    u = math.exp(vol*math.sqrt(dt))
    d = math.exp(-vol*math.sqrt(dt))
    p = (math.exp(r0*dt)-d)/(u-d)
    priceT = np.zeros((N+1,N+1)) #price tree
    priceT[0][0] = S
    for c in range(N):
        priceT[0][c+1] = priceT[0][c]*u
        for r in range(c+1):
            priceT[r+1][c+1] = priceT[r][c]*d
    probT = np.zeros((N+1,N+1)) #probability tree
    probT[0][0] = 1
    for c in range(N):
        for r in range(c+1):
            probT[r][c+1] += probT[r][c]*p #往右推
            probT[r+1][c+1] += probT[r][c]*(1-p) #往右下推
    call = 0
    for r in range(N+1):
        if(priceT[r][N]>=L):
            call += (priceT[r][N]-L)*probT[r][N]
    return call*math.exp(-r0*T)

def BisectionBLS(S,L,T,r,call,tol): #tol:tolerance:容許誤差
    left = 0.00000000000001
    right = 1
    while(right-left>tol):
        middle = (right+left)/2
        if((BLSprice(S,L,T,r,middle)-call)*(BLSprice(S,L,T,r,left)-call)<0):
            right = middle
        else:
            left = middle
    return(left+right)/2
    
    
S = 50
L = 40
T = 2
r = 0.08
vol = 0.2
N = 100
St = np.zeros((N+1)) 
Sa = MCsim(S,T,r,vol,N)

M = 1000
call = 0
for i in range(M):
    Sa = MCsim(S,T,r,vol,N)
    #plt.plot(Sa)
    if(Sa[-1]-L>0):
        call += Sa[-1]-L

print('N =',N,' M =',M)   
print('MC =',call/M*math.exp(-r*T))
print('BL =',BLSprice(S,L,T,r,vol))
print('difference =',call/M*math.exp(-r*T)-BLSprice(S,L,T,r,vol))
#print('N =',N)
#print('BT =',BTcall(S,T,r,vol,N,L))
#print('BL =',BLSprice(S,L,T,r,vol))
#print('difference =',BTcall(S,T,r,vol,N,L)-BLSprice(S,L,T,r,vol))
S = 10889.96
T = 7/365
r = 1.065/100
L = [10700,10800,10900,11000,11100,11200,11300,11400]
call = {10700:214,10800:136,10900:69,11000:26,11100:7,11200:1.7,11300:0.9,11400:0.5}
b = np.zeros(8)
i=0
for l in L:
    b[i] = BisectionBLS(S,l,T,r,call[l],0.00001)
    i+=1
plt.plot(L,b,'-o')
plt.show()