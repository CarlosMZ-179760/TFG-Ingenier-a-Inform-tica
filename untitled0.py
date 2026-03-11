# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 12:57:58 2026

@author: TESTER
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy

p=0.4
n1=30
n2=90
x1=np.arange(1500,1000)
x2=np.arange(1500,1000)
def empirical_failure_rate_low_failures(x):
    return (scipy.stats.binom(x,n1,p))
def empirical_failure_rate_high_failures(x):
    return scipy.stats.binom(x,n2,p)

y1=np.array(scipy.stats.nbinom.pmf(x1,n1,p),dtype=np.longdouble)

y2=scipy.stats.nbinom.pmf(x2,n2,p)
print(y1)
#print(np.flip(xOffset))
#plt.plot(xOffset1,y1)
xinv1=np.array([n1/(x+n1) for x in x1])
xinv2=np.array([n2/(x+n2) for x in x2])
print(y1,xinv1)
#print(xinv)
#print(xinv1-p)
#plt.loglog(xinv1+1-p,y1)
#plt.loglog(xinv2+1-p,y2)
#plt.axvline(x=1, color="black")

meany1=(sum(xinv1*(y1-p)))
meany1sq=(sum(xinv1*((y1-p)**2)))
meany2=(sum(xinv2*(y2-p)))
meany2sq=(sum(xinv2*((y2-p)**2)))
var1=meany1sq-meany1**2
var2=meany2**2-meany2sq
print(var1,var2)


