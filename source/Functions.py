# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 11:30:48 2020

@author: mathias chastan
"""
from math import pi, cos, sin
from random import random
import numpy as np
from scipy.spatial.distance import cdist, euclidean
from matplotlib.pyplot import scatter
from collections import Counter 

def point(h, k, r):
    theta = random() * 2 * pi
    return h + cos(theta) * r, k + sin(theta) * r

def geometric_median(X, eps=1e-5):
    y = np.mean(X, 0)

    while True:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros/r
            y1 = max(0, 1-rinv)*T + min(1, rinv)*y

        if euclidean(y, y1) < eps:
            return y1

        y = y1
               
def CalculateFakePositivePercent(pred,true):
    prec = 0
    size1 = 0
    for i in range(0,len(pred)):
        if pred[i] == 1:
            size1 = size1 + 1
            if pred[i] != true[i]:
                prec = prec + 1
    if size1 > 0:
        return prec / size1
    else :
        return 0        
    
def rand_jitter(arr):
    stdev = .01 * (max(arr) - min(arr))
    return arr + np.random.randn(len(arr)) * stdev

def jitter(x, y, s=20, c='b', marker='o', cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, verts=None, hold=None, **kwargs):
    return scatter(rand_jitter(x), rand_jitter(y), s=s, c=c, marker=marker, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, alpha=alpha, linewidths=linewidths, **kwargs)

  
def most_frequent(List): 
    occurence_count = Counter(List) 
    return occurence_count.most_common(1)[0][0]