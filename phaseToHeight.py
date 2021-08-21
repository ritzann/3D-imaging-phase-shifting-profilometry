# -*- coding: utf-8 -*-
"""
Created on Mon Nov 06 10:48:02 2017

@author: Galaxea
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from pylab import ginput
from scipy.io import loadmat, savemat
import os

def phaseToHeight(path):
    """
    Returns the actual height/depth of the object.
    """
    
    os.chdir(path)
    im = loadmat('final.mat')['final']
    N = 15
    plt.imshow(im, interpolation = None)
    vals = ginput(N)
    Xi_0 = [int(i[0]) for i in vals]
    Yi_0 = [int(i[1]) for i in vals]
    plt.close()

    ## Get C values (UC = 1)
    U = []
    for n in xrange(len(Xi_0)):
        xi_0 = Xi_0[n]
        yi_0 = Yi_0[n]
        Phi = im[yi_0,xi_0]
        U.append([-Phi, -xi_0, -xi_0*Phi, -yi_0, -yi_0*Phi])
    U = np.matrix(U)

    C = ((U.T*U).I)*U.T*np.matrix(np.ones((N,1)))
    
#    heights = [0,4.5,2,4.5,4,2,4,3.5,2,3.5,3,2,3,2.5,2,2.5,2.4,2,2.4,2.3,2,2.3,0]
#    widths = [2,2,2,1.5,1.5,2,1.5,1.5,2,1.5,1.5,2,1.5,1.5,2,1.5,1.5,2,1.5,1.5,2,2,2]    
    
    ## Get D values (WD = V)
    ## height in mm
#    z1 = 4.5
#    z2 = 4
#    z3 = 2.3
#    z4 = 2.0
    
    z1 = 5
    z2 = 10
    z3 = 15
    z4 = 20
    
    N = 6
    ## Select at least 6 points
    plt.imshow(im, interpolation = None)
    vals = ginput(N)
    Xi_1 = [int(i[0]) for i in vals]
    Yi_1 = [int(i[1]) for i in vals]
    plt.close()
    
    plt.imshow(im, interpolation = None)
    vals = ginput(N)
    Xi_2 = [int(i[0]) for i in vals]
    Yi_2 = [int(i[1]) for i in vals]
    plt.close()
    
    plt.imshow(im, interpolation = None)
    vals = ginput(N)
    Xi_3 = [int(i[0]) for i in vals]
    Yi_3 = [int(i[1]) for i in vals]    
    plt.close()
    
    plt.imshow(im)
    vals = ginput(N)
    Xi_4 = [int(i[0]) for i in vals]
    Yi_4 = [int(i[1]) for i in vals]
    plt.close()
    
#    plt.imshow(im)
#    vals = ginput(N)
#    Xi_5 = [int(i[0]) for i in vals]
#    Yi_5 = [int(i[1]) for i in vals]    
#    plt.close()
    
    Xi_n = [Xi_0, Xi_1, Xi_2, Xi_3, Xi_4]
    Yi_n = [Yi_0, Yi_1, Yi_2, Yi_3, Yi_4]
    
    W = []
    V = []
    X = np.array([Xi_1, Xi_2, Xi_3, Xi_4])
    Y = np.array([Yi_1, Yi_2, Yi_3, Yi_4,])
    z = np.array([z1, z2, z3, z4])
    
    for l in range(np.size(X, 0)):
        for n in range(np.size(X, 1)):
            zk = z[l]
            xi = X[l,n]
            yi = Y[l,n]
            Phi = im[yi, xi]
            W.append([zk, zk*Phi, zk*xi, zk*xi*Phi, zk*yi, zk*yi*Phi])
            V.append(1 + C[0,0]*Phi + (C[1,0] + C[2,0]*Phi)*xi + (C[3,0] + C[4,0]*Phi)*yi)
    W = np.matrix(W)
    V = np.matrix(V).T
    
    D = (W.T*W).I*W.T*V
    
    ## Compute for world height (zw)
    zw = np.zeros(np.shape(im))
    for xi in range(np.shape(im)[1]):
        for yi in range(np.shape(im)[0]):
            Phi = im[yi,xi]
            zw_num = 1 + C[0,0]*Phi + (C[1,0] + C[2,0]*Phi)*xi + (C[3,0] + C[4,0]*Phi)*yi
            zw_den = D[0,0] + D[1,0]*Phi + (D[2,0] + D[3,0]*Phi)*xi + (D[4,0] + D[5,0]*Phi)*yi
            zw[yi, xi] = zw_num/zw_den
            
    savemat('finalcon.mat', {'final':zw})
    savemat('finalcoords.mat',{'Xi_n':Xi_n,'Yi_n':Yi_n})
    
    return C, D
    
if __name__ == "__main__":

    path = r"H:\MTF_PSP 20.10.17\f90"    
    os.chdir(path)
    im = loadmat('final.mat')['final']
    C = loadmat('C.mat')['C']
    D = loadmat('D.mat')['D']

    zw = np.zeros(np.shape(im))
    for xi in range(np.shape(im)[1]):
        for yi in range(np.shape(im)[0]):
            Phi = im[yi,xi]
            zw_num = 1 + C[0,0]*Phi + (C[1,0] + C[2,0]*Phi)*xi + (C[3,0] + C[4,0]*Phi)*yi
            zw_den = D[0,0] + D[1,0]*Phi + (D[2,0] + D[3,0]*Phi)*xi + (D[4,0] + D[5,0]*Phi)*yi
            zw[yi, xi] = zw_num/zw_den
            
    savemat('finalcon.mat', {'finalcon':zw})