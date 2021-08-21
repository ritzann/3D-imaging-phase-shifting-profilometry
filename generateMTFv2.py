from __future__ import division
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 21:32:51 2016

@author: Ritz Ann Aguilar

Modulation Transfer Function for Analysis of Camera Resolution

"""

from numpy.fft import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat, savemat
import os
from matplotlib import rc

rc('font',**{'family':'serif'})
rc('text', usetex=True)

def rgb2gray(rgb):
    """
    Converts an RGB image to grayscale image.
    """
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray
    

#path = r"H:\Thesis 2017\PSP 2017\fujinon_mini"
#path = r"H:\Thesis 2017\mtf2"
#path = r"H:\Thesis 2017\mtf3 (basler_acerblackmini)"
#path = r"H:\Thesis 2017\mtf4 (basler_acerbiggray)"
#path = r"H:\Thesis 2017\mtf5 (basler_acerwhite)"
#path = r"H:\Thesis 2017\PSP 2017\basler_white\03.21.2017 PSP\basler_acerwhite"
#path = r"C:\Users\Galaxea\Documents\Ritz\mtf basler"

path = r"H:\MTF_PSP 20.10.17"
os.chdir(path)

#im = plt.imread('mtf10c.tif')[:,:,0]/255.
#im = plt.imread('mtf_baslerwhite.tif')[:,:,0]/255.
#im = rgb2gray(plt.imread('mtf4_5.jpg'))/255.
#im = plt.imread('edge_C1.tif')[:,:,0]/255.
#im = plt.imread('edge_C1_P3.tif')[:,:,0]/255.
im = plt.imread('edge.tif')[:,:,0]/255.

#x = np.linspace(0,100,500)
#im = np.array([np.sin(x) for i in x])

im_mean = np.mean(im, axis=0)
lsf = np.diff(im,1,axis=1)
norm_lsf = lsf/np.max(lsf)
mtf = np.abs(fftshift(fft(norm_lsf, axis=1)))
mtf_mean = np.mean(mtf, axis=0)
norm_mtf = mtf_mean/np.max(mtf_mean)

fig = plt.figure(figsize=(16,4))
ax1 = fig.add_subplot(131)
ax1.plot(im[np.shape(im)[0]/2,:],linewidth=2,color="black")
#ax1.plot(im_mean,linewidth=2,color="black")
ax1.set_title('Edge Spread Function')
plt.ylabel('Signal Value')
plt.xlabel('Pixel')

lsf_mean = np.mean(lsf, axis=0)
nlsf = lsf_mean/np.max(lsf_mean)
ax2 = fig.add_subplot(132)
ax2.plot(norm_lsf[np.shape(norm_lsf)[0]/2,:],linewidth=2,color="black")
#ax2.plot(lsf[np.shape(lsf)[0]/2,:],linewidth=2,color="black")
#ax2.plot(nlsf,linewidth=2,color="black")
ax2.set_title('Line Spread Function')
ax2.set_ylabel('Normalized LSF',labelpad=0.7)
plt.xlabel('Pixel')

N = len(norm_mtf)
#dx = 10/(167 - 62) #mft3 (2)
#dx = 10/(190-87) #mtf4_1
#dx = 26/(159-33) #mtf_baslerwhite
#dx = 10/(177-67) #mtf_olyedmund
#dx = 10/(218-81) #mtf_fujiedmund
#dx = 10/(157-55) #mtf_baslerwhite mtf1c
#dx = 36/(144-72) #mtf_baslerwhite mtf10c

#dx = 15/(127-49) #mtf_baslermini mtf2c
#dx = 18/(122-50) #mtf_baslermini mtf5c
#dx = 19/(200-122)#mtf_baslermini mtf9c
#dx = 21/(154-85) #mtf_baslermini mtf13c

#dx = 19/(90-28) #mtf_baslergray mtf3c
#dx = 30/(90-47) #mtf_baslerwhite mtf1c
#df = 1/dx
#dx = 10/(241-106) #mtf basler (C1)
#dx = 16/(94-43) #mtf basler white (C1 P3) THESIS!!

dx = 24/(583-514) #MTF_PSP 20.10.17 (basler_acermini)

f_max = 1/(2*dx)
#freq = 2*np.pi*np.linspace(-f_max, f_max, N)
freq = np.linspace(-f_max, f_max, N)
#freq[:] = [i - 1 for i in freq]
#freq = np.arange(-f_max,f_max,1/(N*dx))

fig1 = plt.figure()
ax3 = fig1.add_subplot(111)
ax3.plot(freq, norm_mtf, linewidth = 2, color="black")
ax3.set_xlim(0, np.max(freq))
#ax3.set_xlim(0, 0.6)
ax3.set_title('Modulation Transfer Function')
plt.ylabel('MTF')
plt.grid()
plt.xlabel('Spatial Frequency (cy/mm)')
fig1.savefig('mtf1.pdf', dpi=300, bbox_inches='tight')
fig1.savefig('mtf1.png', dpi=300, bbox_inches='tight')

#
#fig1 = plt.figure(2)
#ax = fig1.add_subplot(111)
#ax.plot(freq, norm_mtf, marker='.', color="black")
#half_mtf = np.abs(norm_mtf - 0.5)
#e_mtf = np.abs(norm_mtf - np.exp(-1))
#mtf30 = np.abs(norm_mtf - 0.3)
#index = np.where(half_mtf == np.nanmin(half_mtf))[0][0]
#index1 = np.where(e_mtf == np.nanmin(e_mtf))[0][0]
#index2 = np.where(mtf30 == np.nanmin(mtf30))[0][0]
#ax.plot([0,np.abs(freq[index]),np.abs(freq[index])],
#         [norm_mtf[index],norm_mtf[index],0],
#            linestyle = '-', color="black")
##ax.plot([0,np.abs(freq[(N)-index]),np.abs(freq[(N)-index])],
##         [norm_mtf[index],norm_mtf[index],0], 
##            linestyle = '-', color="black")
#
#ax.plot([0,np.abs(freq[index1]),np.abs(freq[index1])],
#        [norm_mtf[index1],norm_mtf[index1],0], 
#            linestyle = '-', color="black")
##ax.plot([0,np.abs(freq[(N)-index1]),np.abs(freq[(N)-index1])],
##        [norm_mtf[index1],norm_mtf[index1],0], 
##            linestyle = '-', color="black")
#
##       
#ax.plot([0,np.abs(freq[index2]),np.abs(freq[index2])],
#          [norm_mtf[index2],norm_mtf[index2],0],
#            linestyle = '-', color="black")
##ax.plot([0,np.abs(freq[(N)-index2]),np.abs(freq[(N)-index2])],
##          [norm_mtf[index2],norm_mtf[index2],0],
##            linestyle = '-', color="black")
#
#
##ax.set_title('Modulation Transfer Function')
#ax.set_yticks([0,.3,np.exp(-1),0.5,1.0])
#ax.set_yticklabels([0,.3,r'$e^{-1}$',0.5,1.0])
#plt.ylabel('MTF')
#plt.xlabel('Spatial Frequency (cy/mm)')
#plt.grid()
#ax.set_xlim(0, np.max(freq))
#x_val = np.abs(freq[(N)-index]) + 0.38
#ax.annotate("%.2f cy/mm" % (freq[(N-1)-index]), xy=(freq[(N-1)-index], 0.5), 
#             xytext=(x_val, 0.55), arrowprops=dict(facecolor='black', shrink=0.05))
#ax.annotate("%.2f cy/mm" % np.abs(freq[(N-1)-index1]), xy=(np.abs(freq[(N-1)-index1]), np.exp(-1)), 
#             xytext=(x_val, 0.455), arrowprops=dict(facecolor='black', shrink=0.05))
#ax.annotate("%.2f cy/mm" % np.abs(freq[(N-1)-index2]), xy=(np.abs(freq[(N-1)-index2]), 0.3), 
#             xytext=(x_val, 0.375), arrowprops=dict(facecolor='black', shrink=0.05))
#             
##ax.annotate("%.2f cy/mm" % np.abs(freq[(N)-index]), xy=(np.abs(freq[(N)-index]), 0.5), 
##             xytext=(x_val, 0.53), arrowprops=dict(facecolor='black', shrink=0.05))
##ax.annotate("%.2f cy/mm" % np.abs(freq[(N-1)-index1]), xy=(np.abs(freq[(N-1)-index1]), np.exp(-1)), 
##             xytext=(x_val, 0.485), arrowprops=dict(facecolor='black', shrink=0.05))
##ax.annotate("%.2f cy/mm" % np.abs(freq[(N)-index2]), xy=(np.abs(freq[(N)-index2]), 0.3), 
##             xytext=(x_val, .385), arrowprops=dict(facecolor='black', shrink=0.05))
#

fig.savefig('mtf.pdf', dpi=300, bbox_inches='tight')
fig.savefig('mtf.jpg', dpi=300, bbox_inches='tight')

#dirName = r"H:/Thesis 2017/2444776hsshdk/figures/"
#fig.savefig(dirName + 'all_C1.pdf', dpi=300, bbox_inches='tight')
#fig.savefig(dirName + 'all_C1_P3.pdf', dpi=300, bbox_inches='tight')

#fig.savefig(dirName + 'all_basleractual.pdf', dpi=300, bbox_inches='tight')
#fig1.savefig(dirName + 'mtf_basleractual.pdf', dpi=300, bbox_inches='tight')
#fig.savefig(dirName + 'all_baslerwhite.pdf', dpi=300, bbox_inches='tight')
#fig1.savefig(dirName + 'mtf_baslerwhite.pdf', dpi=300, bbox_inches='tight')

#print "%s cy/mm" % np.abs(freq[(N-1)-index])
#print "%s cy/mm" % np.abs(freq[(N-1)-index1])
#print "%s cy/mm" % np.abs(freq[(N-1)-index2])

