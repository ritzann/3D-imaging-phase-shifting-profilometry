# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 16:34:13 2017

@author: Ritz Ann Aguilar
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os
import matplotlib as mpl
from matplotlib import rc

mpl.rc('font',**{'family':'serif'})
mpl.rc('text', usetex=True)

#path = r"H:\Thesis 2017\PSP 2017"
path = r"C:\Users\Galaxea\Documents\Ritz\mtf basler"
os.chdir(path)

freq= loadmat('freq.mat')['freq']
mtf = loadmat('mtf.mat')['mtf']
freq_o = freq[:,0]
freq_f = freq[:78,1]

mtf_o = mtf[:,0]
mtf_f = mtf[:78,1]

half_mtf1 = np.abs(mtf_o - 0.5)
index1 = np.where(half_mtf1 == np.nanmin(half_mtf1))[0][0]

half_mtf2 = np.abs(mtf_f - 0.5)
index2 = np.where(half_mtf2 == np.nanmin(half_mtf2))[0][0]

fig = plt.figure()
ax = fig.add_subplot(111)    
ax.plot(freq_o, mtf_o, linewidth = 2, color="black", label = 'C1')
ax.plot(freq_f, mtf_f, linewidth = 2, color="black", label = r'C1 \& P3', linestyle = '--')
ax.plot([0,np.abs(freq_o[index1]),np.abs(freq_o[index1])],
         [mtf_o[index1],mtf_o[index1],0],
            linestyle = '-', color="black")
ax.plot([0,np.abs(freq_f[index2]),np.abs(freq_f[index2])],
         [mtf_f[index2],mtf_f[index2],0],
            linestyle = '-', color="black")
ax.legend(loc='upper right', fontsize = 13)
ax.set_ylabel('MTF', fontsize = 13)
ax.set_xlabel('Spatial Frequency (cy/mm)', fontsize = 13)
ax.set_xlim(0, 1)
plt.grid()
#fig.savefig('mtfc2c3.pdf', pad_inches=0)
fig.savefig('mtfC1_C1P3.jpg', dpi=300, bbox_inches='tight')

dirName = r"H:/Thesis 2017/2444776hsshdk/figures/"
fig.savefig(dirName + 'mtfC1_C1P3v2.pdf', dpi=300, bbox_inches='tight')
fig.savefig(dirName + 'mtfC1_C1P3v2.jpg', dpi=300, bbox_inches='tight')