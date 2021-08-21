# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 02:55:53 2017

@author: Ritz Ann Aguilar
"""


from __future__ import division
from glob import glob
import os
import matplotlib as mpl
import numpy as np
import os, glob
from pylab import ginput
from matplotlib import pyplot as plt
from skimage.restoration import unwrap_phase
from scipy.io import loadmat, savemat
from scipy.optimize import curve_fit
from scipy.misc import imsave
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.fft import *


#mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
#mpl.rc('text', usetex=True)

mpl.rc('font',**{'family':'serif'})
mpl.rc('text', usetex=True)


def crop(img, x1, y1, width, height):
    """
    Crops the image with the given x and y coordinates.
    """
    x1 = np.floor(x1)
    y1 = np.floor(y1)
    width = np.floor(width)
    height = np.floor(height)    
    
    return img[y1:y1+height, x1:x1+width]

def plotHisto(im,bins):
    hist, bin_edges = np.histogram(im.flatten(), normed=True, bins=100)  
    ax.bar(bin_edges[:-1],hist/np.sum(hist),width=bin_edges[1]-bin_edges[0],color='gray', alpha=0.5, label='DBS')
    ax.set_xlabel('phase value')
    ax.set_ylabel('frequency')
    plt.grid()
    plt.legend()
    plt.savefig('stonehisto.pdf', bbox_inches='tight', pad_inches=0)    
    
def std(im):
    """
    Retrieves the std (phase error) of an image.
    """
    std_val = np.std(im)
    
    return std_val


if __name__ == "__main__":
    
#    dirName = r"H:\Thesis 2017\PSP 2017\basler_white\03.21.2017 PSP\basler_acerwhite"
    dirName = r"H:\MTF_PSP 20.10.17"

    os.chdir(dirName)
    
    trials = range(1,6)
    f_vals = range(60,130,10)
    
    steps_rect = loadmat('steps_rect.mat')['steps_rect'][0]
#    spyr_rect = loadmat('spyr_rect.mat')['spyr_rect'][0]
#    bpyr_rect = loadmat('bpyr_rect.mat')['bpyr_rect'][0]
#    whole_rect = loadmat('whole_rect.mat')['whole_rect'][0]    
    
    fig = plt.figure()
    ax = fig.add_subplot(111)    
    
#    for i, f in enumerate(f_vals[::3]):
    for i, f in enumerate(f_vals):
        os.chdir('f%s'%f)
        bpyr = 0
        spyr = 0
        steps = 0
        whole = 0
        final = 0
#        for trial in [1]:
        for trial in trials:
            os.chdir('trial%s'%trial)
            final = loadmat('final.mat')['final']
            steps += crop(final,steps_rect[0],steps_rect[1],steps_rect[2],steps_rect[3])
#            bpyr += crop(final,bpyr_rect[0],bpyr_rect[1],bpyr_rect[2],bpyr_rect[3])
#            spyr += crop(final,spyr_rect[0],spyr_rect[1],spyr_rect[2],spyr_rect[3])
#            whole +=  crop(final,whole_rect[0],whole_rect[1],whole_rect[2],whole_rect[3])  
            os.chdir('..')
#        bpyr = bpyr/len(trials)
#        spyr = spyr/len(trials)
        steps = steps/len(trials)
#        whole = whole/len(trials)
#        final = final/len(trials)
#        bpyr_scan = bpyr[:,np.shape(bpyr)[1]/2]
#        spyr_scan = spyr[:,np.shape(spyr)[1]/2]
        steps_scan = steps[:,np.shape(steps)[1]/2]
        savemat('steps.mat', {'steps':steps})
#        savemat('spyr.mat', {'spyr':spyr})
#        savemat('bpyr.mat', {'bpyr':bpyr})
#        savemat('whole.mat', {'whole':whole})
#        savemat('final.mat', {'final':final})
        savemat('steps_scan.mat', {'steps_scan':steps_scan})
#        savemat('spyr_scan.mat', {'spyr_scan':spyr_scan})
#        savemat('bpyr_scan.mat', {'bpyr_scan':bpyr_scan})
        os.chdir('..')
        ax.plot(steps_scan-steps_scan[0]/max(steps_scan-steps_scan[0]),label=r'$f=%s$'%f, linewidth=2*(i//5 +1))
#        ax.plot((bpyr_scan-bpyr_scan[0])/max(bpyr_scan-bpyr_scan[0]),label=r'$f=%s$'%f)

    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
    ax.set_ylabel('Phase (rad)', fontsize = 13)
    ax.set_xlabel('Pixel', fontsize = 13)
#    fig.savefig('bpyr_scan.pdf', pad_inches=0)
    fig.savefig('steps_scan.pdf', pad_inches=0)

