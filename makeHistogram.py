# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 10:49:32 2017

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
from scipy.misc import imsave
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats
from matplotlib import rc, cm
import scipy.ndimage
import scipy.misc

rc('font',**{'family':'serif'})
rc('text', usetex=True)

def plotHisto(im,bins,f):    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    hist, bin_edges = np.histogram(im.flatten(), bins, density=True)  
    ax.bar(bin_edges[:-1],hist/np.sum(hist),width=bin_edges[1]-bin_edges[0],
           color='gray', alpha=0.5, label=r'$f_o=%s$'%f)    
    ax.set_xlabel('Phase Value',fontsize=13)
    ax.set_ylabel('Probaility Density',fontsize=13)
    ax.set_xlim(-0.12,0.35)
    bbox_props = dict(fc="w", ec="0.5", alpha=0.9)
    ax1 = plt.gca()      
    plt.text(0.98, 0.96, r"$f_o = %s$" % f,  ha="right", va="top",
            size=12, bbox=bbox_props, transform=ax1.transAxes)
    plt.grid()
#    plt.legend()
    path = r"H:/Thesis 2017/2444776hsshdk/figures/histograms/"
    plt.savefig(path + 'hist%s.pdf'%f, bbox_inches='tight', pad_inches=0)
    
    return hist, bin_edges
    
#def imstd(im):
#    """
#    Retrieves the standard deviation (phase error) of an image.
#    """
#    std_val = np.std(im)
#    return std_val
    
    
def improfile(im,x0,y0,x1,y1,num):

    #-- Extract the line...
    # Make a line with "num" points...
    x, y = np.linspace(x0, x1, num), np.linspace(y0, y1, num)
    
    # Extract the values along the line, using cubic interpolation
    zi = scipy.ndimage.map_coordinates(np.transpose(im), np.vstack((x,y)))
    
    return zi    
    
def crop(img, x1, y1, width, height):
    """
    Crops the image with the given x and y coordinates.
    """
    x1 = np.floor(x1)
    y1 = np.floor(y1)
    width = np.floor(width)
    height = np.floor(height)    
    
    return img[y1:y1+height, x1:x1+width]
    
def plotRefMod(f_vals,R_mods,path):   
    """
    Plots modulation of the output sinusoidal fringe patterns
    as a function of spatial freqency.
    """
    plt.figure()
    slope, intercept, r_value, p_value, std_err = stats.linregress(f_vals,R_mods)
    plt.scatter(f_vals,R_mods,color='k',alpha=0.7,s=50)
#    plt.errorbar(f_vals,R_mods,std_err)
    z = np.polyfit(f_vals, R_mods, 1)
    p = np.poly1d(z)
    plt.plot(f_vals,p(f_vals),'k')
    plt.text(0.41, 0.69, r'$y = %.4fx + %.1f$' % (z[0],z[1]),
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax.transAxes, fontsize=13)#, bbox={'facecolor':'white','pad':5})
    plt.text(0.41, 0.62, r'$R^2 = %.3f$' % (r_value**2),
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax.transAxes, fontsize=13)#, bbox={'facecolor':'white','pad':5})
    plt.xlabel(r'$f_o$ (cy/fr)', fontsize = 13)
    plt.ylabel('Contrast', fontsize = 13)
    plt.grid()
    plt.savefig(path+'RefModv2.pdf', bbox_inches='tight', dpi=300)
    
    return r_value

def plotRelativeSpread(f_vals,stdvals,path):
    """
    Plots relative spread of phase values of the reconstructed reference plane
    as a function of spatial freqency.
    """
    plt.figure()
    plt.scatter(f_vals,stdvals,color='k',alpha=0.7,s=50)
    plt.xlabel(r'$f_o$ (cy/fr)', fontsize = 13)
    plt.ylabel(r'Standard Deviation $\sigma$', fontsize = 13)
    plt.grid()
    plt.savefig(path+'RSvsfreqv3.pdf', bbox_inches='tight', dpi=300)

if __name__ == "__main__":
    
    dirName = r"H:\Thesis 2017\PSP 2017\basler_white\03.21.2017 PSP\basler_acerwhite"
    os.chdir(dirName)
    
    trials = range(1,6)
    f_vals = range(60,150,10)
#    f_vals = [60,90,140]
#    f_vals = [60,140]
#    f_vals = [100,130]
#    color = ['k','g','b']
    color = ['k','r']
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
#    color = lambda level, n: "%s" % (level*(0.4/(n-1)))
    linestyle =['--','-']
#    linewidth = [1,2]
    
    stdvals = []
    R_mods = []
    
    for i, f in enumerate(f_vals[::]):
        os.chdir('f%s'%f)
        
        final = loadmat('final.mat')['final']
        ref = crop(final,49.5100,202.5100,297.98-49.51,730.98-202.51)
        savemat('ref.mat', {'ref':ref})
        
        bins = 50
#        hist, bin_edges = plotHisto(ref,bins,f)
#        std = imstd(ref - np.abs(np.mean(ref)))/np.abs(np.mean(ref))
#        stdvals.append(std)
#        stdvals.append(np.std(ref)/np.abs(np.mean(ref)))
        stdvals.append(np.std(ref))
        
        for trial in [1]:
            os.chdir('trial%s'%trial)            
            os.chdir('R')
            R = plt.imread('R1.tif')/255.
            
            x0, y0 = 236, 520
            x1, y1 = 1002, 516
            num = 500
            R_prof = improfile(R,x0,y0,x1,y1,num)
            R_mod = (np.max(R_prof) - np.min(R_prof))/(np.max(R_prof) + np.min(R_prof))
            R_mods.append(R_mod)            
            
            ax.plot(R_prof,label=r'$f_o=%s$'%f,
#                    color=color(i,3),
                    color=color[i%len(color)],
#                    linestyle=linestyle[i%len(linestyle)],
#                    linewidth=linewidth[i%len(linewidth)])
                    linewidth=2*(i//5 +1))
            
            os.chdir('../..')    
        os.chdir('..')

    path = r"H:/Thesis 2017/2444776hsshdk/figures/"
#    plotRelativeSpread(f_vals,stdvals,path)
    plotRefMod(f_vals,R_mods,path)
    
#    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
#    ax.legend(loc='upper right')
#    ax.set_ylabel('Intensity', fontsize = 13)
#    ax.set_xlabel('Pixel', fontsize = 13)
#    ax.set_xlim(0,180)
#    ax.set_ylim(0,.7)
#    fig.savefig(path+'sine%d_%d.pdf' %(f_vals[0],f_vals[1]), pad_inches=0)


#    r_value = plotRefMod(f_vals,R_mods,path)
#    print r_value
#    plotRelativeSpread(f_vals,stdvals,path)