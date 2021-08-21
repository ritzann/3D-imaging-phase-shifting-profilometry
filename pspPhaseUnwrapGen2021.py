# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 10:50:43 2017

@author: Galaxea
"""

from glob import glob
import os
import numpy as np
from skimage.restoration import unwrap_phase
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt


def rgb2gray(rgb):
    """
    Converts an RGB image to grayscale image.
    """

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray
    
def wrapPhase(iscrop, ref='O'):
    if iscrop:
        crop = '_crop'
    else:
        crop = ''
    try:
#        I = [(plt.imread("%s%s/%s%d%s.TIF"%(ref, crop, ref, i, crop))) for i in range(1,5)]
#        I = [rgb2gray(plt.imread("%s%s/%s%d%s.TIF"%(ref, crop, ref, i, crop))) for i in range(1,5)]
        I = [(plt.imread("%s%s/%s%d%s.TIF"%(ref, crop, ref, i, crop)))/255. for i in range(1,5)]
    except:
#        I = [(plt.imread("%s%s/%s%d%s.tif"%(ref, crop, ref, i, crop))) for i in range(1,5)]
#        I = [rgb2gray(plt.imread("%s%s/%s%d%s.tif"%(ref, crop, ref, i, crop))) for i in range(1,5)]
        I = [(plt.imread("%s%s/%s%d%s.tif"%(ref, crop, ref, i, crop)))/255. for i in range(1,5)]
    wrapped = np.arctan2((I[3] - I[1]), (I[0] - I[2]))
    
    return wrapped

def removeRamp(unwrapped):
    """
    Removes ramp of unwrapped phase map.
    """

    step = (np.max(unwrapped)-np.min(unwrapped))/unwrapped.shape[1]  
    rampX = np.arange(np.min(unwrapped), np.max(unwrapped), step)
    ramp = np.tile(rampX[np.arange(unwrapped.shape[1])],[unwrapped.shape[0],1]) 
    unwrappedNoramp = unwrapped - ramp
     
    return unwrappedNoramp

def PSP(dirName, iscrop):
    """
    Performs phase-shifting profilometry (PSP)
    """
    
    os.chdir(dirName)
    
    wrappedR = wrapPhase(iscrop, ref='R')
    print ("Saving reference wrapped phase... ")
    savemat('wrappedR.mat', {'wrappedR':wrappedR})
    print("Unwrapping reference wrapped phase...")
    unwrappedR = unwrap_phase(wrappedR)
    savemat('unwrappedR.mat', {'unwrappedR':unwrappedR})   
    unwrappedRnr = removeRamp(unwrappedR)
    print("Saving reference unwrapped phase...")
    savemat('unwrappedRnr.mat', {'unwrappedRnr':unwrappedRnr})
    
    wrappedO = wrapPhase(iscrop, ref='O')
    print ("Saving object wrapped phase... ")
    savemat('wrappedO.mat', {'wrappedO':wrappedO})
    print("Unwrapping object wrapped phase...")
    unwrappedO = unwrap_phase(wrappedO)
    savemat('unwrappedO.mat', {'unwrappedO':unwrappedO})
    unwrappedOnr = removeRamp(unwrappedO)
    print("Saving object unwrapped phase...")
    savemat('unwrappedOnr.mat', {'unwrappedOnr':unwrappedOnr})  
    
    final = unwrappedOnr - unwrappedRnr

    plt.imshow(final)

    return final

def plotPSP(path):
    """
    Plot wrapped and unwrapped phases from unprocessed and processed images.
    """    
    os.chdir(path)
    wrapped = loadmat('wrappedpy.mat')['wrappedpy']
    unwrapped = loadmat('unwrappedpy.mat')['unwrappedpy']
    rwrapped = loadmat('rwrappedpy.mat')['rwrappedpy']
    runwrapped = loadmat('runwrappedpy.mat')['runwrappedpy']
    finalunwrapped = loadmat('finalunwrappedpy.mat')['finalunwrappedpy']
    
    fig, ax = plt.subplots(2, 2) 
    ax1, ax2, ax3, ax4 = ax.ravel()
    im = []
    cax = []
    for i, axes in enumerate(ax.ravel()):
        axes.set_yticks([])
        axes.set_xticks([])
#        if i in [1,3]:
        divider = make_axes_locatable(axes)
        cax.append(divider.append_axes("right", size="5%", pad=0.05))

    im.append(ax1.imshow(rwrapped, cmap='gray', vmin=-np.pi, vmax=np.pi))
    im.append(ax2.imshow(wrapped, cmap='gray', vmin=-np.pi, vmax=np.pi))
    im.append(ax3.imshow(runwrapped, cmap='jet', vmin=0, vmax=np.max(unwrapped))) 
    im.append(ax4.imshow(unwrapped, cmap='jet', vmin=0, vmax=np.max(unwrapped)))  

    for i in range(4):
        cbar = fig.colorbar(im[i], cax=cax[i])
        if i in [0,1]:
            cbar.set_ticks([-np.pi, 0, np.pi])
            cbar.set_ticklabels([r"$-\pi$", "$0$", r"$\pi$"])
           
    fig.subplots_adjust(wspace = -0.2)
    fig.subplots_adjust(hspace = 0.1)
           
    # fig.savefig('turtle_phases.pdf', bbox_inches='tight', pad_inches=0)

if __name__ == "__main__": 
    
#    dirName = r"C:\Users\Galaxea\Documents\Ritz\PSP 2017\03.21.2017 PSP\basler_acerwhite"
    # dirName = r"H:\MTF_PSP 20.10.17"
    dirName = r"/Users/ritzann/Downloads/psp"
    
    os.chdir(dirName)
    
    fs = range(60,130,10)
    fs = 60
    trials = range(1,3)
    # for f in fs:
    # os.chdir('f%s'%f)
    os.chdir('f60')
    for trial in trials:
        final = PSP('trial%s'%trial,0)
        savemat('final.mat', {'final':final})
        os.chdir('..')
    os.chdir('..')

    plt.imshow(final)