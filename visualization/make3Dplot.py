# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 19:05:43 2017

@author: Ritz Ann Aguilar
"""
from __future__ import division
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os
from scipy.io import loadmat, savemat
import numpy as np
from matplotlib import rc, cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
#import seaborn as sns

rc('font',**{'family':'serif'})
rc('text', usetex=True)

def crop(img, x1, y1, width, height):
    """
    Crops the image with the given x and y coordinates.
    """
    x1 = np.floor(x1)
    y1 = np.floor(y1)
    width = np.floor(width)
    height = np.floor(height)    
    
    return img[y1:y1+height, x1:x1+width]

def set_aspect_equal_3d(ax):
    """Fix equal aspect bug for 3D plots."""

    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()

    from numpy import mean
    xmean = mean(xlim)
    ymean = mean(ylim)
    zmean = mean(zlim)

    plot_radius = max([abs(lim - mean_)
                       for lims, mean_ in ((xlim, xmean),
                                           (ylim, ymean),
                                           (zlim, zmean))
                       for lim in lims])

    ax.set_xlim3d([xmean - plot_radius, xmean + plot_radius])
    ax.set_ylim3d([ymean - plot_radius, ymean + plot_radius])
    ax.set_zlim3d([zmean - plot_radius, zmean + plot_radius])


if __name__=="__main__":

#    dirName = r"H:\Thesis 2017\PSP 2017\basler_white\03.21.2017 PSP\basler_acerwhite"
    dirName = r"H:\MTF_PSP 20.10.17"
    os.chdir(dirName)
    
#    f_vals = range(60,150,10)
#    f_vals = range(70,150,10)
#    f_vals = [90,100,140]
#    calibx = 15/47
#    caliby = 36/114
#    calib = 60/(279-88)
    
#    f_vals = range(60,130,10)
    f_vals = range(90,100,10)
    calib = 15/57.5
    
    for i,f in enumerate(f_vals[:]):
        os.chdir('f%s'%f)
        try:
            steps = loadmat('stepscon.mat')['stepscon']
        except:
            steps = loadmat('stepscon.mat')['zw']
            
#        bpyr = loadmat('bpyr.mat')['bpyr']
#        bpyr_scan = loadmat('bpyr_scan.mat')['bpyr_scan']
#        calibz = 25/(np.max(bpyr_scan)-np.min(bpyr_scan))
#        bpyrcon = bpyr*calibz
#        savemat('bpyrcon.mat',{'bpyrcon':bpyrcon})
                
        sx = np.shape(steps)[0]
        sy = np.shape(steps)[1]
        x = np.linspace(0,sx,sx)*calib
        y = np.linspace(0,sy,sy)*calib
        xx,yy = np.meshgrid(y,x)
        
        fig = plt.figure()
#        ax = fig.gca(projection='3d')
#        ax = Axes3D(fig)
        ax = plt.axes(projection='3d')
#        my_cmap = sns.color_palette("Spectral",10)
#        bpyrmin = bpyrcon - bpyrcon[0]
#        bpyrmax = np.max(bpyrcon - bpyrcon[0])
#        surf = ax.plot_surface(xx,yy,(bpyrcon - bpyrcon[0]),cmap=cm.coolwarm,rstride=1,cstride=1,lw=0)
        
        surf = ax.plot_surface(xx,yy,steps,cmap=cm.coolwarm,rstride=1,cstride=1,lw=0)
        ax.view_init(azim=20,elev=20)
#        ax.set_xticks([])
#        ax.set_yticks([])
#        ax.set_zticks([])
        cbar = plt.colorbar(surf, shrink=0.7, aspect=15)
        cbar.ax.set_title('mm',fontsize=11)
        ax.set_xlabel('mm')
        ax.set_ylabel('mm')
        ax.set_zlabel('mm')
        ax.set_xlim(0,30)
        ax.set_zlim(0,30)
        set_aspect_equal_3d(ax)
        
#        for ii in xrange(0,180,4):
#            el = 50
#            ax.view_init(azim=ii,elev=el)
#            path = r"H:/Thesis 2017/2444776hsshdk/figures/3D_steps/f%d/" % f
#            plt.savefig(path+"steps%d_%d_e%d.png" % (f,ii,el),bbox_inches='tight',
#                        pad_inches=0.3,dpi=300)
            
        plt.savefig("steps90_20_e20.png", bbox_inches='tight', pad_inches=0.3,dpi=300)
                        
#        path = r"H:/Thesis 2017/2444776hsshdk/figures/3D_steps/"
#        plt.savefig(path+"test%s.png"%i)
        plt.show()
        os.chdir('..')
    