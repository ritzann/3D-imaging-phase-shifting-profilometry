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

    dirName = r"/Users/ritzann/Documents/Dissertation/Publications/Phys Rev Applied Paper/[200304_181413]Cone_FGI_LDR_Experiment_mRow=150"
    os.chdir(dirName)

    coneRadius = 1.9894
    coneHeight = 2.11
    sphereRadius = 2.069
    sinePeriod = 2
    sineAmp = 0.5
    
    Z = loadmat('results.mat')['Z']
    Z_ = Z + np.min(Z)
    maxZ = np.max(Z_)
    minZ = np.min(Z_)

    conecalibz = coneHeight/(maxZ - minZ)
    spherecalibz = sphereRadius/(maxZ - minZ)
    sinecalibz = sineAmp/(maxZ - minZ)

    Zcone = Z_*conecalibz
    Zsphere = Z_*spherecalibz
    Zsine = Z_*sinecalibz

    sx = 150
    sy = sx
    calibxy = 3.2/150
    x = np.linspace(0,sx,sx)*calibxy
    y = np.linspace(0,sy,sy)*calibxy
    xx,yy = np.meshgrid(y,x)
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # We change the fontsize of minor ticks label 
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.tick_params(axis='both', which='minor', labelsize=8)

    Zcone = loadmat('results.mat')['Zcone']
    # nanZcone = np.isnan(Zcone)
    # Zcone[nanZcone] = 0
    Zcone = -Zcone
    surf = ax.plot_surface(xx,yy,Zcone,cmap='magma',rstride=1,cstride=1,lw=0, \
        vmin=np.nanmin(Zcone), vmax=np.nanmax(Zcone))
    
    ax.view_init(azim=-56,elev=13)
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_zticks([])
    # ax.grid(False)

    ## Change Background
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.zaxis.pane.set_edgecolor('black')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False


    cbar = plt.colorbar(surf, shrink=0.5, aspect=20)
    # cbar.ax.set_title('mm',fontsize=11)
    ax.set_xlabel('mm')
    ax.set_ylabel('mm')
    ax.set_zlabel('mm')
    # ax.set_xlim(0,3.5)
    # ax.set_ylim(0,3.5)
    # ax.set_zlim(1,3)
    set_aspect_equal_3d(ax)
    
#        for ii in xrange(0,180,4):
#            el = 50
#            ax.view_init(azim=ii,elev=el)
#            path = r"H:/Thesis 2017/2444776hsshdk/figures/3D_steps/f%d/" % f
#            plt.savefig(path+"steps%d_%d_e%d.png" % (f,ii,el),bbox_inches='tight',
#                        pad_inches=0.3,dpi=300)
        
    # plt.savefig("steps90_20_e20.png", bbox_inches='tight', pad_inches=0.3,dpi=300)
    plt.savefig('3D.eps', bbox_inches='tight', pad_inches=0.3)
    plt.show()