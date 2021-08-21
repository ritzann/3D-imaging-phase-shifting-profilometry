# -*- coding: utf-8 -*-
"""
Created on Fri Apr 07 19:22:31 2017

@author: Ritz Ann Aguilar
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, cm
import scipy.ndimage
import scipy.misc
from scipy.io import loadmat, savemat
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

def makeSteps(bias):
    
    x = np.linspace(bias,bias+40,500)
    
    heights = [0,4.5,2,4.5,4,2,4,3.5,2,3.5,3,2,3,2.5,2,2.5,2.4,2,2.4,2.3,2,2.3,0]
    widths = [2,2,2,1.5,1.5,2,1.5,1.5,2,1.5,1.5,2,1.5,1.5,2,1.5,1.5,2,1.5,1.5,2,2,2]
    cum_widths = np.cumsum([bias]+widths)
    
    y = np.zeros_like(x)
    
    for i in range(len(heights)):
        y[np.where((x > cum_widths[i]) & (x < cum_widths[i+1]))] = heights[i]
    
    return x, y

def improfile(im,x0,y0,x1,y1,num):
    
    #-- Generate some data...
#    x, y = np.mgrid[-5:5:0.1, -5:5:0.1]
#    z = np.sqrt(x**2 + y**2) + np.sin(x**2 + y**2)
#    lena = scipy.misc.lena()  # ADDED THIS ASYMMETRIC IMAGE
#    z = lena[320:420,330:430] # ADDED THIS ASYMMETRIC IMAGE
    
    #-- Extract the line...
    # Make a line with "num" points...
    x, y = np.linspace(x0, x1, num), np.linspace(y0, y1, num)
    
    # Extract the values along the line, using cubic interpolation
    zi = scipy.ndimage.map_coordinates(np.transpose(im), np.vstack((x,y)))
    
    return zi    
 
#def plotStepsProfile(steps):        
#    #Plot steps and profile
#    fig, axes = plt.subplots(nrows=2)
#    axes[0].imshow(steps,cmap=cm.coolwarm,label=r'$f=%s$'%f)
#    axes[0].set_xticks([])
#    axes[0].set_yticks([])
#    axes[0].plot([x0, x1], [y0, y1], 'ro-')
#    axes[0].axis('image')
#    axes[1].plot(steps_x,steps_profile,'k',lw=2)
#    axes[1].axis('equal')
#    axes[1].set_ylim(-1,5)
#    plt.legend(loc='lower right')
#    plt.show()

def plotStepsScan(steps,path):
        plt.figure()
        my_cmap = sns.color_palette("Spectral",10)
        s = plt.imshow(steps,cmap=my_cmap,label=r'$f=%s$'%f,as_cmap=True)
#        s = plt.imshow(steps,cmap=cm.coolwarm,label=r'$f=%s$'%f)
        plt.plot([x0, x1], [y0, y1], 'ro-')
        plt.axis('image')
        plt.xticks([])
        plt.yticks([])
        bbox_props = dict(fc="w", ec="0.5", alpha=0.9)
        ax1 = plt.gca()      
        plt.text(0.96, 0.03, r"$f_o = %s$" % f,  ha="right", va="bottom",
            size=11, bbox=bbox_props, transform=ax1.transAxes)
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", size="5%", pad=0.08)
        cbar = plt.colorbar(s, cax=cax)
        cbar.ax.set_title('mm',fontsize=11)
        plt.savefig(path+'stepsscan%s.pdf'%f,dpi = 300, bbox_inches='tight', pad_inches=0.05)

def plotStepsProfile(stepso_x,stepso_y,steps_x,steps_profile,path):
        plt.figure()
        plt.plot(stepso_x, stepso_y,'k',lw=2.5,label='input')
        plt.plot(steps_x,steps_profile,'r',lw=2,label='output')
        plt.axis('equal')
        plt.ylim(-1,5)
        plt.xlim(0,40)
        plt.xlabel('mm',fontsize=13)
        plt.ylabel('Height (mm)', fontsize=13)
        plt.legend(loc='upper right')
        ax2 = plt.gca()
        bbox_props = dict(fc="w", ec="0.5", alpha=0.9)
        plt.text(0.97, 0.05, r"$f_o = %s$" % f,  ha="right", va="bottom",
            size=13, bbox=bbox_props, transform=ax2.transAxes)
        plt.savefig(path+'stepsprofile%s.pdf'%f,dpi=300,bbox_inches='tight',pad_inches=0)


if __name__ == "__main__":
    
    bias = -0.4
    stepso_x, stepso_y = makeSteps(bias)    
    
    dirName = r"H:\Thesis 2017\PSP 2017\basler_white\03.21.2017 PSP\basler_acerwhite"
    os.chdir(dirName)
    
    x0, y0 = 42, 0
#    x1, y1 = 48.6903, 126
    y1 = 134
    m = (119-9)/(24-19)
    x1 = (y1-y0)/m + x0
    num = 500
    calib = 36./114
    steps_x = np.linspace(0,np.sqrt((x1-x0)**2+(y1-y0)**2),num)*calib
    

    f_vals = range(60,150,10)
    f_vals = [60,140]
    f_vals - [90,100,110]
    
    linestyle = ['-','--']
    marker = ['o','^','s']
    color = lambda level, n: "%s" % (level*(0.4/(n-1)))
    
    rc('font',**{'family':'serif'})
    rc('text', usetex=True)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)    
    
    for i, f in enumerate(f_vals[::]):
        os.chdir('f%s'%f)
        try:
            steps = loadmat('stepscon.mat')['stepscon']
        except:
            steps = loadmat('stepscon.mat')['zw']
        steps_profile = improfile(steps,x0,y0,x1,y1,num)
        savemat('steps_profile.mat', {'steps_profile':steps_profile})
        os.chdir('..')
##        ax.plot(steps_profile-steps_profile[0],label=r'$f=%s$'%f, linewidth=2*(i//5 +1))
#        ax.plot(steps_x, steps_profile,label=r'$f_o=%s$'%f, linewidth=2*(i//5 +1))
##        ax.plot(steps_x, steps_profile, linewidth=1, #marker=marker[i%len(marker)], 
##              linestyle=linestyle[0],label=r'$f=%s$'%f)
        
        path = r"H:/Thesis 2017/2444776hsshdk/figures/profiles/sns/"
        plotStepsScan(steps,path)
#        plotStepsProfile(stepso_x,stepso_y,steps_x,steps_profile,path)
         
        
#    ax.plot(stepso_x, stepso_y,lw=2,color='k',label='actual')
##    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
#    ax.legend(loc='upper right')
#    ax.set_ylabel('Height (mm)', fontsize = 13)
#    ax.set_xlabel('mm', fontsize = 13)    
#    ax.set_xlim(0,40)
#    ax.set_ylim(-1,5)
#    path = r"H:/Thesis 2017/2444776hsshdk/figures/"
#    fig.savefig(path+'steps60_140_actual.pdf', pad_inches=0)


    
            

    
    

    
    
    