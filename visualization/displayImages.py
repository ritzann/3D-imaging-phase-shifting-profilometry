from __future__ import division
import matplotlib as mpl
#mpl.use("Agg")
import numpy as np
import skimage.io as ski
import scipy.ndimage as ndimage
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

# mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
# mpl.rc('text', usetex=True)

def saveImages(path):
    """
    Plot wrapped and unwrapped phases from unprocessed and processed images.
    """    

    os.chdir(path)
    im0 = loadmat('results.mat')['newimg0']
    im1 = loadmat('results.mat')['newimg1']
    im2 = loadmat('results.mat')['newimg2']
    im3 = loadmat('results.mat')['newimg3']
    im4 = loadmat('results.mat')['newimg4']
    im5 = loadmat('results.mat')['newimg5']
    
    fig, ax = plt.subplots(3, 2) 
    ax1, ax2, ax3, ax4, ax5, ax6 = ax.ravel()
    im = []
    cax = []
    for i, axes in enumerate(ax.ravel()):
        axes.set_yticks([])
        axes.set_xticks([])
        # divider = make_axes_locatable(axes)
        # cax.append(divider.append_axes("right", size="5%", pad=0.05))

    im.append(ax1.imshow(im0, cmap='gray', label='D1'))
    im.append(ax2.imshow(im4, cmap='gray', label='D2'))
    im.append(ax3.imshow(im1, cmap='gray', label='D3'))
    im.append(ax4.imshow(im3, cmap='gray', label='D4'))
    im.append(ax5.imshow(im5, cmap='gray', label='D5'))
    im.append(ax6.imshow(im2, cmap='gray', label='D6'))
    ax1.set_title('D1')
    ax2.set_title('D4')
    ax3.set_title('D2')
    ax4.set_title('D5')
    ax5.set_title('D3')
    ax6.set_title('D6')
           
    fig.subplots_adjust(wspace = -0.65)
    fig.savefig('images.eps', bbox_inches='tight', pad_inches=0)
    
def saveSpectra(path):
    """
    Plot wrapped and unwrapped phases from unprocessed and processed images.
    """    

    os.chdir(path)
    sp0 = loadmat('results.mat')['newspec0']
    sp1 = loadmat('results.mat')['newspec1']
    sp2 = loadmat('results.mat')['newspec2']
    sp3 = loadmat('results.mat')['newspec3']
    sp4 = loadmat('results.mat')['newspec4']
    sp5 = loadmat('results.mat')['newspec5']
    
    fig, ax = plt.subplots(3, 2) 
    ax1, ax2, ax3, ax4, ax5, ax6 = ax.ravel()
    im = []
    cax = []
    for i, axes in enumerate(ax.ravel()):
        axes.set_yticks([])
        axes.set_xticks([])

    im.append(ax1.imshow(sp0, cmap='gray', label='D1'))
    im.append(ax2.imshow(sp4, cmap='gray', label='D2'))
    im.append(ax3.imshow(sp1, cmap='gray', label='D3'))
    im.append(ax4.imshow(sp3, cmap='gray', label='D4'))
    im.append(ax5.imshow(sp5, cmap='gray', label='D5'))
    im.append(ax6.imshow(sp2, cmap='gray', label='D6'))
    ax1.set_title('D1')
    ax2.set_title('D4')
    ax3.set_title('D2')
    ax4.set_title('D5')
    ax5.set_title('D3')
    ax6.set_title('D6')
           
    fig.subplots_adjust(wspace = -0.65)
    # fig.savefig('spectra.eps', bbox_inches='tight', pad_inches=0)

def compareApodized(path1,path2,path3,path4):  
    os.chdir(path1)
    imSphere = loadmat('results.mat')['img3']
    specSphere = loadmat('results.mat')['spec3']
    imSphereNew = loadmat('results.mat')['newimg3']
    specSphereNew = loadmat('results.mat')['newspec3']

    os.chdir(path2)
    imCone = loadmat('results.mat')['img3']
    specCone = loadmat('results.mat')['spec3']
    imConeNew = loadmat('results.mat')['newimg3']
    specConeNew = loadmat('results.mat')['newspec3']

    os.chdir(path3)
    imSine = loadmat('results.mat')['img3']
    specSine = loadmat('results.mat')['spec3']
    imSineNew = loadmat('results.mat')['newimg3']
    specSineNEw = loadmat('results.mat')['newspec3']
    
    fig, ax = plt.subplots(2, 6) 
    ax1, ax2, ax3, ax4, ax5, ax6, \
        ax7, ax8, ax9, ax10, ax11, ax12 = ax.ravel()
    im = []
    cax = []
    for i, axes in enumerate(ax.ravel()):
        axes.set_yticks([])
        axes.set_xticks([])

    # spec display: (log(abs(spec)+1).^(1/3))
    im.append(ax1.imshow((np.log(abs(specSphere)+1)**(1/3))/255., cmap='gray'))
    im.append(ax2.imshow((np.log(abs(specSine)+1)**(1/3))/255., cmap='gray'))
    im.append(ax3.imshow((np.log(abs(specCone)+1)**(1/3))/255., cmap='gray'))
    im.append(ax4.imshow(imSphere, cmap='gray'))
    im.append(ax5.imshow(imSine, cmap='gray'))
    im.append(ax6.imshow(imCone, cmap='gray'))
    im.append(ax7.imshow(np.log(abs(specSphereNew)+1)**(1/3), cmap='gray'))
    im.append(ax8.imshow(np.log(abs(specSineNEw)+1)**(1/3), cmap='gray'))
    im.append(ax9.imshow(np.log(abs(specConeNew)+1)**(1/3), cmap='gray'))
    im.append(ax10.imshow(imSphereNew, cmap='gray'))
    im.append(ax11.imshow(imSineNew, cmap='gray'))
    im.append(ax12.imshow(imConeNew, cmap='gray'))
           
    fig.subplots_adjust(hspace = -0.75)
    # fig.subplots_adjust(wspace = -0.05)
    os.chdir(path4)
    fig.savefig('compareApodized.eps', bbox_inches='tight', pad_inches=0)


def compareApodized2(path1,path2,path3,path4):  
    os.chdir(path1)
    imSphere = loadmat('results.mat')['img3']
    specSphere = loadmat('results.mat')['spec3']
    imSphereNew = loadmat('results.mat')['newimg3']
    specSphereNew = loadmat('results.mat')['newspec3']

    os.chdir(path2)
    imCone = loadmat('results2.mat')['img3']
    specCone = loadmat('results2.mat')['spec3']
    imConeNew = loadmat('results2.mat')['newimg3']
    specConeNew = loadmat('results2.mat')['newspec3']

    os.chdir(path3)
    imSine = loadmat('results2.mat')['img3']
    specSine = loadmat('results2.mat')['spec3']
    imSineNew = loadmat('results.mat')['newimg3']
    specSineNew = loadmat('results2.mat')['newspec3']
    
    fig, ax = plt.subplots(2, 4) 
    ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8 = ax.ravel()
    im = []
    cax = []
    for i, axes in enumerate(ax.ravel()):
        axes.set_yticks([])
        axes.set_xticks([])
    # spec display: (log(abs(spec)+1).^(1/3))
    im.append(ax1.imshow((np.log(abs(specCone)+1)**(1/3))/255., cmap='gray'))
    im.append(ax2.imshow(np.log(abs(specConeNew)+1)**(1/3)/255., cmap='gray'))
    im.append(ax3.imshow(imCone, cmap='gray'))
    im.append(ax4.imshow(imConeNew, cmap='gray'))
    im.append(ax5.imshow((np.log(abs(specSine)+1)**(1/3))/255., cmap='gray'))
    im.append(ax6.imshow(np.log(abs(specSineNew)+1)**(1/3)/255., cmap='gray'))
    im.append(ax7.imshow(imSine, cmap='gray'))
    im.append(ax8.imshow(imSineNew, cmap='gray'))

    # def get_axis_limits(ax,scale=0.9):
    #     return ax.get_xlim()[1]*scale, ax.get_ylim()[1]*scale
    # ax1.annotate('(a)',xy=get_axis_limits(ax1),color='white')
    # ax2.annotate('(b)',xy=get_axis_limits(ax2),color='white')
    ax1.annotate('(a)',xy=(5,15),color='white')
    ax2.annotate('(b)',xy=(5,15),color='white')
    ax3.annotate('(c)',xy=(5,15),color='white')
    ax4.annotate('(d)',xy=(5,15),color='white')
    ax5.annotate('(e)',xy=(5,15),color='white')
    ax6.annotate('(f)',xy=(5,15),color='white')
    ax7.annotate('(g)',xy=(5,15),color='black')
    ax8.annotate('(h)',xy=(5,15),color='black')
    fig.subplots_adjust(hspace = -0.5)
    fig.subplots_adjust(wspace = 0.1)
    os.chdir(path4)
    fig.savefig('compareApodized2.eps', bbox_inches='tight', pad_inches=0)

def plotIntensityErrorMap(path1,path2,path3,path4,color):  
    os.chdir(path1)
    sphereD = loadmat('results_unapodized.mat')['Ierr1']
    sphereSC = loadmat('results_unapodized.mat')['Ierr2']
    sphereDA = loadmat('results_apodized.mat')['Ierr1']
    sphereSCA = loadmat('results_apodized.mat')['Ierr2']

    os.chdir(path2)
    coneD = loadmat('results_unapodized.mat')['Ierr1']
    coneSC = loadmat('results_unapodized.mat')['Ierr2']
    coneDA = loadmat('results_apodized.mat')['Ierr1']
    coneSCA = loadmat('results_apodized.mat')['Ierr2']

    os.chdir(path3)
    sineD = loadmat('results_unapodized.mat')['Ierr1']
    sineSC = loadmat('results_unapodized.mat')['Ierr2']
    sineDA = loadmat('results_apodized.mat')['Ierr1']
    sineSCA = loadmat('results_apodized.mat')['Ierr2']
    
    fig, axes = plt.subplots(3, 4, sharex=True, sharey=True) 
    ax1, ax2, ax3, ax4, ax5, ax6, \
        ax7, ax8, ax9, ax10, ax11, ax12 = axes.ravel()
    im = []
    cax = []
    for i, axes in enumerate(axes.ravel()):
        axes.set_yticks([])
        axes.set_xticks([])
    
    im.append(ax1.imshow(sphereD, vmin=0, vmax=0.2))
    im.append(ax2.imshow(sphereDA, vmin=0, vmax=0.2))
    im.append(ax3.imshow(sphereSC, vmin=0, vmax=0.2))
    im.append(ax4.imshow(sphereSCA, vmin=0, vmax=0.2))
    im.append(ax5.imshow(coneD, vmin=0, vmax=0.2))
    im.append(ax6.imshow(coneDA, vmin=0, vmax=0.2))
    im.append(ax7.imshow(coneSC, vmin=0, vmax=0.2))
    im.append(ax8.imshow(coneSCA, vmin=0, vmax=0.2))
    im.append(ax9.imshow(sineD, vmin=0, vmax=0.2))
    im.append(ax10.imshow(sineDA, vmin=0, vmax=0.2))
    im.append(ax11.imshow(sineSC, vmin=0, vmax=0.2))
    im.append(ax12.imshow(sineSCA, vmin=0, vmax=0.2))

    # cax, kw = mpl.colorbar.make_axes([j for j in ax.flat])
    # fig.colorbar(im, cax=cax)
           
    fig.subplots_adjust(hspace = 0.05)
    fig.subplots_adjust(wspace = 0.05)
    os.chdir(path4)
    fig.savefig('intensityErrorMaps.pdf', bbox_inches='tight', pad_inches=0)

    # def get_axis_limits(ax,scale=0.9):
    #     return ax.get_xlim()[1]*scale, ax.get_ylim()[1]*scale
    ax1.annotate('(a)',xy=(5,15),color='white')
    ax2.annotate('(b)',xy=(5,15),color='white')
    ax3.annotate('(c)',xy=(5,15),color='white')
    ax4.annotate('(d)',xy=(5,15),color='white')
    ax5.annotate('(e)',xy=(5,15),color='white')
    ax6.annotate('(f)',xy=(5,15),color='white')
    ax7.annotate('(g)',xy=(5,15),color='white')
    ax8.annotate('(h)',xy=(5,15),color='white')
    ax9.annotate('(i)',xy=(5,15),color='white')
    ax10.annotate('(i)',xy=(5,15),color='white')
    ax11.annotate('(k)',xy=(5,15),color='white')
    ax12.annotate('(l)',xy=(5,15),color='white')
    fig.subplots_adjust(hspace = 0.05)
    fig.subplots_adjust(wspace = 0.05)
    os.chdir(path4)
    fig.savefig('intensityErrorMaps2.pdf', bbox_inches='tight', pad_inches=0)
#------------------------------------------------------------------------------
    
if __name__ == "__main__":    
    path = './'
    path1 = r"/Users/ritzann/Documents/Dissertation/Publications/Phys Rev Applied Paper/[200305_095757]Sphere_FGI_LDR_Experiment_mRow=150"
    path2 = r"/Users/ritzann/Documents/Dissertation/Publications/Phys Rev Applied Paper/[200304_181413]Cone_FGI_LDR_Experiment_mRow=150"
    path3 = r"/Users/ritzann/Documents/Dissertation/Publications/Phys Rev Applied Paper/[200305_132525]Sine_FGI_LDR_Experiment_mRow=150"
    pathGen = r"/Users/ritzann/Documents/Dissertation/Publications/Phys Rev Applied Paper/figs2"

    # saveImages(path2)
    # saveSpectra(path2)
    # compareApodized2(path1,path2,path3,pathGen)
    color = 'magma'
    plotIntensityErrorMap(path1,path2,path3,pathGen,color)