# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 13:52:59 2015

@author: Ritz Ann
"""
import matplotlib as mpl
#mpl.use("Agg")
import numpy as np
import skimage.io as ski
import scipy.ndimage as ndimage
import cv2
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

#_dir = "../stone wall nighttime/skewed/"
#wrapped = loadmat(_dir+'cropped_wrapped.mat')['wrpdphase_4crop']

def rgb2gray(rgb):
    """
    Converts an RGB image to grayscale image.
    """

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray
    
def imagesc (filename):
    
    scaled = (filename - np.min(filename))/np.max(filename-np.min(filename))
    
    return scaled

#def crop(img, y1=1427, x1=580, y2=2210, x2=1333): #pebbled wall
#def crop(img, x1=1704,  y1=1912, x2=1960, y2=2168): #man - small pr
#def crop(img, x1=249, y1=253, x2=1394, y2=1950): #man w/ ref
def crop(img, x1=273, y1=481, x2=1183, y2=1747): #man
#def crop(img, x1=409, y1=157, x2=3070, y2=2354): #two men - all
#def crop(img, x1=1550, y1=267, x2=1806, y2=523): #two men - small pyr
#def crop(img, x1=401, y1=725, x2=3070, y2=2346): #two men w/ ref
#def crop(img, x1=421, y1=921, x2=1594, y2=2234): #twomen - man1
#def crop(img, x1=1732, y1=868, x2=3057, y2=2373): #twomen - man2
#def crop(img, x1=1132, y1=236, x2=1399, y2=492): #turtle - smallypyr
#def crop(img, x1=265, y1=781, x2=2766, y2=2322): #turtle
    return img[y1:y2, x1:x2]
    
#def std2(unwrapped):
#    
#    unwrapped[np.where(np.flatten(unwrapped) > 3*np.std(unwrapped))] = np.std(unwrapped)
#    np.where(np.flatten(unwrapped) < 3*np.std(unwrapped)) = np.std(unwrapped)

    
def removeVignetting(bg, blocksize):
    """
    Smoothens the background by taking the mean of a specified M x N block
    and applies bicubic interpolation.
    """
    
    if len(bg.shape) == 3:
        xyshape = np.ceil(1.*np.array(bg.shape[:2])/blocksize)
        sbg = np.empty(tuple((xyshape[0], xyshape[1],3)))
        for i, row in enumerate(range(0, bg.shape[0], blocksize)):
            for j, col in enumerate(range(0, bg.shape[1], blocksize)):
                sbg[i, j] = np.mean(bg[row:row + blocksize, col:col + blocksize])
        
    #    noVignette = cv2.resize(sbg, bg.shape, interpolation=cv2.INTER_CUBIC)
        newdim = (bg.shape[1], bg.shape[0])
        a = cv2.resize(sbg[:,:,0], newdim, interpolation=cv2.INTER_CUBIC)
        b = cv2.resize(sbg[:,:,1], newdim, interpolation=cv2.INTER_CUBIC)
        c = cv2.resize(sbg[:,:,2], newdim, interpolation=cv2.INTER_CUBIC)
        
        noVignette = np.dstack((a,b,c))
        return noVignette

    elif len(bg.shape) == 2:
       print "here"
       xyshape = np.ceil(1.*np.array(bg.shape[:2])/blocksize)
       sbg = np.empty(xyshape)
       for i, row in enumerate(range(0, bg.shape[0], blocksize)):
           for j, col in enumerate(range(0, bg.shape[1], blocksize)):
               sbg[i, j] = np.mean(bg[row:row + blocksize, col:col + blocksize])
       newdim = (bg.shape[1], bg.shape[0])  
       noVignette = bg - cv2.resize(sbg, newdim, interpolation=cv2.INTER_CUBIC)
       return noVignette
    
def gammaInversion(path, filename):
    """
    Linearizes the image by fitting a nonlinear function to the IO curve
    of the calibration chart.
    """
    os.chdir(path)
    filename = 'grid1.tif'
    inputgrid = plt.imread(filename)
    
    red = inputgrid[:,:,0]
    green = inputgrid[:,:,1]
    blue = inputgrid[:,:,2]
    graygrid = rgb2gray(inputgrid)
    
    plt.imshow(inputgrid)
    vals = ginput(3)
    px = [int(i[0]) for i in vals]
    py = [int(i[1]) for i in vals]
    plt.close()
    
#    py = [223, 2184, 216]
#    px = [808, 848, 2381]
    
    num_cells = 15
    l = 20
    R = []
    G = []
    B = []
    gray = []
    for y in xrange(py[0], py[1], (py[1]-py[0])/num_cells):
        for x in xrange(px[0], px[2], (px[2]-px[0])/num_cells):
            d1 = np.mean(red[y-l:y+2, x-l:x+2])
            d2 = np.mean(green[y-l:y+2, x-l:x+2])
            d3 = np.mean(blue[y-l:y+2, x-l:x+2])
            d4 = np.mean(graygrid[y-l:y+2, x-l:x+2])
            R.append(d1)
            G.append(d2)
            B.append(d3)
            gray.append(d4)
            
    gray = np.array(gray)
    actual = np.arange(256)
    poly4d = lambda x, p1, p2, p3, p4, p5: p1*x**4 + p2*x**3 + p3*x**2 + p4**x + p5
    p0 = [3.4e-007, -0.0001, 0.01, 1.9, 22.8]
    p_fit, pcov = curve_fit(poly4d, actual, gray, p0=p0)
    poly_fit = poly4d(actual, *p_fit)
    
    lin_gray = np.interp(graygrid, actual, poly_fit)
    ski.imsave('outputpy.jpg', imagesc(lin_gray))  
    
    
    return actual, R, G, B, gray, poly_fit
    
        
#postprocessing techniques
def FFTfilter(image_gray):
    """
    Filtering in the Fourier domain (frequency space).    
    
    Returns the filtered unwrapped phase with no sinusoidal fringe artifacts.
    """
    
    im = fftshift(fft2(image_gray))
    fft_image = np.log(np.abs(im))
    
    nx = np.size(image_gray, 1)
    ny = np.size(image_gray, 0)
    x = np.linspace(-1,1,nx)
    y = np.linspace(-1,1,ny)
    X,Y = np.meshgrid(x,y)
    
    #create filter mask
    sx = 0.002
    sy = 0.004
    m = 0.175
    m1 = 0.35
    
    cross = np.ones((ny,nx))
    cross[np.where(np.abs(X) < sx)] = 0
    cross[np.where(np.abs(Y) < sy)] = 0
#    cross(np.where(np.abs(Y + m) < sy)) = 0
#    cross(np.where(np.abs(Y - m) < sy)) = 0
#    cross(np.where(np.abs(Y + m1) < sx)) = 0
#    cross(np.where(np.abs(Y - m1) < sx)) = 0
    
    r = np.sqrt(X**2 + Y**2)
    circle = np.zeros((ny,nx))
    circle[np.where(r < 0.08)] = 1

    mask = cross + circle
    mask[np.where(mask >= 2)] = 1
    
    #apply filter mask
    filter_mask = im*(mask)
    fft_filter = fft2(filter_mask)
    filter_mask = mask*fft_image        
    
    FINAL = imagesc(np.abs((fft_filter)))
#    FFT_FINAL = np.log(np.abs(fftshift(fft2(FINAL))))
    
    return fft_image, filter_mask, FINAL
 
def phaseToheight(path):
    """
    Returns the actual height/depth of the object.
    """
    
    os.chdir(path)
    im = loadmat('finalunwrappedpy.mat')['finalunwrappedpy']
    N = 15
    plt.imshow(im, interpolation = None)
    vals = ginput(N)
    Xi_0 = [int(i[0]) for i in vals]
    Yi_0 = [int(i[1]) for i in vals]
    plt.clf()

    ## Get C values (UC = 1)
    U = []
    for n in xrange(len(Xi_0)):
        xi_0 = Xi_0[n]
        yi_0 = Yi_0[n]
        Phi = im[yi_0,xi_0]
        U.append([-Phi, -xi_0, -xi_0*Phi, -yi_0, -yi_0*Phi])
    U = np.matrix(U)

    C = ((U.T*U).I)*U.T*np.matrix(np.ones((N,1)))
    
    ## Get D values (WD = V)
    ## height in mm
    z1 = 10
    z2 = 15
    z3 = 25
#    z4 = 20
#    z5 = 25        
    
    ## Select at least 6 points
    plt.imshow(im, interpolation = None)
    vals = ginput(N)
    Xi_1 = [int(i[0]) for i in vals]
    Yi_1 = [int(i[1]) for i in vals]
    plt.clf()
    
    plt.imshow(im, interpolation = None)
    vals = ginput(N)
    Xi_2 = [int(i[0]) for i in vals]
    Yi_2 = [int(i[1]) for i in vals]
    plt.clf()
    
    plt.imshow(im, interpolation = None)
    vals = ginput(N)
    Xi_3 = [int(i[0]) for i in vals]
    Yi_3 = [int(i[1]) for i in vals]    
    plt.close()
    
#    plt.imshow(im)
#    vals = ginput(N)
#    Xi_4 = [int(i[0]) for i in vals]
#    Yi_4 = [int(i[1]) for i in vals]
#    plt.close()
#    
#    plt.imshow(im)
#    vals = ginput(N)
#    Xi_5 = [int(i[0]) for i in vals]
#    Yi_5 = [int(i[1]) for i in vals]    
#    plt.close()
#    
    
    Xi_n = [Xi_1, Xi_2, Xi_3]
    
    W = []
    V = []
    X = np.array([Xi_1, Xi_2, Xi_3])
    Y = np.array([Yi_1, Yi_2, Yi_3])
    z = np.array([z1, z2, z3])
    
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
            
    savemat('finalunwrappedpy_converted1.mat', {'uconverted':zw})
    
    return C, D
    
    
## A Different Approach

#    os.chdir(path)
#    im = loadmat('final.mat')['final']
#    N = 400
#    plt.imshow(im, interpolation = None)
#    vals = ginput(1)
#    x = int(vals[0][0])
#    y = int(vals[0][1])
#    Yi_0, Xi_0 = np.where((im>im[y,x]-0.2) & (im<im[y,x]+0.2))
#    plt.close()
#    
#    chooser = np.random.randint(len(Yi_0), size=N)
#    Yi_0 = Yi_0[chooser]
#    Xi_0 = Xi_0[chooser]
#    ## Get C values (UC = 1)
#    U = []
#    for n in xrange(len(Xi_0)):
#        xi_0 = Xi_0[n]
#        yi_0 = Yi_0[n]
#        Phi = im[yi_0,xi_0]
#        U.append([-Phi, -xi_0, -xi_0*Phi, -yi_0, -yi_0*Phi])
#    U = np.matrix(U)
#
#    C = ((U.T*U).I)*U.T*np.matrix(np.ones((N,1)))
#    
#    ## Get D values (WD = V)
#    ## height in mm
#    z1 = 5
#    z2 = 10
#    z3 = 25
#    z4 = 20
#    z5 = 25
#   
#    ## Select at least 6 points
#    plt.imshow(im)
#    vals = ginput(1)
#    x = int(vals[0][0])
#    y = int(vals[0][1])
#    Yi_1, Xi_1 = np.where((im>im[y,x]-0.1) & (im<im[y,x]+0.1))
#    plt.close()
#    
#    chooser = np.random.randint(len(Yi_1), size=N)
#    Yi_1 = Yi_1[chooser]
#    Xi_1 = Xi_1[chooser]
#  
#    
#    plt.imshow(im)
#    vals = ginput(1)
#    x = int(vals[0][0])
#    y = int(vals[0][1])
#    Yi_2, Xi_2 = np.where((im>im[y,x]-0.2) & (im<im[y,x]+0.2))
#    plt.close()
#    
#    chooser = np.random.randint(len(Yi_2), size=N)
#    Yi_2 = Yi_2[chooser]
#    Xi_2 = Xi_2[chooser]
#    
#    plt.imshow(im)
#    vals = ginput(1)
#    x = int(vals[0][0])
#    y = int(vals[0][1])
#    Yi_3, Xi_3 = np.where((im>im[y,x]-0.2) & (im<im[y,x]+0.2))
#    plt.close()
#    
#    chooser = np.random.randint(len(Yi_3), size=N)
#    Yi_3 = Yi_3[chooser]
#    Xi_3 = Xi_3[chooser]
#  
#    plt.imshow(im)
#    vals = ginput(1)
#    x = int(vals[0][0])
#    y = int(vals[0][1])
#    Yi_4, Xi_4 = np.where((im>im[y,x]-0.2) & (im<im[y,x]+0.2))
#    plt.close()
#    
#    chooser = np.random.randint(len(Yi_4), size=N)
#    Yi_4 = Yi_4[chooser]
#    Xi_4 = Xi_4[chooser]
#    
#    plt.imshow(im)
#    vals = ginput(1)
#    x = int(vals[0][0])
#    y = int(vals[0][1])
#    Yi_5, Xi_5 = np.where((im>im[y,x]-0.2) & (im<im[y,x]+0.2))
#    plt.close()
#    
#    chooser = np.random.randint(len(Yi_5), size=N)
#    Yi_5 = Yi_5[chooser]
#    Xi_5 = Xi_5[chooser]
#       
#    Xi_n = [Xi_1, Xi_2, Xi_3, Xi_4, Xi_5]
#    
#    W = []
#    V = []
#    X = np.array([Xi_1, Xi_2, Xi_3, Xi_4, Xi_5])
#    Y = np.array([Yi_1, Yi_2, Yi_3, Yi_4, Yi_5])
#    z = np.array([z1, z2, z3, z4, z5])
#    
#    for l in range(np.size(X, 0)):
#        for n in range(np.size(X, 1)):
#            zk = z[l]
#            xi = X[l,n]
#            yi = Y[l,n]
#            Phi = im[yi, xi]
#            W.append([zk, zk*Phi, zk*xi, zk*xi*Phi, zk*yi, zk*yi*Phi])
#            V.append(1 + C[0,0]*Phi + (C[1,0] + C[2,0]*Phi)*xi + (C[3,0] + C[4,0]*Phi)*yi)
#    W = np.matrix(W)
#    V = np.matrix(V).T
#    
#    D = (W.T*W).I*W.T*V
#    
#    ## Compute for world height (zw)
#    zw = np.zeros(np.shape(im))
#    for xi in range(np.shape(im)[1]):
#        for yi in range(np.shape(im)[0]):
#            Phi = im[yi,xi]
#            zw_num = 1 + C[0,0]*Phi + (C[1,0] + C[2,0]*Phi)*xi + (C[3,0] + C[4,0]*Phi)*yi
#            zw_den = D[0,0] + D[1,0]*Phi + (D[2,    0] + D[3,0]*Phi)*xi + (D[4,0] + D[5,0]*Phi)*yi
#            zw[yi, xi] = zw_num/zw_den
#            
#    savemat('final_converted3.mat', {'uconverted':zw})    

def processImages(path, proc_path):
    os.chdir(path)
    try:
        os.mkdir(proc_path)
    except:
        pass
    for image in glob.glob('O*.TIF'):
        raw = 1.*plt.imread(image)
        
        noVignette = raw - 1.*plt.imread('white.TIF')
#        noVignette = raw - removeVignetting(raw,10)
#        noGamma = gammaInversion(noVignette)
#        ski.imsave(proc_path+image, noGamma)
        os.chdir(os.path.join(path,proc_path))
        savemat(image[:-4] + '.mat', {image[:-4]: rgb2gray(noVignette)})
        ski.imsave(image, 
                   (noVignette-np.min(noVignette))/np.max(noVignette-np.min(noVignette)))
#        plt.imshow(noVignette)
#        plt.savefig("processed no-vignette/plot"+image)
#        plt.clf()
        
###############################################################################
##                       PHASE-SHIFT PROFILOMETRY
###############################################################################


def PSP(path):
    """
    Performs phase-shift profilometry (PSP)
    """

    os.chdir(path)
    
    print("Loading object images...")    
    I = [rgb2gray(crop(plt.imread("O%d.TIF"%i))) for i in range(1,5)]
    R = [rgb2gray(crop(plt.imread("R%d.TIF"%i))) for i in range(1,5)]


#    I = [(rgb2gray(loadmat("O%d.mat"%i)['O%d'%i])) for i in range(1,5)]
#    print("Loading reference images...")
#    R = [(rgb2gray(loadmat("R%d.mat"%i)['R%d'%i])) for i in range(1,5)]
    print("Wrapping object phase...")
    wrapped = np.arctan2((I[3] - I[1]), (I[0] - I[2]))
    print("Wrapping reference phase...")
    wrappedr = np.arctan2((R[3] - R[1]), (R[0] - R[2]))
    print ("Saving object wrapped phase... ")
    savemat('wrappedpy.mat', {'wrappedpy':wrapped})
    print ("Saving reference wrapped phase... ")
    savemat('rwrappedpy.mat', {'rwrappedpy':wrappedr})
    del I
    del R
     
    print("Unwrapping object wrapped phase...")
    unwrapped = unwrap_phase(wrapped)
    print("Unwrapping reference wrapped phase...")
    unwrappedr = unwrap_phase(wrappedr)
    
    savemat('unwrapped.mat', {'unwrapped':unwrapped})
    savemat('unwrappedr.mat', {'unwrappedr':unwrappedr})   
    
    unwrappedpy = rampPSP(unwrapped)
    print("Saving object unwrapped phase...")
    savemat('unwrappedpy.mat', {'unwrappedpy':unwrappedpy})      
    
    runwrappedpy = rampPSP(unwrappedr)
    print("Saving reference unwrapped phase...")
    savemat('runwrappedpy.mat', {'runwrappedpy':runwrappedpy})
    
    final = unwrappedpy - runwrappedpy
    savemat('finalunwrappedpy.mat', {'finalunwrappedpy':final})
    
    return unwrapped
    
def rampPSP(unwrapped):

    delta_Q = (np.max(unwrapped)-np.min(unwrapped))/unwrapped.shape[1]  
    q = np.arange(np.min(unwrapped), np.max(unwrapped), delta_Q)
    q_real = q[np.arange(unwrapped.shape[1])]
    ramp = np.tile(q_real,[unwrapped.shape[0],1]) 
    unwrappedpy = unwrapped - ramp
       
    return unwrappedpy
    
def plotPSP(path):
    """
    Plot wrapped and unwrapped phases from unprocessed and processed images.
    """    
#    os.chdir(path)
#    wrappedm = loadmat('wrappedpy.mat')['wrappedpy']
#    unwrappedm = loadmat('unwrappedpy.mat')['unwrappedpy']
#    
#    os.chdir(processed_path)
#    wrappedpy = loadmat('wrappedpy.mat')['wrappedpy']
#    unwrappedpy = loadmat('unwrappedpy.mat')['unwrappedpy']
    os.chdir(path)
    wrapped = loadmat('wrappedpy.mat')['wrappedpy']
    unwrapped = loadmat('unwrappedpy.mat')['unwrappedpy']
    rwrapped = loadmat('rwrappedpy.mat')['rwrappedpy']
    runwrapped = loadmat('runwrappedpy.mat')['runwrappedpy']
#    finalunwrapped = loadmat('finalunwrappedpy.mat')['finalunwrappedpy']
    
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

#    for j,i in enumerate([1,3]):
#        cbar = fig.colorbar(im[i], cax=cax[j])
#        if i in [0,1]:
#           cbar.set_ticks([-np.pi, 0, np.pi])
#           cbar.set_ticklabels([r"$-\pi$", "$0$", r"$\pi$"])
           
    fig.subplots_adjust(wspace = -0.2)
    fig.subplots_adjust(hspace = 0.1)
           
    fig.savefig('turtle_phases.pdf', bbox_inches='tight', pad_inches=0)
    
#    plt.imshow(finalunwrapped, cmap='copper')    
#    plt.savefig('stone_RBS.pdf', bbox_inches='tight', pad_inches=0)
#    
#    plt.imshow(afterDBS, cmap='copper')  
#    plt.savefig('afterDBS.pdf', bbox_inches='tight', pad_inches=0)
    
    
def plotFFT(path):
    
    os.chdir(path)
    unwrapped = loadmat('final.mat')['final']
    
    sbg = removeVignetting(unwrapped, 90)
    print("Filtering image...")
    fft_image, filter_mask, FINAL = FFTfilter(imagesc(unwrapped))
    print("Rotating image...")
    FINAL = FINAL[::-1,::-1]

    fig, ax = plt.subplots(1, 3)
    ax1, ax2, ax3 = ax.ravel()
    im = []
    cax = []
    im.append(ax1.imshow(fft_image, cmap='gray'))
    im.append(ax2.imshow(filter_mask, cmap='gray'))
    im.append(ax3.imshow(FINAL, cmap='copper'))
    for i, axes in enumerate(ax.ravel()):
        axes.set_yticks([])
        axes.set_xticks([])
        if i in [2]:
            divider = make_axes_locatable(axes)
            cax.append(divider.append_axes("right", size="5%", pad=0.05))
    cbar = fig.colorbar(im[2], cax=cax[0])
        
    
    return fft_image, filter_mask, FINAL
    
#    fig.subplots_adjust(wspace = 0.1)          
#    fig.savefig('stone_fft.pdf', bbox_inches='tight', pad_inches=0)
    
#------------------------------------------------------------------------------
    
if __name__ == "__main__":    
    path = './'
    path = r"C:\Users\Ritz Ann\Desktop\man\man"
    
#    proc_path = r"Processed BS"
#    processed_path = os.path.join(path, proc_path)

#    PSP(path)
#    plotPSP(path)
    
   
#    largeim = loadmat('RBS_DBS.mat')['RDBS']
#    largeim = loadmat('finalunwrappedpy.mat')['finalunwrappedpy']
#    im = loadmat('finalunwrappedpy.mat')['finalunwrappedpy']
#    C = loadmat('C.mat')['C']
#    D = loadmat('D.mat')['D']
#    largezw = np.zeros(largeim.shape)
#    
#    xstep = largeim.shape[1]/4
#    ystep = largeim.shape[0]/4
#    
#    for l in range(4):
#        for k  in range(4):
#            if l==3:
#                endy=-1
#            else:
#                endy = (k+1)*ystep
#            if k==3:
#                endx=-1
#            else:
#                endx=(l+1)*xstep
#            
#            im = largeim[k*ystep:endy, l*xstep:endx]
#
#    
#    
#            zw = np.zeros(np.shape(im))
#            for xi in range(np.shape(im)[1]):
#                for yi in range(np.shape(im)[0]):
#                    Phi = im[yi,xi]
#                    zw_num = 1 + C[0,0]*Phi + (C[1,0] + C[2,0]*Phi)*xi + (C[3,0] + C[4,0]*Phi)*yi
#                    zw_den = D[0,0] + D[1,0]*Phi + (D[2,0] + D[3,0]*Phi)*xi + (D[4,0] + D[5,0]*Phi)*yi
#                    zw[yi, xi] = zw_num/zw_den
#            
#            largezw[k*ystep:endy, l*xstep:endx] = im
#    savemat('man1_actual_chunked.mat', {'uconverted':largezw})
    
    
#    zw = np.zeros(np.shape(im))
#    for xi in range(np.shape(im)[1]):
#        for yi in range(np.shape(im)[0]):
#            Phi = im[yi,xi]
#            zw_num = 1 + C[0,0]*Phi + (C[1,0] + C[2,0]*Phi)*xi + (C[3,0] + C[4,0]*Phi)*yi
#            zw_den = D[0,0] + D[1,0]*Phi + (D[2,0] + D[3,0]*Phi)*xi + (D[4,0] + D[5,0]*Phi)*yi
#            zw[yi, xi] = zw_num/zw_den
#    
#    savemat('turtle_actual.mat', {'uconverted':zw})
    

#    processImages(path, proc_path)
#    PSP(processed_path)        
    

#    fft_image, filter_mask, FINAL = plotFFT(path)
    
#    C, D = phaseToheight(path)
#    
#    plt.figure()
#    plt.xticks([])
#    plt.yticks([])
#    plt.imshow(fft_image, cmap='gray')
#    plt.savefig('combined_fft.pdf', bbox_inches='tight', pad_inches=0)
#
#    plt.imshow(filter_mask, cmap='gray')
#    plt.savefig('combined_mask.pdf', bbox_inches='tight', pad_inches=0)
#    
#    im = plt.imshow(FINAL, cmap='jet')
#    divider = make_axes_locatable(plt.gca())
#    cax = divider.append_axes("right", size="5%", pad=0.05)
#    cbar = fig.colorbar(im, cax=cax)
#    plt.savefig('combined_final04.pdf', bbox_inches='tight', pad_inches=0)
    
###############################################################################
##              Crop an image
###############################################################################
    
#    os.chdir(path)
#    o = plt.imread('O1.TIF');
#    obj = crop(o, y1=313, x1=205, y2=2253, x2=2981)
#    plt.imsave('objectfringe.pdf', obj)
    

###############################################################################
##              Save Plots with Colorbar
###############################################################################
#
#    os.chdir(path)
#    un = loadmat('unwrappedpy.mat')['unwrappedpy']
#    run = loadmat('runwrappedpy.mat')['runwrappedpy']
#    DBS = loadmat('afterDBS.mat')['afterDBS']
#    RBS = loadmat('afterRBS.mat')['afterRBS']
#    os.chdir(path)
    filtered = loadmat('manfiltered.mat')['manfiltered']
    plt.figure()
    plt.xticks([])
    plt.yticks([]) 
    im = plt.imshow(im, cmap='copper')
#    divider = make_axes_locatable(plt.gca())
#    cax = divider.append_axes("right", size="5%", pad=0.05)
#    cbar = plt.colorbar(im, cax=cax)
#    cbar.set_label('phase')
    plt.savefig('manfiltered.jpg', dpi = 300, bbox_inches='tight', pad_inches=0)

###############################################################################
##              Create Histogram Plot of in 3D
###############################################################################
    
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    hist, bin_edges = np.histogram(RBS.flatten(), normed=True, bins=100)
#    ax.bar(bin_edges[:-1],hist/np.sum(hist),width=bin_edges[1]-bin_edges[0],color='red', alpha=0.5, label='RBS')
#    hist, bin_edges = np.histogram(DBS.flatten(), normed=True, bins=100)    
#    ax.bar(bin_edges[:-1],hist/np.sum(hist),width=bin_edges[1]-bin_edges[0],color='gray', alpha=0.5, label='DBS')
#    ax.set_xlabel('phase value')
#    ax.set_ylabel('frequency')
#    plt.grid()
#    plt.legend()
#    plt.savefig('stonehisto.pdf', bbox_inches='tight', pad_inches=0)

    
###############################################################################
##                           Plot in 3D
###############################################################################
#    unwrappedpy = zw
#    x = np.arange(unwrappedpy.shape[1])
#    y = np.arange(unwrappedpy.shape[0])
#    X, Y = np.meshgrid(x,y)
#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection='3d')
#    ax.plot_surface(X, Y, unwrappedpy, rstride = 1, cstride = 1, linewidth=0, cmap='jet')
#    plt.show()

#    os.chdir(path)
#    im = loadmat('finalunwrappedpy.mat')['finalunwrappedpy']


###############################################################################
##                   Plot IO curve from Gamma Inversion
###############################################################################

#    actual, R, G, B, gray, poly_fit = gammaInversion(path, 'grid1.tif')
#    
#    os.chdir(path)
#    plt.plot(actual, R, 'r.', label='Red Channel')
#    plt.plot(actual, G, 'g.', label='Green Channel')
#    plt.plot(actual, B, 'b.', label='Blue Channel')
#    plt.plot(actual, gray, color = '0.4', marker = '.', label='Grayscale')
#    plt.xlim([0,255])
#    plt.ylim([0,260])
#    plt.ylabel('Output')
#    plt.xlabel('Input')
#    plt.legend(loc='lower right',frameon=False)
#    plt.savefig('IOcurve.pdf', bbox_inches='tight', pad_inches=0)
#    plt.clf()
#    
#    
#    plt.plot(actual, poly_fit, 'r-', label='4th Degree Polynomial')
#    plt.plot(actual, gray, color = '0.4', marker = '.', label='Grayscale')
#    plt.xlim([0,255])
#    plt.ylim([0,260])
#    plt.ylabel('Output')
#    plt.xlabel('Input')
#    plt.legend(loc='lower right',frameon=False)
#    plt.savefig('fits.pdf', bbox_inches='tight', pad_inches=0)
#    plt.clf()


###############################################################################
##        Phase To Height Conversion (Apply C and D parameters)
###############################################################################

#    os.chdir(path)
#    im = loadmat('finalunwrappedpy.mat')['finalunwrappedpy']
#    C = loadmat('C.mat')['C']
#    D = loadmat('D.mat')['D']
#
#    zw = np.zeros(np.shape(im))
#    for xi in range(np.shape(im)[1]):
#        for yi in range(np.shape(im)[0]):
#            Phi = im[yi,xi]
#            zw_num = 1 + C[0,0]*Phi + (C[1,0] + C[2,0]*Phi)*xi + (C[3,0] + C[4,0]*Phi)*yi
#            zw_den = D[0,0] + D[1,0]*Phi + (D[2,0] + D[3,0]*Phi)*xi + (D[4,0] + D[5,0]*Phi)*yi
#            zw[yi, xi] = zw_num/zw_den
#            
#    savemat('man_actual.mat', {'uconverted':zw})
#
#    data = loadmat('m1.mat')['m1']
#    plt.plot(data, color = '0.4', marker = '.')
#    plt.ylabel('mm')
#    plt.xlabel('px')
#    plt.savefig('20m1.pdf', bbox_inches='tight', pad_inches=0)
    