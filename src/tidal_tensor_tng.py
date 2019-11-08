from pygadgetreader import *
import numpy as np
import numpy.linalg as npl
import numpy.fft as npf
from scipy.ndimage import gaussian_filter
from illustris_python.snapshot import loadHalo, snapPath, loadSubhalo, loadSubset
#import pyfits as fits
#from mb2 import *
from numpy.core.records import fromarrays
import fitsio as fi
import gc
import h5py 
import os

import pylab as plt
plt.switch_backend('agg')
plt.style.use('y1a1')


base_dir = '/physics/nkhandai/mb2/snapdir/'
dens_dir = '/Volumes/groke/'
#'/home/rmandelb.proj/ssamurof/mb2_tidal/'



def build_shape_cube(snap, resolution=512, npart=1024):
    """
    Computes a density cube for the specified snapshot
    """
    base_name = base_dir+ 'snapdir_%03d/snapshot_%03d'%(snap,snap)

    # Read the size of the box in Mpc
    boxsize = 205
    #readheader(base_name, 'boxsize')/1000 # h^-1 Mpc


    cat_name = 'data/cats/fits/TNG300-1_%d_non-reduced_galaxy_shapes.fits'%snap
    shape_cat = fi.FITS(cat_name)[1].read()
    ngal = len(shape_cat)

    sij = np.zeros((resolution,resolution,resolution,3,3))
    #sigma = np.zeros((resolution,resolution,resolution,3,3))
    num = np.zeros((resolution,resolution,resolution,3,3))

    X,Y,Z = np.meshgrid(np.arange(0,resolution,1),np.arange(0,resolution,1),np.arange(0,resolution,1))
    B = np.linspace(0,boxsize,resolution+1)
    lower,upper = B[:-1],B[1:]

    # Compute the shape cube by adding galaxies one at a time to cells
    for i in range(ngal):
        print ("Processing galaxy %d/%d"%(i,ngal))

        # put this galaxy into a cell
        info = shape_cat[i]
        ix = np.argwhere((shape_cat['x'][i]>=lower) & (shape_cat['x'][i]<upper))[0,0]
        iy = np.argwhere((shape_cat['y'][i]>=lower) & (shape_cat['y'][i]<upper))[0,0]
        iz = np.argwhere((shape_cat['z'][i]>=lower) & (shape_cat['z'][i]<upper))[0,0]

        # reconstruct the 3x3 shape matrix for this galaxy
        a0 = np.array([shape_cat['av_x'][i],shape_cat['av_y'][i],shape_cat['av_z'][i]])
        b0 = np.array([shape_cat['bv_x'][i],shape_cat['bv_y'][i],shape_cat['bv_z'][i]])
        c0 = np.array([shape_cat['cv_x'][i],shape_cat['cv_y'][i],shape_cat['cv_z'][i]])
        V = np.diag(np.array([shape_cat['a'][i]**2, shape_cat['b'][i]**2, shape_cat['c'][i]**2]))
        v = np.array([a0,b0,c0])
        v0=np.linalg.inv(v)
        I = np.dot(v0,np.dot(V,v))

        #print(I)
       # import pdb ; pdb.set_trace()

        # add it to the correct cell
        sij[ix,iy,iz,:,:] += I
        num[ix,iy,iz,:,:] += 1

    import pdb ; pdb.set_trace()
    sij /= num
    sij[np.isinf(sij)]=0
    sij[np.isnan(sij)]=0

   # import pdb ; pdb.set_trace()

    # also process the shape dispersion per bin
#    for i in range(ngal):
#        print ("Processing sigma for galaxy %d/%d"%(i,ngal))
#
#        # put this galaxy into a cell
#        info = shape_cat[i]
#        ix = np.argwhere((shape_cat['x'][i]>=lower) & (shape_cat['x'][i]<upper))[0,0]
#        iy = np.argwhere((shape_cat['y'][i]>=lower) & (shape_cat['y'][i]<upper))[0,0]
#        iz = np.argwhere((shape_cat['z'][i]>=lower) & (shape_cat['z'][i]<upper))[0,0]
#
#        # reconstruct the 3x3 shape matrix for this galaxy
#        a0 = np.array([shape_cat['av_x'][i],shape_cat['av_y'][i],shape_cat['av_z'][i]]) * shape_cat['a'][i]
#        b0 = np.array([shape_cat['bv_x'][i],shape_cat['bv_y'][i],shape_cat['bv_z'][i]]) * shape_cat['b'][i]
#        c0 = np.array([shape_cat['cv_x'][i],shape_cat['cv_y'][i],shape_cat['cv_z'][i]]) * shape_cat['c'][i]
#        I = np.array([a0,b0,c0])
#
#        # add it to the correct cell
#        sigma[ix,iy,iz,:,:] += (I-sij[ix,iy,iz,:,:]) * (I-sij[ix,iy,iz,:,:])


    del(shape_cat)
    #del(num)
    del(X)
    del(Y)
    del(Z)
    del(B)
    del(lower)
    del(upper)
    gc.collect()

#    for i in range(3):
#        for j in range(3):
#            sigma[:,:,:,i,j] /= num
#            sigma[:,:,:,i,j] = np.sqrt(sigma[:,:,:,i,j])
#
#    sigma[np.isinf(sigma)]=0
#    sigma[np.isnan(sigma)]=0

    return sij

def gen_shape_cubes(snaps=[], resolution=512):
    """
    Outputs a fits file with the density cube for the corresponding snapshots
    """
    for sn in snaps:
        gamma = build_shape_cube(sn, resolution=resolution)
        outfits = fi.FITS("stellar_shape_vects_%03d_%d.fits"%(sn,resolution), 'rw')
        outfits.write(gamma)
        #fits.writeto("stellar_shape_var_%03d.fits"%(sn), dgamma)


def build_density_cube(snap, resolution=512, npart=600, ptype='dm'):
    """
    Computes a density cube for the specified snapshot
    """
    base_name = base_dir+ 'snapdir_%03d/snapshot_%03d'%(snap,snap)
    basePath = "/Volumes/muskrat/other_peoples_datasets/illustrisTNG/TNG300-1/output/"

    # Read the size of the box in mpc
    boxsize = 205
    d = {'dm':1,'star':4}
    gName = "PartType" + str(d[ptype])

    i=0

    density = np.zeros((resolution,resolution,resolution))
    # Compute the density cube by reading each file one by one
    #for i in range(npart):
     #   print ("Processing part %d/%d"%(i,npart)) 

    for j in range(npart):
        f = h5py.File(snapPath(basePath, snap, chunkNum=j), 'r')
        print(j)
        #import pdb ; pdb.set_trace()
        header = dict(f['Header'].attrs.items())
        ptcl_coords = f[gName]["Coordinates"][:]/1000.
        hist,edges = np.histogramdd(ptcl_coords, resolution, range=[[0,boxsize],[0,boxsize],[0,boxsize]])
        density = density + hist
        f.close()
        #i+=1

    #import pdb ; pdb.set_trace()

    return density

def gen_density_cubes(snaps=[], ptype='dm', resolution=512):
    """
    Outputs a fits file with the density cube for the corresponding snapshots
    """
    for sn in snaps:
        dens = build_density_cube(sn, ptype=ptype, resolution=resolution)
        os.system("rm /Volumes/groke/%s_density_%03d_%d.fits"%(ptype,sn,resolution))
        outfits = fi.FITS("/Volumes/groke/%s_density_%03d_%d.fits"%(ptype,sn,resolution),'rw')
        outfits.write(dens)
        outfits.close()

def compute_tidal_tensor(dens, smoothing=0.25, pixel_size=0.1953):
    """
    Computes the tidal tensor given a density field
    Pixel size and smoothing scale in h^{-1} Mpc
    """
    nx = dens.shape[0]

    print('pixel scale = %3.3f'%pixel_size)

    k  = npf.fftfreq(nx)[np.mgrid[0:nx,0:nx,0:nx]]
    tidal_tensor = np.zeros((nx,nx,nx,3,3),dtype=np.float32)
    tidal_tensor2 = np.zeros((nx,nx,nx,1),dtype=np.float32)
    sigma = smoothing/pixel_size
    fft_dens = npf.fftn(gaussian_filter(dens,sigma,mode='wrap')) # 3D (512 x 512 x 512) grid ; each cell is a k mode
    for i in range(3):
        for j in range(3):
            # k[i], k[j] are 3D matrices, as is k
            temp =  fft_dens * k[i]*k[j]/(k[0]**2 + k[1]**2 + k[2]**2)
            temp[0,0,0] = 0

            tidal_tensor[:,:,:,i,j] = npf.ifftn(temp).real
            import pdb ; pdb.set_trace()
    
    return tidal_tensor

def gen_tidal_tensors(snaps=[], smoothing=[], ptype='dm', resolution=512):
    """
    Computes the tidal tensor for the snapshots provided
    """
    for i in snaps:
        dens = fi.FITS(dens_dir+'%s_density_%03d_%d.fits'%(ptype,i,resolution))[-1].read()
        for s in smoothing:
            tid  = compute_tidal_tensor(dens, smoothing=s, pixel_size=205./resolution)
            out = fi.FITS(dens_dir+'%s_tidal_%03d_%0.2f_%d.fits'%(ptype,i,s,resolution),'rw') ; out.write(tid) ; out.close()
            # Diagonalise the tidal matrix while we are at it
            vals, vects = npl.eigh(tid)
            out = fi.FITS(dens_dir+'%s_tidal_vals_%03d_%0.2f_%d.fits'%(ptype,i,s,resolution),'rw') ; out.write(vals) ; out.close()
            out = fi.FITS(dens_dir+'%s_tidal_vects_%03d_%0.2f_%d.fits'%(ptype,i,s,resolution),'rw') ; out.write(vects) ; out.close()

def snapPath(basePath, snapNum, chunkNum=0):
    """ Return absolute path to a snapshot HDF5 file (modify as needed). """
    snapPath = basePath + '/snapdir_' + str(snapNum).zfill(3) + '/'
    filePath = snapPath + 'snap_' + str(snapNum).zfill(3)
    filePath += '.' + str(chunkNum) + '.hdf5'
    return filePath