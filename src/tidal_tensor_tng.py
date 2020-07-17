#from pygadgetreader import *
import numpy as np
import numpy.linalg as npl
import numpy.fft as npf
from scipy.ndimage import gaussian_filter
#from illustris_python.snapshot import loadHalo, snapPath, loadSubhalo, loadSubset
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
#dens_dir = '/Volumes/groke/'
dens_dir = '/home/rmandelb.proj/ssamurof/mb2_tidal/'


def get_trace_free_matrix(I):
    ndim = len(I[0])

    T = np.matrix.trace(I)
    K = 1./float(ndim) * T * np.identity(ndim)

    I_TF = I - K

    return I_TF


def build_shape_cube(snap, resolution=512, npart=1024, average_type='tensor'):
    """
    Computes a density cube for the specified snapshot
    """
    base_name = base_dir+ 'snapdir_%03d/snapshot_%03d'%(snap,snap)

    # Read the size of the box in Mpc
    boxsize = 205 # TNG300 value, again
    #readheader(base_name, 'boxsize')/1000 # h^-1 Mpc

    print('AVERAGING MODE: %s'%average_type)


    cat_name = 'data/cats/fits/TNG300-1_%d_non-reduced_galaxy_shapes.fits'%snap
    shape_cat = fi.FITS(cat_name)[1].read()
    ngal = len(shape_cat)

    sij = np.zeros((resolution,resolution,resolution,3,3))
    dsij = np.zeros((resolution,resolution,resolution,3,3))
    num = np.zeros((resolution,resolution,resolution,3,3))
    A = np.zeros((resolution,resolution,resolution,3,3))

    # global averages
    dI_global = np.zeros((3,3))
    I_global = np.zeros((3,3))

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

        I_TF = get_trace_free_matrix(I)


        #print(I)
       # import pdb ; pdb.set_trace()

        # add it to the correct cell
        if (average_type=='tensor'):
            sij[ix,iy,iz,:,:] += I_TF
            I_global += I_TF
        elif (average_type=='tensor_eigen'):
            sij[ix,iy,iz,:,:] += I
            I_global += I
        else:
            #import pdb ; pdb.set_trace()
            sij[ix,iy,iz,:,:] += v
            A[ix,iy,iz,:] += V


        num[ix,iy,iz,:,:] += 1

    import pdb ; pdb.set_trace()
    sij /= num
    sij[np.isinf(sij)]=-9999.
    sij[np.isnan(sij)]=-9999.

    I_global/=np.sum(num)

    for i in range(ngal):
        print ("Processing galaxy error %d/%d"%(i,ngal))

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

        I_TF = get_trace_free_matrix(I) 

        #print(I)
        

        # add it to the correct cell
        dI_global += (I_TF - I_global) * (I_TF - I_global) 
  

    
    dI_global /= np.sum(num)
    dI_global = np.sqrt(dI_global)

    for i in range(3):
        for j in range(3):
            dsij[:,:,:,i,j] = dI_global[i,j]

    #import pdb ; pdb.set_trace()


    if (average_type=='tensor_eigen'):
        for ix in range(resolution):
            for iy in range(resolution):
                for iz in range(resolution):
                    I0 = np.copy(sij[ix,iy,iz,:,:])
                    if (I0==-9999.).all():
                        continue
                    _,v = np.linalg.eigh(I0)
                    sij[ix,iy,iz,:,:] = v
                    #import pdb ; pdb.set_trace()

    del(shape_cat)
    del(X)
    del(Y)
    del(Z)
    del(B)
    del(lower)
    del(upper)
    gc.collect()

    import pdb ; pdb.set_trace()

    return sij, dsij

def gen_shape_cubes(snaps=[], resolution=512):
    """
    Outputs a fits file with the density cube for the corresponding snapshots
    """
    for sn in snaps:
        gamma, dgamma = build_shape_cube(sn, resolution=resolution)
        outfits = fi.FITS("stellar_shape_%03d_%d.fits"%(sn,resolution), 'rw')
        outfits2 = fi.FITS("stellar_shape_var_%03d_%d.fits"%(sn,resolution), 'rw')
        outfits.write(gamma)
        outfits.close()
        outfits2.write(dgamma)
        outfits2.close()
        #fits.writeto("stellar_shape_var_%03d.fits"%(sn), dgamma)

    print('Done all.')
    return None


def build_density_cube(snap, resolution=512, npart=600, ptype='dm'):
    """
    Computes a density cube for the specified snapshot
    """
    base_name = base_dir+ 'snapdir_%03d/snapshot_%03d'%(snap,snap)
    # This is hard coded, and will remain so until and unless anyone else needs to use it
   # basePath = "/Volumes/muskrat/other_peoples_datasets/illustrisTNG/TNG300-1/output/"
    basePath = "/home/rmandelb.proj/ssamurof/data/Illustris/TNG300-1/output/"

    # Read the size of the box in mpc
    boxsize = 205
    d = {'dm':1,'star':4}
    gName = "PartType" + str(d[ptype])

    i=0

    density = np.zeros((resolution,resolution,resolution))
    # Compute the density cube by reading each file one by one

    for j in range(npart):
        f = h5py.File(snapPath(basePath, snap, chunkNum=j), 'r')
        print(j)
        header = dict(f['Header'].attrs.items())
        ptcl_coords = f[gName]["Coordinates"][:]/1000.

        if ptype=='dm':
            Mp = None
        else:
            Mp = f[gName]['Masses'][:]

     #   import pdb ; pdb.set_trace()
        hist,edges = np.histogramdd(ptcl_coords, resolution, range=[[0,boxsize],[0,boxsize],[0,boxsize]], weights=Mp)
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
        os.system("rm %s_density_%03d_%d.fits"%(ptype,sn,resolution))
        outfits = fi.FITS("%s_density_%03d_%d.fits"%(ptype,sn,resolution),'rw')
        outfits.write(dens)
        outfits.close()

    print('Done all.')
    return None


def compute_tidal_tensor2(dens, smoothing=0.25, pixel_size=0.1953):
    """
    Computes the squared tidal tensor given a density field
    Pixel size and smoothing scale in h^{-1} Mpc
    """
    nx = dens.shape[0]
    dfilter = False 

    print('pixel scale = %3.3f'%pixel_size)

    k  = npf.fftfreq(nx)[np.mgrid[0:nx,0:nx,0:nx]]
    tidal_tensor = np.zeros((nx,nx,nx,3,3),dtype=np.float32)
    
    if dfilter:
        sigma = smoothing/pixel_size
        print('filtering, sigma=%3.3f'%sigma)
        G = gaussian_filter(dens,sigma,mode='wrap')
    else:
        print('not filtering')
        G = dens

    fft_dens = npf.fftn(G) # 3D (512 x 512 x 512) grid ; each cell is a k mode
    import pdb ; pdb.set_trace()
    for i in range(3):
        for j in range(3):
            
            # k[i], k[j] are 3D matrices, as is k
            temp =  fft_dens * k[i]*k[j]/(k[0]**2 + k[1]**2 + k[2]**2)

            # subtract off the trace...
            if (i==j):
                temp -= 1./3 * fft_dens
            temp[0,0,0] = 0

            tidal_tensor[:,:,:,i,j] = npf.ifftn(temp).real
            #import pdb ; pdb.set_trace()


#    for i in range(nx):
#        for j in range(nx):
#            for k in range(nx):
#                sij = tidal_tensor[i,j,k,:,:]
#                T = np.identity(3) * (1./3) * np.matrix.trace(sij)
#                tidal_tensor[i,j,k,:,:] = tidal_tensor[i,j,k,:,:] - T


    #import pdb ; pdb.set_trace()

    return tidal_tensor

def compute_tidal_tensor(dens, smoothing=0.25, pixel_size=0.1953):
    """
    Computes the tidal tensor given a density field
    Pixel size and smoothing scale in h^{-1} Mpc
    """
    nx = dens.shape[0]
    dfilter = False 

    norm = nx * nx * nx

    print('pixel scale = %3.3f'%pixel_size)
    

    #k  = npf.rfftfreq(nx, d=205./nx)[np.mgrid[0:nx,0:nx,0:nx]]
    k  = npf.fftfreq(nx, d=205./nx)[np.mgrid[0:nx,0:nx,0:nx]]
    tidal_tensor = np.zeros((nx,nx,nx,3,3),dtype=np.float32)
    
    if dfilter:
        sigma = smoothing/pixel_size
        print('filtering, sigma=%3.3f'%sigma)
        G = gaussian_filter(dens,sigma,mode='wrap')
    else:
        print('not filtering')
        G = dens + 1

    fft_dens = npf.fftn(G) / norm # 3D (512 x 512 x 512) grid ; each cell is a k mode
    #import pdb ; pdb.set_trace()
    for i in range(3):
        for j in range(3):
            
            # k[i], k[j] are 3D matrices, as is k
            #fft_dens = np.sqrt(fft_dens.real**2)

            temp = fft_dens * k[i]*k[j]/(k[0]**2 + k[1]**2 + k[2]**2)

            # subtract off the trace...
            if (i==j):
                temp -= 1./3 * fft_dens

            temp[0,0,0] = 0

            tidal_tensor[:,:,:,i,j] = npf.ifftn(temp).real * norm /nx
            #import pdb ; pdb.set_trace()


#    for i in range(nx):
#        for j in range(nx):
#            for k in range(nx):
#                sij = tidal_tensor[i,j,k,:,:]
#                T = np.identity(3) * (1./3) * np.matrix.trace(sij)
#                tidal_tensor[i,j,k,:,:] = tidal_tensor[i,j,k,:,:] - T



    return tidal_tensor

def gen_tidal_tensors(snaps=[], smoothing=[], ptype='dm', resolution=512, box_size=205., model='point'):
    """
    Computes the tidal tensor for the snapshots provided.
    The default box size is the right value for TNG300 - need to adjust if/when using other sims
    """

    A =  6.508621698085799e-4 # 4 * \pi * G * X, where X is the factor to convert an unnormalised 3D histogram
    # of TNG dark matter particles to comoving mass density  

    if model=='none':
        model_name=''
    else:
        model_name = '_%smodel'%model



    for i in snaps:
        dens = fi.FITS(dens_dir+'density/%s_density%s_%03d_%d.fits'%(ptype,model_name,i,resolution))[-1].read()
        K = np.mean(dens)
        #import pdb ; pdb.set_trace()
        #K = (box_size*1./resolution)**3
        #H0,b0=np.histogram(dens,bins=1000)
        #x0 = (b0[:-1]+b0[1:])/2
        #K = x0[H0==H0.max()][0]
        for s in smoothing:

            tid  = compute_tidal_tensor(dens/K -1 , smoothing=s, pixel_size=box_size/resolution)
            #import pdb ; pdb.set_trace()
            os.system('rm ' + dens_dir+'%s_tidal%s_%03d_%0.2f_%d.fits'%(ptype,model_name,i,s,resolution))
            out = fi.FITS(dens_dir+'%s_tidal%s_%03d_%0.2f_%d.fits'%(ptype,model_name,i,s,resolution),'rw')
            out.write(tid)
            out.close()

            # Diagonalise the tidal matrix while we are at it
            vals, vects = npl.eigh(tid)
            os.system('rm ' + dens_dir+'%s_tidal_vals%s_%03d_%0.2f_%d.fits'%(ptype,model_name,i,s,resolution))
            out = fi.FITS(dens_dir+'%s_tidal_vals%s_%03d_%0.2f_%d.fits'%(ptype,model_name,i,s,resolution),'rw') 
            out.write(vals)
            out.close()

            os.system('rm ' + dens_dir+'%s_tidal_vects%s_%03d_%0.2f_%d.fits'%(ptype,model_name,i,s,resolution))
            out = fi.FITS(dens_dir+'%s_tidal_vects%s_%03d_%0.2f_%d.fits'%(ptype,model_name,i,s,resolution),'rw') 
            out.write(vects)
            out.close()

    print('Done all.')
    return None

def snapPath(basePath, snapNum, chunkNum=0):
    """ Return absolute path to a snapshot HDF5 file (modify as needed). """
    snapPath = basePath + '/snapdir_' + str(snapNum).zfill(3) + '/'
    filePath = snapPath + 'snap_' + str(snapNum).zfill(3)
    filePath += '.' + str(chunkNum) + '.hdf5'
    return filePath
