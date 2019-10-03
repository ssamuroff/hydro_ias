from pygadgetreader import *
import numpy as np
import numpy.linalg as npl
import numpy.fft as npf
from scipy.ndimage import gaussian_filter
import pyfits as fits
#from mb2 import *
from numpy.core.records import fromarrays
import fitsio as fi
import gc

import pylab as plt
plt.switch_backend('agg')
plt.style.use('y1a1')


base_dir = '/physics/nkhandai/mb2/snapdir/'
dens_dir = '/home/rmandelb.proj/ssamurof/mb2_tidal/'



def build_shape_cube(snap, resolution=512, npart=1024):
    """
    Computes a density cube for the specified snapshot
    """
    base_name = base_dir+ 'snapdir_%03d/snapshot_%03d'%(snap,snap)

    # Read the size of the box in mpc
    boxsize = readheader(base_name, 'boxsize')/1000 # h^-1 Mpc


    cat_name = 'data/cats/fits/MBII_%d_non-reduced_galaxy_shapes.fits'%snap
    shape_cat = fi.FITS(cat_name)[1].read()
    ngal = len(shape_cat)

    sij = np.zeros((resolution,resolution,resolution,3,3))
    sigma = np.zeros((resolution,resolution,resolution,3,3))
    num = np.zeros((resolution,resolution,resolution))

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
        a0 = np.array([shape_cat['av_x'][i],shape_cat['av_y'][i],shape_cat['av_z'][i]]) * shape_cat['a'][i]
        b0 = np.array([shape_cat['bv_x'][i],shape_cat['bv_y'][i],shape_cat['bv_z'][i]]) * shape_cat['b'][i]
        c0 = np.array([shape_cat['cv_x'][i],shape_cat['cv_y'][i],shape_cat['cv_z'][i]]) * shape_cat['c'][i]
        I = np.array([a0,b0,c0])

        # add it to the correct cell
        sij[ix,iy,iz,:,:] += I.T
        num[ix,iy,iz]+=1

    for i in range(3):
        for j in range(3):
            sij[:,:,:,i,j] /= num
 #           sigma[:,:,:,i,j] /= num

    sij[np.isinf(sij)]=0
    sij[np.isnan(sij)]=0

   # import pdb ; pdb.set_trace()

    # also process the shape dispersion per bin
    for i in range(ngal):
        print ("Processing sigma for galaxy %d/%d"%(i,ngal))

        # put this galaxy into a cell
        info = shape_cat[i]
        ix = np.argwhere((shape_cat['x'][i]>=lower) & (shape_cat['x'][i]<upper))[0,0]
        iy = np.argwhere((shape_cat['y'][i]>=lower) & (shape_cat['y'][i]<upper))[0,0]
        iz = np.argwhere((shape_cat['z'][i]>=lower) & (shape_cat['z'][i]<upper))[0,0]

        # reconstruct the 3x3 shape matrix for this galaxy
        a0 = np.array([shape_cat['av_x'][i],shape_cat['av_y'][i],shape_cat['av_z'][i]]) * shape_cat['a'][i]
        b0 = np.array([shape_cat['bv_x'][i],shape_cat['bv_y'][i],shape_cat['bv_z'][i]]) * shape_cat['b'][i]
        c0 = np.array([shape_cat['cv_x'][i],shape_cat['cv_y'][i],shape_cat['cv_z'][i]]) * shape_cat['c'][i]
        I = np.array([a0,b0,c0])

        # add it to the correct cell
        sigma[ix,iy,iz,:,:] += (I-sij[ix,iy,iz,:,:]) * (I-sij[ix,iy,iz,:,:])


    del(shape_cat)
    #del(num)
    del(X)
    del(Y)
    del(Z)
    del(B)
    del(lower)
    del(upper)
    gc.collect()

    for i in range(3):
        for j in range(3):
            sigma[:,:,:,i,j] /= num
            sigma[:,:,:,i,j] = np.sqrt(sigma[:,:,:,i,j])

    sigma[np.isinf(sigma)]=0
    sigma[np.isnan(sigma)]=0

    return sij, sigma

def gen_shape_cubes(snaps=[]):
    """
    Outputs a fits file with the density cube for the corresponding snapshots
    """
    for sn in snaps:
        gamma, dgamma = build_shape_cube(sn)
        fits.writeto("stellar_shape_%03d.fits"%(sn), gamma)
        fits.writeto("stellar_shape_var_%03d.fits"%(sn), dgamma)


def build_density_cube(snap, resolution=512, npart=1024, ptype='dm'):
    """
    Computes a density cube for the specified snapshot
    """
    base_name = base_dir+ 'snapdir_%03d/snapshot_%03d'%(snap,snap)

    # Read the size of the box in mpc
    boxsize = readheader(base_name, 'boxsize')


    density = np.zeros((resolution,resolution,resolution))
    # Compute the density cube by reading each file one by one
    for i in range(npart):
        print ("Processing part %d/1024"%i) 
        data = readsnap(base_name+'.%d'%i,'pos', ptype,single=1)
        hist,edges = np.histogramdd(data, resolution, range=[[0,boxsize],[0,boxsize],[0,boxsize]])
        density = density + hist

    return density

def gen_density_cubes(snaps=[], ptype='dm'):
    """
    Outputs a fits file with the density cube for the corresponding snapshots
    """
    for sn in snaps:
        dens = build_density_cube(sn, ptype=ptype)
        fits.writeto("%s_density_%03d.fits"%(pytype,sn), dens)

def compute_tidal_tensor(dens, smoothing=0.25, pixel_size=0.1953):
    """
    Computes the tidal tensor given a density field
    Pixel size and smoothing scale in h^{-1} Mpc
    """
    nx = dens.shape[0]
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

def gen_tidal_tensors(snaps=[], smoothing=[], ptype='dm'):
    """
    Computes the tidal tensor for the snapshots provided
    """
    for i in snaps:
        dens = fits.getdata(dens_dir+'%s_density_%03d.fits'%(ptype,i))
        for s in smoothing:
            tid  = compute_tidal_tensor(dens,smoothing=s)
            fits.writeto(dens_dir+'%s_tidal_%03d_%0.2f.fits'%(ptype,i,s), tid)
            # Diagonalise the tidal matrix while we are at it
            vals, vects = npl.eigh(tid)
            fits.writeto(dens_dir+'%s_tidal_vals_%03d_%0.2f.fits'%(ptype,i,s), vals)
            fits.writeto(dens_dir+'%s_tidal_vects_%03d_%0.2f.fits'%(ptype,i,s), vects)

def load_tidal_matrix(snaps=[], smoothing=[]):
    # Connect to the database
    db = mbdb(dbname='mb2_hydro')

    for s in smoothing:
        name = "tidal_%d"%(s * 1000)
        # Create insert statement for database transaction
        stmt = "INSERT INTO "+name+" VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        data = []

        # Obtain a cursor
        curs = db.db.cursor()

        for sid in snaps:
            # Load the tidal matrix eigenvalues and eigenvectors
            vals  = fits.getdata(dens_dir+'tidal_vals_%03d_%0.2f.fits' %(sid,s))
            vects = fits.getdata(dens_dir+'tidal_vects_%03d_%0.2f.fits'%(sid,s))

            # Get the x,y,z position of all galaxies in the subfind snapshot
            curs.execute("select subfindId, x, y, z from mb2_hydro.subfind_halos where snapnum = %d"%sid)
            res = fromarrays(np.array(curs.fetchall()).squeeze().T,names="subfindId, x, y, z")
            x = np.digitize(res['x'], np.linspace(0,100000,512,endpoint=False)) - 1
            y = np.digitize(res['y'], np.linspace(0,100000,512,endpoint=False)) - 1
            z = np.digitize(res['z'], np.linspace(0,100000,512,endpoint=False)) - 1
            eigvals = vals[x,y,z]
            eigvects = vects[x,y,z]

            data = []
            for i in range(len(res)):

                data.append((res['subfindId'][i].astype('int64'), eigvals[i][2], eigvals[i][1], eigvals[i][0],
                             eigvects[i][0,2], eigvects[i][1,2], eigvects[i][2,2],
                             eigvects[i][0,1], eigvects[i][1,1], eigvects[i][2,1],
                             eigvects[i][0,0], eigvects[i][1,0], eigvects[i][2,0]))

                if (i % 10000 == 0) or i > len(res) - 10:
                    print ("Processing %d"%i)

                    try:
                        curs.executemany(stmt, data)

                        # Commit changes to the DB
                        db.db.commit()
                    except:
                        # Rollback in case of error
                        db.db.rollback()
                        print("Error happened at entry %i"%i)
                        break
                    data = []
