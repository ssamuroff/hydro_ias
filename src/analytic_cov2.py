import numpy as np
import sys
import os
import fitsio as fi
import pylab as plt
plt.switch_backend('pdf')
plt.style.use('y1a1')
from scipy.interpolate import interp1d 
from scipy.interpolate import interp2d 
from scipy.integrate import quad as scipy_int1d
import scipy.integrate as sint
from src import hankel_transform as sukhdeep
from src import wigner_transform as wigner
from src import power_spectra as pkutil
from astropy.cosmology import FlatLambdaCDM

cosmologies ={'mbii': FlatLambdaCDM(H0=69., Om0=0.30, Ob0=0.048), 'tng':FlatLambdaCDM(H0=67.74, Om0=0.3089, Ob0=0.0486), 'illustris':FlatLambdaCDM(H0=70.04, Om0=0.2726, Ob0=0.0456)}

kernels = {'gg':0,'gp':2,'pp':(0,4)}

class Cov:
	def __init__(self, params):
		fits = fi.FITS(params['datafile'])
		self.fits = fits
		self.settings = params
		self.nr = params['nbins']

		self.area = params['box_size']**2
		self.z = np.array(params['redshifts'].split()).astype(float)
		self.cosmology = cosmologies[self.settings['cosmology']]

		self.initialise_block(fits)

		self.wigner_transforms = {}
		for x in self.z:
			self.wigner_transforms[x] = None

		print('Setting up Hankel transform')
		print("This initialisation step takes a little while, but the integration is quick once it's done.")
		self.transform = sukhdeep.hankel_transform(rmin=self.settings['rlim'][0]*0.95, rmax=self.settings['rlim'][1]+10, kmin=self.settings['kmin'], kmax=self.settings['kmax'], j_nu=[0,2,4], n_zeros=9100)

		print('Done')
		return None


	def initialise_block(self, fits):
		ndim = 0
		for name in self.settings['correlations'].split():
			ndim+=len(fits[name]['VALUE'].read())

		self.block = np.zeros((ndim,ndim)) - 9999.
		print('Initialised %dx%d covariance matrix'%(ndim,ndim))

		self.settings['rmin'] = fits['wgg']['SEP'][:].min()
		self.settings['rmax'] = fits['wgg']['SEP'][:].max()
		return None


	def get_sigmae(self,i):
		s = float(np.atleast_1d(self.settings['sigma_e'].split())[i])
		return s

	def get_ng(self,i,m):
		#import pdb ; pdb.set_trace()
		#nlist = self.settings['ng'].split()
		if m=='g':
			j=0
		else:
			j=1

		s = np.atleast_1d(self.settings['ng'].split())[i].replace('(','').replace(')','').split(',')[j]
		s = float(s)

		#s = float(np.atleast_1d(self.settings['ng'].split())[i])
		return s


	def get_bias(self,pk_name,i):
		bias = float(np.atleast_1d(self.settings['bias'].split())[i])
		if pk_name=='intrinsic_power':
			return 1.
		elif pk_name=='galaxy_intrinsic_power':
			return bias
		elif pk_name=='galaxy_power':
			return bias * bias

	def get_aia(self,pk_name,i):
		A = float(np.atleast_1d(self.settings['aia'].split())[i])
		if pk_name=='galaxy_power':
			return 1.
		elif pk_name=='galaxy_intrinsic_power':
			return A
		elif pk_name=='intrinsic_power':
			return A * A

	def choose_noise(self, m1, m2, z1, z2):

		s = self.get_sigmae(z1)
		ng = self.get_ng(z1,m1)

		if (m1!=m2):
			return 0

		elif (m1=='g'):
			return 1./ng

		elif (m1=='p'):
			return s*s/ng

	def get_pk(self, meas1, meas2, zbin1, zbin2):

		def get_pk_name(m1,m2):
			name = '%c%c'%(m1,m2)
			if (name=='gg'):
				return 'galaxy_power'
			elif (name=='gp') or (name=='pg'):
				return 'galaxy_intrinsic_power'
			elif (name=='pp'):
				return 'intrinsic_power'
			else:
				print('Unrecognised power spectrum type:',name)

		# load the theory power spectrum
		# this should be a 2D grid, nk x nz
		pk_name = get_pk_name(meas1,meas2)
		bg = self.get_bias(pk_name, zbin1)
		A_IA = self.get_aia(pk_name, zbin1)

		P = np.loadtxt('data/pk/%s/%s/p_k.txt'%(self.settings['cosmology'],pk_name)) * bg * A_IA
		k = np.loadtxt('data/pk/%s/%s/k_h.txt'%(self.settings['cosmology'],pk_name))
		z = np.loadtxt('data/pk/%s/%s/z.txt'%(self.settings['cosmology'],pk_name))

#		if zbin1!=zbin2:
#			return k, np.zeros_like(P[0])

		N = self.choose_noise(meas1,meas2,zbin1,zbin2)

		D = abs(z-self.z[zbin1])
		ind = np.argwhere(D==D.min())[0,0]
		Pw = P[ind,:] + N

		return k, Pw

#		if (P.min()>0):
#			logint = True
#			lnpk = np.log(P)
#		else:
#			#print('Negative Pk values - will interpolate in linear space.')
#			logint = False
#			lnpk = P
#
#		interp_pk = interp2d(np.log(k), z, lnpk, kind='linear')
#
#		Pw = interp_pk(np.log(k), self.z[zbin1])
#
#		if logint:
#			#print('logint')
#			Pw = 10**Pw
#			import pdb ; pdb.set_trace()
#
#		Pw+=N


#		return k, Pw

	def evaluate_block_wigner(self, r1, r2, a, b, c, d, P1, P2, P3, P4, k, z0):
		nu = kernels['%c%c'%(c,d)]
		mu = kernels['%c%c'%(a,b)]
		resampled_block = np.zeros((self.nr,self.nr))

		if z==0:
			chi = self.cosmology.comoving_transverse_distance(z=0.05).value
		else:
			chi = self.cosmology.comoving_transverse_distance(z=z0).value

		ell = np.unique(np.int32(self.transform.k[0]*chi))
		theta=self.transform.r[0]/chi
		WT_kwargs={'l': ell,'theta': theta,'s1_s2':[(0,2),(2,0),(0,0)]} #(2,2),(2,-2)

		if self.wigner_transforms[z0] is None:
			WT = wigner_transform(**WT_kwargs)
			self.wigner_transforms[z0] = WT

		Ap = self.area
		chi = self.cosmology.comoving_transverse_distance(z=z0).value
		import pdb ; pdb.set_trace()

		if (nu==mu) and (nu==2):
			rW,cov_ggkkW = self.wigner_transforms[z0].projected_covariance(l_cl=k*chi,cl_cov=P1*P2/chi**2,s1_s2=(0,2))


#
#rW=rW*chi
#r_reW,cov_ggkk_reW=HT.bin_cov(r=rW,cov=cov_ggkkW,r_bins=r_bins)
## corr=HT.corr_matrix(cov=cov_ggkk_reW)
#
#
#rW,cov_gkgkW=WT.projected_covariance(l_cl=kh*chi,cl_cov=p_gk_cov*p_gk_cov/chi**2,s1_s2=(0,2))
#rW=rW*chi
#r_reW,cov_gkgk_reW=HT.bin_cov(r=rW,cov=cov_gkgkW,r_bins=r_bins)
## corr=HT.corr_matrix(cov=cov_gkgk_reW)
#
#
#cov_finalW=(cov_ggkk_reW+cov_gkgk_reW)/area_comoving
#corrW=HT.corr_matrix(cov=cov_finalW)
#errorsW=HT.diagonal_err(cov=cov_finalW)




		
		
		

		
		

		return resampled_block/Ap





	def evaluate_block(self, r1, r2, a, b, c, d, P1, P2, P3, P4, k):
		nu = kernels['%c%c'%(c,d)]
		mu = kernels['%c%c'%(a,b)]
		resampled_block = np.zeros((self.nr,self.nr))

		# call Sukhdeep's code to generate the relevant block of the covariance matrix
		# resample it to the desired rp binning
		if (nu==mu):
			for index in np.atleast_1d(nu):
				r_cov,block = self.transform.projected_covariance(k_pk=k, pk1=P1, pk2=P2, j_nu=index, taper=True)
				r_cov2,block2 = self.transform.projected_covariance(k_pk=k, pk1=P3, pk2=P4, j_nu=index, taper=True)

				# downsample to the scale bins required
				r_edges1, C1 = self.transform.bin_cov(r=r_cov, cov=block, bin_center=r1)
				r_edges2, C2 = self.transform.bin_cov(r=r_cov2, cov=block2, bin_center=r2)
				#import pdb ; pdb.set_trace()
				resampled_block+=C1+C2



		Ap = self.area
		import pdb ; pdb.set_trace()
		

		
		

		return resampled_block/Ap




	def find_rp(self, corr, zbin):
		# get the exact bin centres for a particular 2pt function from the FITS file

		hdu = self.fits[corr].read()

		# select the array section relevant for this correlation
		zmask = (hdu['BIN']==zbin)
		r = hdu['SEP'][zmask]

		return r

	def bin_cov(self,r=[],cov=[],bin_center=[]):
		#bin_center = np.sqrt(r_bins[1:]*r_bins[:-1])
		dr = np.mean(bin_center[1:]/bin_center[:-1])
		r_bins = np.append(bin_center[0]/np.sqrt(dr) , [x0*np.sqrt(dr) for x0 in bin_center])
		n_bins=len(bin_center)
		cov_int=np.zeros((n_bins,n_bins),dtype='float64')
		bin_idx=np.digitize(r,r_bins)-1
		r2=np.sort(np.unique(np.append(r,r_bins))) #this takes care of problems around bin edges
		dr=np.gradient(r2)
		r2_idx=[i for i in np.arange(len(r2)) if r2[i] in r]
		dr=dr[r2_idx]
		r_dr=r*dr
		cov_r_dr=cov*np.outer(r_dr,r_dr)
		for i in np.arange(min(bin_idx),n_bins):
			xi=bin_idx==i
			for j in np.arange(min(bin_idx),n_bins):
				xj=bin_idx==j
				norm_ij=np.sum(r_dr[xi])*np.sum(r_dr[xj])
				if norm_ij==0:
					continue
				cov_int[i][j]=np.sum(cov_r_dr[xi,:][:,xj])/norm_ij
		return bin_center,cov_int




	def write(self, i0, j0, cosvar):
		dx = len(cosvar[0])
		dy = len(cosvar.T[0])

		nx = len(self.block[0])

		if i0>=nx:
			i0=0
			j0+=dy

		self.block[j0:j0+dy, i0:i0+dx] = cosvar
		self.block[i0:i0+dx, j0:j0+dy] = cosvar.T


		i0+=dx

		return i0,j0

	def save(self):
		filename = '%s-covmat-analytic.txt'%self.settings['cosmology']
		np.savetxt(filename, self.block)
		return None



def run(params):
	covmat = Cov(params)
	done = []
	i0,j0 = 0,0

	fits = fi.FITS(params['datafile'])
	for c1 in covmat.settings['correlations'].split():	

		for z1 in fits[c1]['BIN'].read():

			for c2 in covmat.settings['correlations'].split():

				for z2 in fits[c2]['BIN'].read():
					if ('%s%d'%(c1,z1)+'%s%d'%(c2,z2)) in done:
						continue

					k, P1 = covmat.get_pk(c1[1:][0],c2[1:][0], z1, z2)
					k, P2 = covmat.get_pk(c1[1:][1],c2[1:][1], z1, z2)
					k, P3 = covmat.get_pk(c1[1:][0],c2[1:][1], z1, z2)
					k, P4 = covmat.get_pk(c1[1:][1],c2[1:][0], z1, z2)

					a,b = c1[1:][0], c1[1:][1]
					c,d = c2[1:][0], c2[1:][1]

					r1 = covmat.find_rp(c1,z1)
					r2 = covmat.find_rp(c2,z2)

					if covmat.settings['wigner']:
						B = covmat.evaluate_block_wigner(r1, r2, a, b, c, d, P1, P2, P3, P4, k, z1) 
					else:
						B = covmat.evaluate_block(r1, r2, a, b, c, d, P1, P2, P3, P4, k)

					i0,j0 = covmat.write(i0,j0,B)
					print(c1,c2,i0,j0)
					done.append('%s%d'%(c1,z1)+'%s%d'%(c2,z2))
					

	print('Saving...')
	covmat.save()
	print('Done all.')


def get_sigma_e(cat, area):
        """
        Calculate sigma_e for shape catalog.
        """

        e1  = cat['e1']
        e2  = cat['e2']
        w   = np.ones_like(cat['e1'])

        snvar = 0.24#np.sqrt((cat['e1'][mask][cat['snr'][mask]>100].var()+cat['e2'][mask][cat['snr'][mask]>100].var())/2.)
        var = 1./w - snvar**2
        var[var < 0.] = 0.
        s = 1.
        w[w > snvar**-2] = snvar**-2
        print( 'var',var.min(),var.max())

        mean_e1 = np.asscalar(np.average(cat['e1'],weights=w))
        mean_e2 = np.asscalar(np.average(cat['e2'],weights=w))

        a1 = np.sum(w**2 * (e1-mean_e1)**2)
        a2 = np.sum(w**2 * (e2-mean_e2)**2)
        b  = np.sum(w**2)
        c  = np.sum(w * s)

        d  = np.sum(w)

        sigma_e =  np.sqrt( (a1/c**2 + a2/c**2) * (d**2/b) / 2. ) 
        sigma_ec = np.sqrt( np.sum(w**2 * (e1**2 + e2**2 - var)) / (2.*np.sum(w**2 * s**2)) ) 

        return sigma_e, sigma_ec, mean_e1, mean_e2
