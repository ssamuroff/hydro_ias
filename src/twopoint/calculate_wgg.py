import fitsio as fi
import treecorr
import numpy as np 
import os
import argparse
import yaml
from halotools.mock_observables.two_point_clustering import tpcf 
from halotools.mock_observables.two_point_clustering import wp 

periods={'massiveblackii':100, 'illustris':75}


def compute(options, binning, snapshot, ijk):

	posfile = options['positions'].replace(r'${SNAPSHOT}', '%d'%snapshot)

	print('Density data : %s'%posfile)

	positions = fi.FITS(posfile)[-1].read()
	split_type, positions = split_catalogue(positions,options,0)

	if (ijk>-1):
		pmask = get_jk_mask(ijk, options['njk'], options['box_size'], positions)
	else:
		pmask = np.ones(positions.size).astype(bool)

	rpbins = np.logspace(np.log10(options['rlim'][0]), np.log10(options['rlim'][1]), binning+1)

	print('Setting up correlations')

	# don't need randoms here if we know the period of the box
	print('Computing correlation functions.')

	filename = '%s/wgg_%d%s%s.txt'%(options['savedir'],snapshot,split_type,split_type)
	if (ijk>-1):
		filename = filename.replace('.txt', '-jk%d.txt'%ijk)

	if os.path.exists(filename):
		print('file exists: %s'%filename)
		return None

	#c0c0 = compute_gg_treecorr(positions[pmask], positions, options, binning, ijk)
	c0c0 = compute_gg(positions[pmask], positions, options, binning, ijk)
	export(filename, rpbins, c0c0)
		
	print('Done')

def randoms(cat1, cat2, ijk, period=100):
	# Initialise randoms
	# Fix the random seed so the randoms are always the same for a catalog of given length
	# Should probably migrate this to a config option
	np.random.seed(9000)
	f = 3.
	print('No. of Randoms = %3.1f x %d'%(f,cat1.size))

	# apply jackknife if necessary
	rdat = {}
	for k in ['x','y','z']: 
		rdat[k] = np.random.rand(int(cat1.size*f))*period
	if ijk>-1:
		rmask = get_jk_mask(ijk, options['njk'], options['box_size'], rcat1)
		for k in ['x','y','z']: 
			rdat[k] = rdat[k][rmask] 

	rcat11 = treecorr.Catalog(x=rdat['x'], y=rdat['y'], z=rdat['z'])
	rcat12 = treecorr.Catalog(x=np.random.rand(int(cat1.size*f))*period, y=np.random.rand(int(cat1.size*f))*period, z=np.random.rand(int(cat1.size*f))*period)
	rcat21 = treecorr.Catalog(x=np.random.rand(int(cat2.size*f))*period, y=np.random.rand(int(cat2.size*f))*period, z=np.random.rand(int(cat2.size*f))*period)
	rcat22 = treecorr.Catalog(x=np.random.rand(int(cat2.size*f))*period, y=np.random.rand(int(cat2.size*f))*period, z=np.random.rand(int(cat2.size*f))*period)

	return rcat11, rcat12, rcat21, rcat22

def randoms_halotools(cat1, period=100):
	# Initialise randoms
	# Fix the random seed so the randoms are always the same for a catalog of given length
	# Should probably migrate this to a config option
	np.random.seed(9000)
	rcat = np.array([np.random.rand(cat1.size*10)*period, np.random.rand(cat1.size*10)*period, np.random.rand(cat1.size*10)*period])
	return rcat.T

def finish_nn(corr, cat1, cat2, rcat1, rcat2, options, nbins):

	if cat2 is None:
		cat2 = cat1

	rr = treecorr.NNCorrelation(min_sep=options['2pt']['rmin'], max_sep=options['2pt']['rmax'], nbins=nbins)
	rn = treecorr.NNCorrelation(min_sep=options['2pt']['rmin'], max_sep=options['2pt']['rmax'], nbins=nbins)
	nr = treecorr.NNCorrelation(min_sep=options['2pt']['rmin'], max_sep=options['2pt']['rmax'], nbins=nbins)

	# Do the pair counting
	print('Processing randoms',)
	print('RR',)
	rr.process(rcat1,rcat2)
	print('DR',)
	nr.process(cat1,rcat2)
	print('RD',)
	rn.process(rcat1,cat2)

	# Finish off
	xi, var = corr.calculateXi(rr, dr=nr, rd=rn)
	setattr(corr, 'xi', xi)

	return corr

def export_treecorr_output(filename,corr,errors):
    R = np.exp(corr.logr)

    xi = corr.xi

    out = np.vstack((R,xi,corr.weight, errors))
    print('Saving %s'%filename)
    np.savetxt(filename, out.T)




def compute_gg_treecorr(cat1, cat2, options, nbins, ijk):

	print('Using treecorr', treecorr.version)

	slop = 0.1

	# arrays to store the output
	r    = np.zeros(nbins)
	DD   = np.zeros_like(r)
	DR   = np.zeros_like(r)
	RR   = np.zeros_like(r)
	RD   = np.zeros_like(r)


	# Set up the catalogues
	rcat1, rcat2, _, _ = randoms(cat1, cat2, ijk, period=options['box_size'])
	#import pdb ; pdb.set_trace()

	cat1  = treecorr.Catalog(g1=None, g2=None, x=cat1['x'], y=cat1['y'], z=cat1['z'])
	cat2  = treecorr.Catalog(g1=None, g2=None, x=cat2['x'], y=cat2['y'], z=cat2['z'])

	NR1 = rcat1.x.size * 1.0
	NR2 = rcat2.x.size * 1.0
	ND1 = cat1.x.size * 1.0
	ND2 = cat2.x.size * 1.0

	f0 = (NR1*NR2)/(ND1*ND2)
	f1 = (NR1*NR2)/(ND1*NR2)
	f2 = (NR1*NR2)/(ND2*NR1)

	print('Processing DD')
	nn = treecorr.NNCorrelation(nbins=nbins, min_sep=options['rlim'][0], max_sep=options['rlim'][1], bin_slop=slop, verbose=0, period=options['box_size'])
	#nn.process(rcat1, rcat2, period=options['box_size'], metric='Periodic')
	nn.process(cat1, cat2, metric='Periodic')
	nn.finalize()
	DD = np.copy(nn.weight)
	rp_bins = np.copy(nn.rnom)
	nn.clear()

	print('Processing RD')
	nn = treecorr.NNCorrelation(nbins=nbins, min_sep=options['rlim'][0], max_sep=options['rlim'][1], bin_slop=slop, verbose=0, period=options['box_size'])
	nn.process(rcat1, cat2, metric='Periodic')
	RD = np.copy(nn.weight)
	nn.clear()

	print('Processing DR')
	nn = treecorr.NNCorrelation(nbins=nbins, min_sep=options['rlim'][0], max_sep=options['rlim'][1], bin_slop=slop, verbose=0, period=options['box_size'])
	nn.process(cat1, rcat2, metric='Periodic')
	DR = np.copy(nn.weight)
	nn.clear()

	print('Processing RR')
	nn = treecorr.NNCorrelation(nbins=nbins, min_sep=options['rlim'][0], max_sep=options['rlim'][1], bin_slop=slop, verbose=0, period=options['box_size'])
	nn.process(rcat1, rcat2, metric='Periodic')
	RR = np.copy(nn.weight)
	nn.clear()

	gg = (f0 * DD/RR) - (f1 * DR/RR) - (f2 * RD/RR) + 1.0	

	return gg



def compute_gg(cat1, cat2, options, nbins, ijk):
	pvec1 = np.vstack((cat1['x'], cat1['y'], cat1['z'])).T
	pvec2 = np.vstack((cat2['x'], cat2['y'], cat2['z'])).T
	rbins = np.logspace(np.log10(options['rlim'][0]), np.log10(options['rlim'][1]), nbins+1 )

	pi_max = options['rlim'][1]

	

	if len(cat1)==len(cat2):
		gg = wp(pvec2, rbins, pi_max, sample2=pvec1, period=options['box_size'], num_threads=1, estimator='Landy-Szalay') 
	else:
		# wp returns w_11, w_12, w22 
		_,gg,_ = wp(pvec2, rbins, pi_max, sample2=pvec1, period=options['box_size'], num_threads=1, estimator='Landy-Szalay') 
	#period=periods[options['simulation']]
	#import pdb ; pdb.set_trace()

	return gg


def export(path, edges, result):
	x = np.sqrt(edges[:-1]*edges[1:])
	out = np.vstack((x, result))
	print('Exporting to %s'%path)
	np.savetxt(path, out.T)



def get_jk_mask(ijk, njk, box_size, catalogue):
	j = 0
	labels = np.zeros_like(catalogue['x'])
	patch_size = box_size/njk

	for ix in range(njk):
		xmask = (catalogue['x'] >= patch_size*ix) & (catalogue['x'] < patch_size*(ix+1)) 
		for iy in range(njk):
			ymask = (catalogue['y'] >= patch_size*iy) & (catalogue['y'] < patch_size*(iy+1)) 
			for iz in range(njk):
				zmask = (catalogue['z'] >= patch_size*iz) & (catalogue['z'] < patch_size*(iz+1)) 
				mask_tot = xmask & ymask & zmask
				labels[mask_tot] = j
				j+=1

	mask = (labels!=ijk)
	print('mask excludes %d/%d'%(labels[np.invert(mask)].size, labels.size))

	return mask

def split_catalogue(cat, options, i):

	if (not ('split_type' in options.keys())):
		return '', cat
	else:
		s = options['split_type'].split()[i]

	if (s=='central'):
		mask = (cat['gal_id']==cat['central_id'])
	elif (s=='satellite'):
		mask = np.invert(cat['gal_id']==cat['central_id'])

	elif (s=='stellar_mass_high'):
		Ms = np.median(cat['stellar_mass_all'])
		mask = (cat['stellar_mass_all']>=Ms)

	elif (s=='stellar_mass_low'):
		Ms = np.median(cat['stellar_mass_all'])
		mask = (cat['stellar_mass_all']<Ms)
	else:
		raise ValueError('Unrecognised split type:',s)

	print('Split leaves %d objects'%(cat['x'][mask].size))

	return '_%s'%options['split_type'].split()[i], cat[mask]
