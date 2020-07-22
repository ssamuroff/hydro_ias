import fitsio as fi
import treecorr
import numpy as np 
import os
import argparse
import yaml
from halotools_ia.correlation_functions import gi_plus_projected

period={'massiveblackii':100, 'illustris':75}

def compute(options, binning, snapshot, ijk):

	shapefile = options['shapes'].replace(r'${SNAPSHOT}', '%d'%snapshot)
	posfile = options['positions'].replace(r'${SNAPSHOT}', '%d'%snapshot)

	print('Shape data : %s'%shapefile)
	print('Density data : %s'%posfile)

	shapes = fi.FITS(shapefile)[-1].read()
	positions = fi.FITS(posfile)[-1].read()

	split_type1,shapes = split_catalogue(shapes,options, 1)
	split_type2,positions = split_catalogue(positions,options, 0)


	if 'weights' in options:
		print('Applying weights from file',options['weights'])
		wts = np.loadtxt(options['weights'])
		wts/=np.sum(wts)
		wts *= len(wts)
		apply_weights=True
	else:
		wts1 = np.ones_like(shapes['x'])
		wts2 = np.ones_like(positions['x'])
		apply_weights=False

	filename = '%s/wgp_%d%s%s.txt'%(options['savedir'],snapshot,split_type1,split_type2)
	if apply_weights:
		filename = filename.replace('.txt', '-weighted.txt')
	if (ijk>-1):
		filename = filename.replace('.txt', '-jk%d.txt'%ijk)

	if os.path.exists(filename):
		print('file exists: %s'%filename)
		return None

	
	#import pdb ; pdb.set_trace()

	if (ijk>-1):
		pmask = get_jk_mask(ijk, options['njk'], options['box_size'], positions)
	else:
		pmask = np.ones(positions.size).astype(bool)


	rpbins = np.logspace(np.log10(options['rlim'][0]), np.log10(options['rlim'][1]), binning+1)

	print('Setting up correlations')

	# don't need randoms here if we know the period of the box
	print('Computing correlation functions.')

	c0c0 = compute_gi(positions[pmask], shapes, options, rpbins=rpbins, period=options['box_size'], nbins=binning, weights1=wts1, weights2=wts2[pmask])

	
	export_array(filename, rpbins, c0c0)
		
	print('Done')

def compute_gi(cat1, cat2, options, period=100., rpbins=None, nbins=6, weights1=None, weights2=None):

	aname = 'av_%c'
	ename = 'e%d'

	avec = np.vstack((cat2[aname%'x'], cat2[aname%'y'])).T
	pvec1 = np.vstack((cat1['x'], cat1['y'], cat1['z'])).T
	pvec2 = np.vstack((cat2['x'], cat2['y'], cat2['z'])).T
	evec = np.sqrt(cat2[ename%1]*cat2[ename%1] + cat2[ename%2]*cat2[ename%2])

	rpbins = np.logspace(np.log10(options['rlim'][0]), np.log10(options['rlim'][1]), nbins+1)
	
	pi_max = options['rlim'][1]
	#import pdb ; pdb.set_trace()

	mask2=avec.T[0]!=0.0


	gip = gi_plus_projected(pvec2[mask2], avec[mask2], evec[mask2], pvec1, rpbins, pi_max, period=period, num_threads=1, weights1=weights1, weights2=weights2) 

	return gip

def export_array(path, edges, result):
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
        elif (s=='all'):
                mask = np.ones_like(cat['gal_id']).astype(bool)
	else:
		raise ValueError('Unrecognised split type:',s)

	print('Split leaves %d objects'%(cat['x'][mask].size))

	return '_%s'%options['split_type'].split()[i], cat[mask]
