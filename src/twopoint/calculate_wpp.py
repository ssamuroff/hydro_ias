import fitsio as fi
import treecorr
import numpy as np
import os
import argparse
import yaml
from halotools.mock_observables.alignments import ii_plus_projected

period={'massiveblackii':100, 'illustris':75}



def compute(options, binning, snapshot, ijk):


	shapefile = options['shapes'].replace(r'${SNAPSHOT}', '%d'%snapshot)

	print('Shape data : %s'%shapefile)

	shapes = fi.FITS(shapefile)[-1].read()

	split_type, shapes = split_catalogue(shapes, options,1)

	if (ijk>-1):
		smask = get_jk_mask(ijk, options['njk'], options['box_size'], shapes)
	else:
		smask = np.ones(shapes.size).astype(bool)


	rpbins = np.logspace(np.log10(options['rlim'][0]), np.log10(options['rlim'][1]), binning+1)

	print('Setting up correlations')

	# don't need randoms here if we know the period of the box
	print('Computing correlation functions.')

	c0c0 = compute_ii(shapes[smask], shapes, options, rpbins=rpbins, period=options['box_size'], nbins=binning)

	filename = '%s/wpp_%d%s%s.txt'%(options['savedir'],snapshot,split_type,split_type)
	if (ijk>-1):
		filename = filename.replace('.txt', '-jk%d.txt'%ijk)

	if os.path.exists(filename):
		print('file exists: %s'%filename)
		return None

	export_array(filename, rpbins, c0c0)
		
	print('Done')



def compute_ii(cat1, cat2, options, period=100., rpbins=None, nbins=6):

	aname = 'av_%c'
	ename = 'e%d'

	avec1 = np.vstack((cat1[aname%'x'], cat1[aname%'y'])).T
	avec2 = np.vstack((cat2[aname%'x'], cat2[aname%'y'])).T
	pvec1 = np.vstack((cat1['x'], cat1['y'], cat1['z'])).T
	pvec2 = np.vstack((cat2['x'], cat2['y'], cat2['z'])).T
	evec1 = np.sqrt(cat1[ename%1]*cat1[ename%1] + cat1[ename%2]*cat1[ename%2])
	evec2 = np.sqrt(cat2[ename%1]*cat2[ename%1] + cat2[ename%2]*cat2[ename%2])

	
	rpbins = np.logspace(np.log10(options['rlim'][0]), np.log10(options['rlim'][1]), nbins+1)
	
	pi_max = options['rlim'][1]

	mask1=avec1.T[0]!=0.0
	mask2=avec2.T[0]!=0.0

	ii = ii_plus_projected(pvec2[mask2], avec2[mask2], evec2[mask2], pvec1[mask1], avec1[mask1], evec1[mask1], rpbins, pi_max, period=period, num_threads=1) 

	return ii

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
	else:
		raise ValueError('Unrecognised split type:',s)

	print('Split leaves %d objects'%(cat['x'][mask].size))

	return '_%s'%options['split_type'].split()[i], cat[mask]
