import fitsio as fi
import treecorr
import numpy as np
import os
import argparse
import yaml
from halotools.mock_observables.alignments import ii_plus_projected

period={'massiveblackii':100, 'illustris':75}
colour_split_params = {99 : (0.045, 1.84), 78: (0.045, 1.84), 62: (0.055, 1.95), 50: (0.022, 1.19)}

auto=True

def compute(options, binning, snapshot, ijk):


	shapefile = options['shapes'].replace(r'${SNAPSHOT}', '%d'%snapshot)

	print('Shape data : %s'%shapefile)

	shapes = fi.FITS(shapefile)[-1].read()
	split_type, shapes = split_catalogue(shapes, options, snapshot, 1)

	if auto:
		shapes2 = shapes
		split_type2 = split_type
	else:
		shapes2 = fi.FITS(shapefile)[-1].read()
		split_type2, shapes2 = split_catalogue(shapes2, options, snapshot, 0)
		print('Warning: using hacked cross correlation mode')

	if (ijk>-1):
		smask = get_jk_mask(ijk, options['njk'], options['box_size'], shapes)
	else:
		smask = np.ones(shapes.size).astype(bool)


	if ('use_weights' in options):
		if options['use_weights']:

			print('Applying weights from file:')
			base = os.path.dirname(shapefile.replace('/fits/','/txt/') )
			sim, s, _, _, _ = os.path.basename(shapefile).split('_')
			weightsfile = base + '/%s_%s_weights.dat'%(sim,s)
			print(weightsfile)

			wts1 = np.loadtxt(weightsfile)
			wts2 = np.loadtxt(weightsfile)
			wts1/=np.sum(wts1)
			wts1 *= len(wts1)
			wts2/=np.sum(wts2)
			wts2 *= len(wts2)
			apply_weights = True
		else:
			apply_weights = False
	else:
		apply_weights = False

	if (not apply_weights):
		wts1 = np.ones_like(shapes2['x'])
		wts2 = np.ones_like(shapes['x'])
		apply_weights=False


	rpbins = np.logspace(np.log10(options['rlim'][0]), np.log10(options['rlim'][1]), binning+1)

	print('Setting up correlations')

	# don't need randoms here if we know the period of the box
	print('Computing correlation functions.')
	#import pdb ; pdb.set_trace()

	c0c0 = compute_ii(shapes[smask], shapes2, options, rpbins=rpbins, period=options['box_size'], nbins=binning, weights1=wts1, weights2=wts2[smask])

	filename = '%s/wpp_%d%s%s.txt'%(options['savedir'],snapshot,split_type2,split_type)
	if (ijk>-1):
		filename = filename.replace('.txt', '-jk%d.txt'%ijk)

	if os.path.exists(filename):
		print('file exists: %s'%filename)
		return None

	export_array(filename, rpbins, c0c0)
		
	print('Done')



def compute_ii(cat1, cat2, options, period=100., rpbins=None, nbins=6, weights1=None, weights2=None):

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

	ii = ii_plus_projected(pvec2[mask2], avec2[mask2], evec2[mask2], pvec1[mask1], avec1[mask1], evec1[mask1], rpbins, pi_max, period=period, num_threads=1, weights1=weights1[mask1], weights2=weights2[mask2]) 

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

def split_catalogue(cat, options, snapshot, i):

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
	elif (s=='red'):
		m0,c0 = colour_split_params[snapshot]
		gi = cat['gmag']-cat['imag']
		mask = (gi>=(m0*cat['rmag']+c0))
	elif (s=='blue'):
		m0,c0 = colour_split_params[snapshot]
		gi = cat['gmag']-cat['imag']
		mask = (gi<(m0*cat['rmag']+c0))


	elif (s=='high_mass_red'):
		m0,c0 = colour_split_params[snapshot]
		gi = cat['gmag']-cat['imag']
		Ms = np.median(cat['stellar_mass_all'])
		mask = (gi>=(m0*cat['rmag']+c0)) & (cat['stellar_mass_all']>=Ms)
	elif (s=='high_mass_blue'):
		m0,c0 = colour_split_params[snapshot]
		gi = cat['gmag']-cat['imag']
		Ms = np.median(cat['stellar_mass_all'])
		mask = (gi<(m0*cat['rmag']+c0)) & (cat['stellar_mass_all']>=Ms)

	elif (s=='low_mass_red'):
		m0,c0 = colour_split_params[snapshot]
		gi = cat['gmag']-cat['imag']
		Ms = np.median(cat['stellar_mass_all'])
		mask = (gi>=(m0*cat['rmag']+c0)) & (cat['stellar_mass_all']<Ms)
	elif (s=='low_mass_blue'):
		m0,c0 = colour_split_params[snapshot]
		gi = cat['gmag']-cat['imag']
		Ms = np.median(cat['stellar_mass_all'])
		#import pdb ; pdb.set_trace()
		mask = (gi<(m0*cat['rmag']+c0)) & (cat['stellar_mass_all']<Ms)

	elif (s=='blue0'):
		m0,c0 = colour_split_params[snapshot]
		gi = cat['gmag']-cat['imag']
		#import pdb ; pdb.set_trace()
		bmask = (gi<(m0*cat['rmag']+c0))

		edges = find_bin_edges(cat['imag'][bmask],4)
		imask = (cat['imag']>=edges[3])

		mask = bmask & imask

		print('rmag=%2.3f'%cat['rmag'][mask].mean())

	elif (s=='blue1'):
		m0,c0 = colour_split_params[snapshot]
		gi = cat['gmag']-cat['imag']
		#import pdb ; pdb.set_trace()
		bmask = (gi<(m0*cat['rmag']+c0))

		edges = find_bin_edges(cat['imag'][bmask],4)
		imask = (cat['imag']>=edges[2]) & (cat['imag']<edges[3])

		mask = bmask & imask

		print('rmag=%2.3f'%cat['rmag'][mask].mean())

	elif (s=='blue2'):
		m0,c0 = colour_split_params[snapshot]
		gi = cat['gmag']-cat['imag']
		#import pdb ; pdb.set_trace()
		bmask = (gi<(m0*cat['rmag']+c0))

		edges = find_bin_edges(cat['imag'][bmask],4)
		imask = (cat['imag']>=edges[1]) & (cat['imag']<edges[2])

		mask = bmask & imask

		print('rmag=%2.3f'%cat['rmag'][mask].mean())

	elif (s=='blue3'):
		m0,c0 = colour_split_params[snapshot]
		gi = cat['gmag']-cat['imag']
		#import pdb ; pdb.set_trace()
		bmask = (gi<(m0*cat['rmag']+c0))

		edges = find_bin_edges(cat['imag'][bmask],4)
		imask = (cat['imag']>=edges[0]) & (cat['imag']<edges[1])

		mask = bmask & imask

		print('rmag=%2.3f'%cat['rmag'][mask].mean())



	elif (s=='all'):
		mask = np.ones_like(cat['x']).astype(bool)


	else:
		raise ValueError('Unrecognised split type:',s)

	print('Split (%s) leaves %d objects'%(s, cat['x'][mask].size))

	return '_%s'%options['split_type'].split()[i], cat[mask]



def find_bin_edges(x,nbins,w=None):
    """
    For an array x, returns the boundaries of nbins equal (possibly weighted by w) bins.
    """

    if w is None:
      xs=np.sort(x)
      r=np.linspace(0.,1.,nbins+1.)*(len(x)-1)
      return xs[r.astype(int)]

    fail=False
    ww=np.sum(w)/nbins
    i=np.argsort(x)
    k=np.linspace(0.,1.,nbins+1.)*(len(x)-1)
    k=k.astype(int)
    r=np.zeros((nbins+1))
    ist=0
    for j in xrange(1,nbins):
      # print(k[j],r[j-1])
      if k[j]<r[j-1]:
        print('Random weight approx. failed - attempting brute force approach')
        fail=True
        break
      w0=np.sum(w[i[ist:k[j]]])
      if w0<=ww:
        for l in xrange(k[j],len(x)):
          w0+=w[i[l]]
          if w0>ww:
            r[j]=x[i[l]]
            ist=l
            break
      else:
        for l in xrange(k[j],0,-1):
          w0-=w[i[l]]
          if w0<ww:
            r[j]=x[i[l]]
            ist=l
            break

    if fail:

      ist=np.zeros((nbins+1))
      ist[0]=0
      for j in xrange(1,nbins):
        wsum=0.
        for k in xrange(ist[j-1].astype(int),len(x)):
          wsum+=w[i[k]]
          if wsum>ww:
            r[j]=x[i[k-1]]
            ist[j]=k
            break

    r[0]=x[i[0]]
    r[-1]=x[i[-1]]

    return r