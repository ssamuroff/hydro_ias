import fitsio as fi
import numpy as np
import sys


def project_ellipticities(i, data):

	a3d = np.array([[data['av_x'][i], data['av_y'][i], data['av_z'][i]]])
	b3d = np.array([[data['bv_x'][i], data['bv_y'][i], data['bv_z'][i]]])
	c3d = np.array([[data['cv_x'][i], data['cv_y'][i], data['cv_z'][i]]])
	q3d = np.array([data['a'][i]/data['c'][i]])
	s3d = np.array([data['b'][i]/data['c'][i]])


	e1,e2 = project_3d_shape(a3d, b3d, c3d, q3d, s3d)

	return e1,e2

def project_3d_shape(a3d, b3d, c3d, q3d, s3d):
    """
    This function projects 3D ellipsoidal shapes onto a 2D ellipse and returns
    the 2 cartesian components of the ellipticity.
    See Piras2017:1707.06559 section 3.1
    and Joachimi2013:1203.6833
    """

    s = np.stack([a3d, b3d, c3d])
    w = np.stack([np.ones_like(q3d), q3d, s3d])

    #import pdb ; pdb.set_trace()

    k = np.sum(s[:,:,0:2]*np.expand_dims(s[:,:,2], axis=-1) / np.expand_dims(w[:,:]**2, axis=-1), axis=0)
    a2 =np.sum(s[:,:,2]**2/w[:,:]**2, axis=0)
    Winv = np.sum(np.einsum('ijk,ijl->ijkl', s[:,:,0:2], s[:,:,0:2]) / np.expand_dims(np.expand_dims(w[:,:]**2,-1),-1), axis=0) - np.einsum('ij,ik->ijk', k,k)/np.expand_dims(np.expand_dims(a2,-1),-1)
    W = np.linalg.inv(Winv)
    d = np.sqrt(np.linalg.det(W))
    e1 = (W[:,0,0] - W[:,1,1])/( W[:,0,0] + W[:,1,1] + 2*d)
    e2 = 2 * W[:,0,1]/( W[:,0,0] + W[:,1,1] + 2*d)
    return e1, e2


def export(filename, cat, new_columns, column_names, extra_files=[]):
	outfits = fi.FITS(filename,'rw')

	#if len(extra_files)>0:
	ind = np.argsort(cat['gal_id'])

	# keep all the existing columns
	out_dict = {}
	for name in cat.dtype.names:
		out_dict[name] = cat[name][ind]

	# also add the extra columns, as specified
	for col,name in zip(new_columns,column_names):
		if len(col)!=len(cat):
			print('Warning: column %s appears to have the wrong size (%d %d)'%(len(col),len(cat)))
		else:
			out_dict[name] = col[ind]

	for f in extra_files:
		print('Also including file: %s'%f)
		new = np.genfromtxt(f,names=True)
		ncol = len(new.dtype.names)
		print('%d columns'%ncol)

		cmask = np.argsort(new['gal_id'])
		import pdb ; pdb.set_trace()
		for name in new.dtype.names:
			out_dict[name] = new[name][cmask]

	print(out_dict.keys())

	outfits.write(out_dict)
	outfits.close()

	return None

if __name__ == "__main__":
	path = sys.argv[1]
	extra_files = sys.argv[2:]
	cat = np.genfromtxt(path, names=True)
	ngal = len(cat)

	print('Catalogue contains %d galaxies'%ngal)
	print('Computing ellipticities...')

	e1,e2 = [],[]
	for i in range(ngal):
		ellip1, ellip2 = project_ellipticities(i, cat)
		e1.append(ellip1)
		e2.append(ellip2)

	e1 = np.concatenate(e1)
	e2 = np.concatenate(e2)

	export(path.replace('.dat','.fits'), cat, [e1,e2], ['e1','e2'], extra_files=extra_files)

	print('faerdiga allt')