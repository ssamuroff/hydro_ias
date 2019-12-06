import fitsio as fi
import numpy as np
import sys
import os
import yaml

base = yaml.load(open(sys.argv[-1]))['savedir']
options = yaml.load(open(sys.argv[-1]))['fits']

filename = options['output']
os.system('rm %s'%filename)
f = fi.FITS(filename, 'rw')

corrs = options['correlations'].split()
snapshots = np.array(str(options['snapshots']).split()).astype(int) 


def initialise_empty_dict():
	out = {}
	out['BIN']=[]
	out['SEP']=[]
	out['VALUE']=[]
	return out

lengths = []
for c in corrs:
	print('saving %s'%c)
	out = initialise_empty_dict()
	info = []
	
	for i, s in enumerate(snapshots):
		print('     snapshot %d'%s)
		path = base + '/%s_%d.txt'%(c,s)
		x, w = np.genfromtxt(path).T
		out['SEP'].append(x)
		out['VALUE'].append(w)
		out['BIN'].append([i]*len(x))
		info.append(('SNAPSHOT_%d'%i,s))

	for k in out.keys():
		out[k] = np.concatenate(out[k])

	f.write(out)
	f[-1].write_key('EXTNAME',c)
	for lab,ind in info:
		f[-1].write_key(lab,ind)
	lengths.append(len(out['VALUE']))


if options['covariance'] is not None:
	cov = np.loadtxt(options['covariance'])
	f.write(cov)
	f[-1].write_key('EXTNAME','COVMAT')

	f[-1].write_key('STRT_0',0)
	count = 0
	for i,(l,c) in enumerate(zip(lengths,corrs)):
		count+=l
		f[-1].write_key('NAME_%d'%i,c)
		try:
			f[-1].write_key('STRT_%d'%(i+1),count)
		except:
			pass
else:
	print('No covariance matrix specified :(')

f.close()

print('Done')

