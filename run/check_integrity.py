import numpy as np
import h5py
import glob
import sys


base = sys.argv[1]
files = glob.glob('%s/snap*hdf5'%base)

print('%d files in %s'%(len(files),base))

X = []

print('Will their check integrity.')

for f in files:
	#import pdb ; pdb.set_trace()
	try:
		tmp = h5py.File(f)
	except:
		X.append(f)
		print('%s cannot be opened.'%f)


print('Done all')


