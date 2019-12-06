import fitsio as fi
import sys
import numpy as np


file1 = sys.argv[1]
file2 = sys.argv[2]

f1 = fi.FITS(file1)[1].read()
f2 = fi.FITS(file2)[1].read()



halo_masses = f1['mass'][f2['gal_id']]

import pdb ; pdb.set_trace()


out={}
for k in f2.dtype.names:
    out[k] = f2[k]

out['host_halo_mass_200m'] = halo_masses

outfits = fi.FITS('new_cat.fits','rw')
outfits.write(out)
outfits.close()
print('Done')
