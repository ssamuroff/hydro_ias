import numpy as np
import sys
import src.tidal_tensor as tt

snapshots = np.atleast_1d(sys.argv[1:]).astype(int)
ptype='star'

tt.gen_density_cubes(snaps=snapshots,ptype=ptype)
