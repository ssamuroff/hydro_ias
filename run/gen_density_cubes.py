import numpy as np
import sys
import src.tidal_tensor_tng as tt

snapshots = np.atleast_1d(sys.argv[1:]).astype(int)
ptype='star'

resolution=2**6
print('resolution:%d'%resolution)

tt.gen_density_cubes(snaps=snapshots,ptype=ptype, resolution=resolution)


ptype='dm'
tt.gen_density_cubes(snaps=snapshots,ptype=ptype, resolution=resolution)
