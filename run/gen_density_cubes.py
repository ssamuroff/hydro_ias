import numpy as np
import sys
import src.tidal_tensor_tng as tt

snapshots = np.atleast_1d(sys.argv[1:]).astype(int)



#resolution=16
#resolution=32

#resolution=64

#resolution=128
resolution=256
print('resolution: %d'%resolution)

ptype='dm'
tt.gen_density_cubes(snaps=snapshots,ptype=ptype, resolution=resolution)


ptype='star'
tt.gen_density_cubes(snaps=snapshots,ptype=ptype, resolution=resolution)


