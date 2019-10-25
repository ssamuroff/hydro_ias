import numpy as np
import sys
import src.tidal_tensor_tng as tt

snapshots = np.atleast_1d(sys.argv[1:]).astype(int)

resolution=2**6
print('Resolution = %d'%resolution)
tt.gen_shape_cubes(snaps=snapshots, resolution=resolution)
