import numpy as np
import sys
import src.tidal_tensor as tt

snapshots = np.atleast_1d(sys.argv[1:]).astype(int)

tt.gen_shape_cubes(snaps=snapshots)
