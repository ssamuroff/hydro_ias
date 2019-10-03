import numpy as np
import sys
import src.tidal_tensor as tt

snapshots = np.atleast_1d(sys.argv[1:]).astype(int)
ptype='star'
tt.gen_tidal_tensors(snaps=snapshots,smoothing=[0.25]*len(snapshots), ptype=ptype)
