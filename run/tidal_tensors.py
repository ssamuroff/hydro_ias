import numpy as np
import sys
import src.tidal_tensor_tng as tt

snapshots = np.atleast_1d(sys.argv[1:]).astype(int)
resoultion=2**5

ptype='dm'
tt.gen_tidal_tensors(snaps=snapshots,smoothing=[0.25]*len(snapshots), ptype=ptype, resolution=resoultion)



ptype='star'
tt.gen_tidal_tensors(snaps=snapshots,smoothing=[0.25]*len(snapshots), ptype=ptype, resolution=resoultion)
