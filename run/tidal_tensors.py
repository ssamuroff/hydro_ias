import numpy as np
import sys
import yaml
import src.tidal_tensor_tng as tt

config = yaml.load(open(sys.argv[-1], 'rb'))
snapshots = np.atleast_1d(config['snapshot']).astype(int)

resoultion=config['resolution']
model = config['model']


ptype='dm'
tt.gen_tidal_tensors(snaps=snapshots,smoothing=[0.25]*len(snapshots), model=model, ptype=ptype, resolution=resoultion)

ptype='star'
tt.gen_tidal_tensors(snaps=snapshots,smoothing=[0.25]*len(snapshots), model=model, ptype=ptype, resolution=resoultion)
