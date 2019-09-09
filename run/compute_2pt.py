import argparse
import numpy as np
import yaml

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--config', '-c', type=str, action='store')
args = parser.parse_args()

options = yaml.load(open(args.config))

correlations = options['correlations'].split()
snapshots = np.array(options['snapshots'].split()).astype(int)

for correlation in correlations:
	for snapshot in snapshots:
		print('Processing %s (snapshot %d)'%(correlation, snapshot) )
		exec('from src.twopoint import calculate_%s as fns'%correlation)
		fns.compute(options, options['nbins'], snapshot, -1)