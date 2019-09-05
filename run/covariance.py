import numpy as np
import sys
import yaml
import fitsio as fi
import pylab as plt
plt.switch_backend('pdf')
plt.style.use('y1a1')


from src import analytic_cov2 as cov

params = yaml.load(open(sys.argv[-1]))

cov.run(params)
print('Done all')
