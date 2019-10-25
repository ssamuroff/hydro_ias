import numpy as np
import pylab as plt
from pylab import gca
plt.switch_backend('pdf')
plt.style.use('y1a1')
from matplotlib import rcParams
import fitsio as fi


cmap = 'inferno'

fontsize=16

rcParams['xtick.major.size'] = 3.5
rcParams['xtick.minor.size'] = 1.7
rcParams['ytick.major.size'] = 3.5
rcParams['ytick.minor.size'] = 1.7
rcParams['xtick.direction']='in'
rcParams['ytick.direction']='in'



d = fi.FITS('/Volumes/groke/dm_density_tng_099_512.fits')[-1].read()

y = d[256,:,:]/d.mean()
plt.imshow(np.log(y), interpolation='none', cmap=cmap)
plt.colorbar(label=r'$\mathrm{ln}( \delta_{m}+1 )$')

plt.xticks([0, 124.9, 249.8, 374.6, 499.5], [0, 50,100,150,200], fontsize=fontsize)
plt.yticks([0, 124.9, 249.8, 374.6, 499.5], [0, 50,100,150,200], fontsize=fontsize)
plt.xlabel('$x$ / $h^{-1}$ Mpc', fontsize=16)
plt.ylabel('$y$ / $h^{-1}$ Mpc', fontsize=16)

plt.subplots_adjust(bottom=0.14,left=0.14)
plt.savefig('tng_256_yz_%s.pdf.pdf'%cmap)
plt.savefig('tng_256_yz_%s.pdf.png'%cmap)