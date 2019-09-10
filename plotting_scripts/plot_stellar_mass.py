import numpy as np
import pylab as plt
plt.switch_backend('pdf')
plt.style.use('y1a1')
from matplotlib import rcParams
import fitsio as fi

rcParams['xtick.major.size'] = 3.5
rcParams['xtick.minor.size'] = 1.7
rcParams['ytick.major.size'] = 3.5
rcParams['ytick.minor.size'] = 1.7
rcParams['xtick.direction']='in'
rcParams['ytick.direction']='in'

def get_x(b):
	return (b[:-1]+b[1:])/2

path1 = 'data/cats/fits/MBII_85_non-reduced_galaxy_shapes.fits'
path2 = 'data/cats/fits/Illustris-1_135_non-reduced_galaxy_shapes.fits'
path3 = 'data/cats/fits/TNG300-1_99_non-reduced_galaxy_shapes.fits'

logm1 = np.log10(fi.FITS(path1)[-1].read()['stellar_mass']*1e10)
logm2 = np.log10(fi.FITS(path2)[-1].read()['stellar_mass_all'])
logm3 = np.log10(fi.FITS(path3)[-1].read()['stellar_mass_all'])

H1,b1 = np.histogram(logm1,bins=np.linspace(8,12,100), normed=True)
H2,b2 = np.histogram(logm2,bins=np.linspace(8,12,100), normed=True)
H3,b3 = np.histogram(logm3,bins=np.linspace(8,12,100), normed=True)

x = get_x(b1)

plt.close()
plt.plot(x,H1,ls='-',lw=1.5,color='midnightblue',label='MBII')
plt.plot(x,H2,ls='-',lw=1.5,color='forestgreen',label='Illustris-1')
plt.plot(x,H3,ls='-',lw=1.5,color='darkmagenta',label='TNG')

plt.xlabel(r'Stellar Mass $\mathrm{log}M$ / $h^{-1} M_{\odot}$', fontsize=16)
plt.yticks(visible=False)
plt.ylim(ymin=0)
plt.subplots_adjust(bottom=0.14)
plt.legend(loc='upper right')
plt.savefig('simcomp_stellar_mass.pdf')
plt.savefig('simcomp_stellar_mass.png')
plt.close()