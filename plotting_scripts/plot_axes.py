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

a1 = fi.FITS(path1)[-1].read()['a']
b1 = fi.FITS(path1)[-1].read()['b']
R1 = b1/a1

a2 = fi.FITS(path2)[-1].read()['a']
b2 = fi.FITS(path2)[-1].read()['b']
R2 = b2/a2

a3 = fi.FITS(path3)[-1].read()['a']
b3 = fi.FITS(path3)[-1].read()['b']
R3 = b3/a3

H1,x1 = np.histogram(R1,bins=np.linspace(0,1,150), normed=True)
H2,x2 = np.histogram(R2,bins=np.linspace(0,1,150), normed=True)
H3,x3 = np.histogram(R3,bins=np.linspace(0,1,150), normed=True)

x = get_x(x1)

plt.close()
plt.plot(x,H1,ls='-',lw=1.5,color='midnightblue',label='MBII')
plt.plot(x,H2,ls='-',lw=1.5,color='forestgreen',label='Illustris-1')
plt.plot(x,H3,ls='-',lw=1.5,color='darkmagenta',label='TNG')

plt.xlabel(r'$b/a$', fontsize=16)
plt.yticks(visible=False)
plt.ylim(ymin=0)
plt.xlim(0,1)
plt.subplots_adjust(bottom=0.14)
plt.legend(loc='upper left')
plt.savefig('simcomp_axis_ratios1.pdf')
plt.savefig('simcomp_axis_ratios1.png')
plt.close()






a1 = fi.FITS(path1)[-1].read()['a']
c1 = fi.FITS(path1)[-1].read()['c']
R1 = c1/a1

a2 = fi.FITS(path2)[-1].read()['a']
c2 = fi.FITS(path2)[-1].read()['c']
R2 = c2/a2

a3 = fi.FITS(path3)[-1].read()['a']
c3 = fi.FITS(path3)[-1].read()['c']
R3 = c3/a3

H1,x1 = np.histogram(R1,bins=np.linspace(0,1,150), normed=True)
H2,x2 = np.histogram(R2,bins=np.linspace(0,1,150), normed=True)
H3,x3 = np.histogram(R3,bins=np.linspace(0,1,150), normed=True)

x = get_x(x1)

plt.close()
plt.plot(x,H1,ls='-',lw=1.5,color='midnightblue',label='MBII')
plt.plot(x,H2,ls='-',lw=1.5,color='forestgreen',label='Illustris-1')
plt.plot(x,H3,ls='-',lw=1.5,color='darkmagenta',label='TNG')

plt.xlabel(r'$c/a$', fontsize=16)
plt.yticks(visible=False)
plt.ylim(ymin=0)
plt.xlim(0,1)
plt.subplots_adjust(bottom=0.14)
plt.legend(loc='upper left')
plt.savefig('simcomp_axis_ratios2.pdf')
plt.savefig('simcomp_axis_ratios2.png')
plt.close()