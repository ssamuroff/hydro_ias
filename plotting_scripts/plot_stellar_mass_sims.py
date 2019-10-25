import numpy as np
import pylab as plt
plt.switch_backend('pdf')
plt.style.use('y1a1')
from matplotlib import rcParams
import fitsio as fi

V = {'tng':205**3, 'illustris': 75**3, 'mbii':100**3}
colours = ['darkmagenta', 'royalblue', 'forestgreen', 'plum']
redshift = [0,0.33,0.62,1.]

rcParams['xtick.major.size'] = 3.5
rcParams['xtick.minor.size'] = 1.7
rcParams['ytick.major.size'] = 3.5
rcParams['ytick.minor.size'] = 1.7
rcParams['xtick.direction']='in'
rcParams['ytick.direction']='in'

def get_x(b):
	return (b[:-1]+b[1:])/2

def mass_function(logm, V, dM):
	H,b = np.histogram(logm,bins=np.linspace(8,13,90), normed=False)
	H = H*1. /V/dM
	return H,b


dM = np.linspace(8,13,90)[1] - np.linspace(8,13,90)[0]


mbii_phi = []
for s in [85,79,73,68]:
	path = 'data/cats/fits/MBII_%d_non-reduced_galaxy_shapes.fits'%s
	if s==85:
		logm = np.log10(fi.FITS(path)[-1].read()['stellar_mass']*1e10)
	else:
		logm = np.log10(fi.FITS(path)[-1].read()['stellar_mass_all'])

	H,b = mass_function(logm,V['mbii'], dM)
	x = get_x(b)
	mbii_phi.append(H)



tng_phi = []
for s in [99,78,62,50]:
	path = 'data/cats/fits/TNG300-1_%d_non-reduced_galaxy_shapes.fits'%s
	logm = np.log10(fi.FITS(path)[-1].read()['stellar_mass_all'])

	H,b = mass_function(logm,V['tng'], dM)
	tng_phi.append(H)


plt.close()
plt.subplot(111)

for i,z in enumerate(redshift):
	plt.plot(x,mbii_phi[i],ls='-',lw=1.5,color=colours[i],label='MBII $z=%3.2f$'%z)
	#plt.fill_between(x,mbii_phi[i],lw=1.5,color=colours[i],alpha=0.15)


plt.ylabel(r'Mass Function $\mathit{\Phi} (M_*)$', fontsize=16)
plt.xlabel(r'Stellar Mass $\mathrm{log}M_*$ / $h^{-1} M_{\odot}$', fontsize=16)
plt.yticks(visible=True)

plt.legend(loc='upper right')
plt.yscale('log')
plt.xlim(9,13.5)

plt.subplots_adjust(bottom=0.14, left=0.145, right=0.98, top=0.98)
plt.savefig('mbii_stellar_mass_log.pdf')
plt.savefig('mbii_stellar_mass_log.png')
plt.close()

plt.close()
plt.subplot(111)

for i,z in enumerate(redshift):
	plt.plot(x,tng_phi[i],ls='-',lw=1.5,color=colours[i],label='TNG $z=%3.2f$'%z)
	#plt.fill_between(x,tng_phi[i],lw=1.5,color=colours[i],alpha=0.15)


plt.ylabel(r'Mass Function $\mathit{\Phi} (M_*)$', fontsize=16)
plt.xlabel(r'Stellar Mass $\mathrm{log}M_*$ / $h^{-1} M_{\odot}$', fontsize=16)
plt.yticks(visible=True)

plt.legend(loc='upper right')
plt.yscale('log')
plt.xlim(9,13.5)



plt.subplots_adjust(bottom=0.14, left=0.145, right=0.98, top=0.98)
plt.savefig('tng_stellar_mass_log.pdf')
plt.savefig('tng_stellar_mass_log.png')
plt.close()

