import numpy as np
import pylab as plt
plt.switch_backend('pdf')
plt.style.use('y1a1')
from matplotlib import rcParams
import fitsio as fi

V = {'tng':205**3, 'illustris': 75**3, 'mbii':100**3}


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


H1,b1 = np.histogram(logm1,bins=np.linspace(8,13,90), normed=False)
H1=H1*1./V['mbii']
H2,b2 = np.histogram(logm2,bins=np.linspace(8,13,90), normed=False)
H2=H2*1./V['illustris']
H3,b3 = np.histogram(logm3,bins=np.linspace(8,13,90), normed=False)
H3=H3*1./V['tng']

x = get_x(b1)

plt.close()
plt.plot(x,H1,ls='-',lw=1.5,color='midnightblue',label='MBII')
plt.plot(x,H2,ls='-',lw=1.5,color='forestgreen',label='Illustris-1')
plt.plot(x,H3,ls='-',lw=1.5,color='darkmagenta',label='TNG')


plt.fill_between(x,H1,lw=1.5,color='midnightblue',alpha=0.15)
plt.fill_between(x,H2,lw=1.5,color='forestgreen',alpha=0.15)
plt.fill_between(x,H3,lw=1.5,color='darkmagenta',alpha=0.15)
plt.xlabel(r'Stellar Mass $\mathrm{log}M$ / $h^{-1} M_{\odot}$', fontsize=16)
plt.yticks(visible=False)
plt.ylim(ymin=0)
plt.subplots_adjust(bottom=0.14)
plt.legend(loc='upper right')
plt.savefig('simcomp_stellar_mass.pdf')
plt.savefig('simcomp_stellar_mass.png')
plt.close()


plt.close()
plt.plot(x,H1,ls='-',lw=1.5,color='midnightblue',label='MBII')
plt.plot(x,H2,ls='-',lw=1.5,color='forestgreen',label='Illustris-1')
plt.plot(x,H3,ls='-',lw=1.5,color='darkmagenta',label='TNG')


plt.ylabel(r'Mass Function $\mathrm{log} \mathit{\Phi} (M_*)$', fontsize=16)
plt.xlabel(r'Stellar Mass $\mathrm{log}M_*$ / $h^{-1} M_{\odot}$', fontsize=16)
plt.yticks(visible=True)

plt.legend(loc='upper right')
plt.yscale('log')
plt.xlim(9,13.5)

xf = np.log10(np.array([1e9, 2e9, 3e9, 4e9, 1e10, 2e10, 3e10, 6.2e10,8e10, 9e10,1e11, 3e11, 1e12])*0.7)
yf = np.array([1.5e-2, 1e-2, 8.5e-3, 7e-3, 6.1e-3,5.9e-3, 5.9e-3, 4e-3, 2.9e-3,2.5e-3,2e-3, 3e-4, 2e-5])

#plt.plot(xf,yf,'o',color='grey', alpha=0.4)

plt.fill_between(x,H1,lw=1.5,color='midnightblue',alpha=0.15)
plt.fill_between(x,H2,lw=1.5,color='forestgreen',alpha=0.15)
plt.fill_between(x,H3,lw=1.5,color='darkmagenta',alpha=0.15)

plt.subplots_adjust(bottom=0.14, left=0.145, right=0.98, top=0.98)
plt.savefig('simcomp_stellar_mass_log.pdf')
plt.savefig('simcomp_stellar_mass_log.png')
plt.close()


def cmdens(N,L):
	V = L*L*L
	n = N*1.0/V
	print('comoving number density = %3.3f h^3 Mpc^-3'%n)

cmdens(logm1.size,100)
cmdens(logm2.size,75)
cmdens(logm3.size,205)










path1 = 'data/cats/fits/MBII_68_non-reduced_galaxy_shapes.fits'
#path2 = 'data/cats/fits/Illustris-1_135_non-reduced_galaxy_shapes.fits'
path3 = 'data/cats/fits/TNG300-1_50_non-reduced_galaxy_shapes.fits'

logm1 = np.log10(fi.FITS(path1)[-1].read()['stellar_mass_all'])
#logm2 = np.log10(fi.FITS(path2)[-1].read()['stellar_mass_all'])
logm3 = np.log10(fi.FITS(path3)[-1].read()['stellar_mass_all'])

H1,b1 = np.histogram(logm1,bins=np.linspace(8,13,150), normed=True)
#H2,b2 = np.histogram(logm2,bins=np.linspace(8,13,150), normed=True)
H3,b3 = np.histogram(logm3,bins=np.linspace(8,13,150), normed=True)

x = get_x(b1)

plt.close()
plt.title('$z=1.0$')
plt.plot(x,H1,ls='-',lw=1.5,color='midnightblue',label='MBII')
#plt.plot(x,H2,ls='-',lw=1.5,color='forestgreen',label='Illustris-1')
plt.plot(x,H3,ls='-',lw=1.5,color='darkmagenta',label='TNG')

plt.xlabel(r'Stellar Mass $\mathrm{log}M$ / $h^{-1} M_{\odot}$', fontsize=16)
plt.yticks(visible=False)
plt.ylim(ymin=0)
plt.subplots_adjust(bottom=0.14)
plt.legend(loc='upper right')
plt.savefig('simcomp_stellar_mass_z3.pdf')
plt.savefig('simcomp_stellar_mass_z3.png')
plt.close()


plt.close()
plt.plot(x,H1,ls='-',lw=1.5,color='midnightblue',label='MBII')
#plt.plot(x,H2,ls='-',lw=1.5,color='forestgreen',label='Illustris-1')
plt.plot(x,H3,ls='-',lw=1.5,color='darkmagenta',label='TNG')

plt.xlabel(r'Stellar Mass $\mathrm{log}M$ / $h^{-1} M_{\odot}$', fontsize=16)
plt.yticks(visible=True)
plt.subplots_adjust(bottom=0.14)
plt.legend(loc='upper right')
plt.yscale('log')
plt.savefig('simcomp_stellar_mass_log_z3.pdf')
plt.savefig('simcomp_stellar_mass_log_z3.png')
plt.close()

print('z=1.0')
def cmdens(N,L):
	V = L*L*L
	n = N*1.0/V
	print('comoving number density = %3.3f h^3 Mpc^-3'%n)

cmdens(logm1.size,100)
cmdens(logm2.size,75)
cmdens(logm3.size,205)
