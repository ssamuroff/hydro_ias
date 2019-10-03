import fitsio as fi
import numpy as np
import pylab as plt
plt.switch_backend('agg')
plt.style.use('y1a1')



tng = fi.FITS('data/cats/fits/TNG300-1_99_non-reduced_galaxy_shapes.fits')[-1].read()
illustris = fi.FITS('data/cats/fits/Illustris-1_135_non-reduced_galaxy_shapes.fits')[-1].read()
mbii = fi.FITS('data/cats/fits/MBII_68_non-reduced_galaxy_shapes.fits')[-1].read()

H_tng_1, b = np.histogram(tng['e1'], bins=np.linspace(-0.5,0.5, 100), normed=1)
H_ill_1, b = np.histogram(illustris['e1'], bins=np.linspace(-0.5,0.5, 100), normed=1)
H_mb2_1, b = np.histogram(mbii['e1'], bins=np.linspace(-0.5,0.5, 100), normed=1)

x = (b[:-1]+b[1:])/2

plt.plot(x, H_tng_1, color='darkmagenta', ls='-', label='TNG300')
plt.plot(x, H_mb2_1, color='midnightblue', ls='-', label='MBII')
plt.plot(x, H_ill_1, color='forestgreen', ls='-', label='Illustris-1')
plt.ylim(ymin=0)
plt.yticks(visible=False)

plt.xlabel('$e_1$', fontsize=16)
plt.legend(loc='upper right', fontsize=12)
plt.subplots_adjust(bottom=0.14)
plt.savefig('simcompare_e1.pdf')
plt.savefig('simcompare_e1.png')
plt.close()


H_tng_2, b = np.histogram(tng['e2'], bins=np.linspace(-0.5,0.5, 100), normed=1)
H_ill_2, b = np.histogram(illustris['e2'], bins=np.linspace(-0.5,0.5, 100), normed=1)
H_mb2_2, b = np.histogram(mbii['e2'], bins=np.linspace(-0.5,0.5, 100), normed=1)

x = (b[:-1]+b[1:])/2

plt.plot(x, H_tng_2, color='darkmagenta', ls='-', label='TNG300')
plt.plot(x, H_mb2_2, color='midnightblue', ls='-', label='MBII')
plt.plot(x, H_ill_2, color='forestgreen', ls='-', label='Illustris-1')
plt.ylim(ymin=0)
plt.yticks(visible=False)
plt.subplots_adjust(bottom=0.14)

plt.xlabel('$e_2$', fontsize=16)
plt.legend(loc='upper right', fontsize=12)
plt.savefig('simcompare_e2.pdf')
plt.savefig('simcompare_e2.png')
plt.close()

def quad(a,b): return np.sqrt(a*a+b*b)

H_tng, b = np.histogram(quad(tng['e1'],tng['e2']), bins=np.linspace(0,1., 200), normed=1)
H_ill, b = np.histogram(quad(illustris['e1'],illustris['e2']), bins=np.linspace(0,1., 200), normed=1)
H_mb2, b = np.histogram(quad(mbii['e1'],mbii['e2']), bins=np.linspace(0,1., 200), normed=1)

x = (b[:-1]+b[1:])/2

plt.plot(x, H_tng, color='darkmagenta', ls='-', label='TNG300')
plt.plot(x, H_mb2, color='midnightblue', ls='-', label='MBII')
plt.plot(x, H_ill, color='forestgreen', ls='-', label='Illustris-1')

plt.fill_between(x, H_tng, color='darkmagenta', alpha=0.15)
plt.fill_between(x, H_mb2, color='midnightblue', alpha=0.15)
plt.fill_between(x, H_ill, color='forestgreen', alpha=0.15)

plt.ylim(ymin=0)
plt.yticks(visible=False)
plt.xlim(0,0.65)
plt.subplots_adjust(bottom=0.14)

plt.xlabel('Ellipticity $e$', fontsize=16)
plt.legend(loc='upper right')
plt.savefig('simcompare_e.pdf')
plt.savefig('simcompare_e.png')
plt.close()
