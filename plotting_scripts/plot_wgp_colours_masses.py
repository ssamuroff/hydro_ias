import numpy as np
import pylab as plt
from pylab import gca
plt.switch_backend('pdf')
plt.style.use('y1a1')
from matplotlib import rcParams
import fitsio as fi
import sys

rcParams['xtick.major.size'] = 3.5
rcParams['xtick.minor.size'] = 1.7
rcParams['ytick.major.size'] = 3.5
rcParams['ytick.minor.size'] = 1.7
rcParams['xtick.direction']='in'
rcParams['ytick.direction']='in'

npts = 12
dx=10**0.04
dx2=10**0.02

for i,snapshot in enumerate([99,78,62,50]):
	x,rr0 = np.loadtxt('tng_rr_lowM/2pt/txt/wgp_%d_all_low_mass_red.txt'%snapshot).T
	x,rr1 = np.loadtxt('tng_rr_highM/2pt/txt/wgp_%d_all_high_mass_red.txt'%snapshot).T
	x,bb0 = np.loadtxt('tng_bb_lowM/2pt/txt/wgp_%d_all_low_mass_blue.txt'%snapshot).T
	x,bb1 = np.loadtxt('tng_bb_highM/2pt/txt/wgp_%d_all_high_mass_blue.txt'%snapshot).T

	Er = np.sqrt(np.diag(fi.FITS('data/2pt/fits/2pt_tng_red_fidcov_iteration2.fits')['COVMAT'][:,:]))[i*npts : (i+1)*npts]*np.sqrt(2)
	Eb = np.sqrt(np.diag(fi.FITS('data/2pt/fits/2pt_tng_blue_fidcov_iteration2.fits')['COVMAT'][:,:]))[i*npts : (i+1)*npts]*np.sqrt(2)

	plt.subplot(4,1,i+1)
	l1=plt.errorbar(x*dx,x*rr0,yerr=x*Er, linestyle='none', color='pink', marker='.', label='Low Mass Red')
	l2=plt.errorbar(x/dx,x*rr1,yerr=x*Er, linestyle='none', color='darkred', marker='.', label='High Mass Red')
	l3=plt.errorbar(x,x*bb0,yerr=x*Eb, linestyle='none', color='steelblue', marker='x', label='Low Mass Blue')
	l4=plt.errorbar(x*dx2,x*bb1,yerr=x*Eb, linestyle='none', color='midnightblue', marker='+', label='High Mass Blue')

	plt.axhline(0, color='k', ls=':')
	plt.axvspan(0.0001, 6, color='grey', alpha=0.1)
	plt.xlim(0.1,120)
	plt.xscale('log')
	plt.ylabel(r'$r_{\rm p} w_{g+}$', fontsize=16)
	if snapshot!=50:
		plt.xticks(visible=False)
	plt.ylim(-2.5,5)
	plt.yticks([-2,0,2,4])
	if i==0:
		L = plt.legend([l1,l2], ["Low Mass Red", "High Mass Red"], fontsize=9, loc='upper left')
		gca().add_artist(L)
	elif i==1:
		L = plt.legend([l3,l4], ["Low Mass Blue", "High Mass Blue"], fontsize=9, loc='upper left')
		gca().add_artist(L)

plt.xlabel(r'$r_{\rm p}$ / $h^{-1}$ Mpc', fontsize=16)
plt.xticks(visible=True)
plt.subplots_adjust(wspace=0,hspace=0, right=0.6, bottom=0.14, top=0.98)
plt.savefig('wgp_colours_masses.pdf')
plt.savefig('wgp_colours_masses.png')
