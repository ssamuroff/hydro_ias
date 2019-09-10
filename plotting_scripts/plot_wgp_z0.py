import numpy as np
import sys
import os
import fitsio as fi
import pylab as plt
plt.switch_backend('pdf')
plt.style.use('y1a1')



x0,y0 = np.loadtxt('mbii/2pt/txt/wgp_85.txt').T
dy = np.sqrt(np.diag(np.loadtxt('mbii/cov/covmat-analytic.txt')))[:12]
x1,y1 = np.loadtxt('illustris/2pt/txt/wgp_135.txt').T
x2,y2 = np.loadtxt('tng/2pt/txt/wgp_99.txt').T


plt.errorbar(x0,y0,yerr=dy,marker='*', linestyle='none', color='midnightblue', label='MBII')
plt.errorbar(x1,y1,marker='.', linestyle='none', color='forestgreen', label='Illustris-1')
plt.errorbar(x2,y2,marker='^', linestyle='none', color='darkmagenta', label='TNG')


plt.xscale('log')
plt.yscale('log')
plt.legend(loc='upper right', fontsize=12)
plt.subplots_adjust(bottom=0.14,left=0.14)
plt.xlabel(r'$r_{\rm p}$ / $h^{-1}$ Mpc',fontsize=16)
plt.ylabel(r'$w_{g+}$',fontsize=16)
plt.savefig('wgp_simcomp.png')
plt.savefig('wgp_simcomp.pdf')

plt.close()


x0,y0 = np.loadtxt('mbii/2pt/txt/wgg_85.txt').T
dy = np.sqrt(np.diag(np.loadtxt('mbii/cov/covmat-analytic.txt')))[96:108]
x1,y1 = np.loadtxt('illustris/2pt/txt/wgg_135.txt').T
x2,y2 = np.loadtxt('tng/2pt/txt/wgg_99.txt').T


plt.errorbar(x0,x0*y0,marker='*', linestyle='none', color='midnightblue', label='MBII')
plt.errorbar(x1,x1*y1,marker='.', linestyle='none', color='forestgreen', label='Illustris-1')
plt.errorbar(x2,x2*y2,marker='^', linestyle='none', color='darkmagenta', label='TNG')


plt.xscale('log')
#plt.yscale('log')
plt.legend(loc='lower left', fontsize=12)
plt.subplots_adjust(bottom=0.14,left=0.14)
plt.xlabel(r'$r_{\rm p}$ / $h^{-1}$ Mpc',fontsize=16)
plt.ylabel(r'$r_{\rm p} w_{gg}$',fontsize=16)
plt.savefig('wgg_simcomp.png')
plt.savefig('wgg_simcomp.pdf')
plt.close()