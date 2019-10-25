import numpy as np
import pylab as plt
plt.switch_backend('pdf')
plt.style.use('y1a1')
from matplotlib import rcParams
import fitsio as fi

V = {'tng':205**3, 'illustris': 75**3, 'mbii':100**3}

fontsize=16
marker='.'

colour='darkmagenta'
facecolour = colour
sim='tng'
snapshot=99

s=None


rcParams['xtick.major.size'] = 3.5
rcParams['xtick.minor.size'] = 1.7
rcParams['ytick.major.size'] = 3.5
rcParams['ytick.minor.size'] = 1.7
rcParams['xtick.direction']='in'
rcParams['ytick.direction']='in'

plt.xscale('log')
plt.yscale('log')
plt.xlim(8e-2,210)
plt.ylim(1e-2,5)
plt.xticks([0.1,1,10,100],visible=True)
plt.yticks([1e-2,1e-1,1e0],visible=True, fontsize=fontsize)

plt.axvline(6,color='pink',ls='--', lw=1.)
plt.axvline(4,color='hotpink',ls='--', lw=1.)
plt.axvline(2,color='mediumvioletred',ls='--', lw=1.)

plt.axvspan(1e-3, 6, color='pink',alpha=0.2)
plt.axvspan(1e-3, 4, color='hotpink',alpha=0.2)
plt.axvspan(1e-3, 2, color='mediumvioletred',alpha=0.2)
plt.axvspan(1e-3, 0.5, color='darkmagenta',alpha=0.2)


x,w = np.loadtxt('%s/2pt/txt/wgp_%d.txt'%(sim, snapshot)).T
C = np.loadtxt('%s/cov/covmat-analytic.txt'%sim)
dy = np.sqrt(np.diag(C))[0:12]



plt.errorbar(x, w, yerr=dy, marker=marker, markeredgecolor=colour, markersize=s, ecolor=colour, markerfacecolor=facecolour,  linestyle='none')



# also the theory lines
xf = np.loadtxt('data/theory/tng/r_p.txt')
y0 = np.loadtxt('data/theory/tng/wgp_99_cut0.txt')
y1 = np.loadtxt('data/theory/tng/wgp_99_cut1.txt')
y2 = np.loadtxt('data/theory/tng/wgp_99_cut2.txt')
y4 = np.loadtxt('data/theory/tng/wgp_99_cut4.txt')

y_nla = np.loadtxt('data/theory/tng/wgp_99.txt')

plt.plot(xf,y_nla,color='k',ls=':',lw=1.5, label=r'NLA $r_{\rm p}>6 h^{-1}$ Mpc')

plt.plot(xf,y0,color='pink',ls='-',lw=1.5, label=r'TATT $r_{\rm p}>6 h^{-1}$ Mpc')
plt.plot(xf,y1,color='hotpink',ls='-',lw=1.5, label=r'TATT $r_{\rm p}>4 h^{-1}$ Mpc')
plt.plot(xf,y2,color='mediumvioletred',ls='-',lw=1.5, label=r'TATT $r_{\rm p}>2 h^{-1}$ Mpc')
plt.plot(xf,y4,color='darkmagenta',ls='-',lw=1.5, label=r'TATT $r_{\rm p}>0.5 h^{-1}$ Mpc')


plt.ylabel(r'$w_{g+}$', fontsize=fontsize)
plt.xlabel(r'$r_{\rm p}$ / $h^{-1}$ Mpc', fontsize=fontsize)

plt.legend(loc='upper right')

plt.subplots_adjust(bottom=0.14,left=0.14, wspace=0.0, hspace=0.15, right=0.98)
plt.savefig('tatt_z0_scales.pdf')
plt.savefig('tatt_z0_scales.png')

#import pdb ; pdb.set_trace()
#
#Cgp = C[:12,:12]
#C0 = np.linalg.inv(Cgp[8:,8:])
#mask=x>6
#fmask = (xf>0.01) & (xf<200)
#R = w[mask] - 10**interp1d(np.log10(xf[fmask]), np.log10(y_nla[fmask]))(np.log10(x[mask]))
#X = np.dot(R,np.dot(C0,R))
#dof = len(R) - 2