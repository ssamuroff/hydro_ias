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

#rcParams['xtick.minor.visible']=True

theory_base = '/physics2/ssamurof/des/code/direct_ia/data/theory/'

fontsize=10
colours = {'mbii':'midnightblue','illustris':'forestgreen','tng':'darkmagenta'}
xmin = 8e-2
xmax = 210
dx=10**0.05

0
# eBOSS
plt.subplot(341)
plt.axvspan(0.001,6,color='grey',alpha=0.1)
plt.title(r'$z=0.0$', fontsize=12)

plt.ylabel(r'$r_\mathrm{p} \times w_{gg}$', fontsize=fontsize)
plt.xscale('log')
#plt.yscale('log')
plt.xlim(xmin,xmax)
plt.ylim(-20,120)
plt.xticks([0.1,1,10,100],visible=False)
plt.yticks([-20, 0,20,40,60,80,100],visible=True, fontsize=fontsize)
plt.axhline(0,color='k',ls=':')

x,w=np.loadtxt('mbii/2pt/txt/wgg_85.txt').T
C = np.loadtxt('mbii/cov/covmat-analytic.txt')
dy = np.sqrt(np.diag(C))[96:108]
plt.errorbar(x,x*w,yerr=x*dy,marker='.',color=colours['mbii'],linestyle='none')


x,w=np.loadtxt('illustris/2pt/txt/wgg_135.txt').T
#C = np.loadtxt('mbii/cov/covmat-analytic.txt')
#dy = np.sqrt(np.diag(C))[96:108]
dy = np.zeros_like(x)
plt.errorbar(x,x*w,yerr=x*dy,marker='.',color=colours['illustris'],linestyle='none')


x,w=np.loadtxt('tng/2pt/txt/wgg_99.txt').T
#C = np.loadtxt('mbii/cov/covmat-analytic.txt')
#dy = np.sqrt(np.diag(C))[96:108]
dy = np.zeros_like(x)
plt.errorbar(x,x*w,yerr=x*dy,marker='.',color=colours['tng'],linestyle='none')





plt.subplot(342)
plt.axvspan(0.001,6,color='grey',alpha=0.1)
plt.title(r'$z=0.30$', fontsize=12)

plt.xscale('log')
plt.xlim(xmin,xmax)
plt.ylim(-20,120)
plt.axhline(0,color='k',ls=':')
plt.xticks([0.1,1,10,100],visible=False)
plt.yticks([-20, 0,20,40,60,80,100],visible=False, fontsize=fontsize)

x,w=np.loadtxt('mbii/2pt/txt/wgg_79.txt').T
C = np.loadtxt('mbii/cov/covmat-analytic.txt')
dy = np.sqrt(np.diag(C))[108:120]
plt.errorbar(x,x*w,yerr=x*dy,marker='.',color=colours['mbii'],linestyle='none')


# redmagic low
plt.subplot(343)
plt.axvspan(0.001,6,color='grey',alpha=0.1)

plt.title(r'$z=0.62$', fontsize=12)
plt.xscale('log')
plt.xlim(xmin,xmax)
plt.ylim(-20,120)
plt.axhline(0,color='k',ls=':')
plt.xticks([0.1,1,10,100],visible=False)
plt.xticks([0.1,1,10,100],visible=False)
plt.yticks([-20, 0,20,40,60,80,100],visible=False, fontsize=fontsize)

x,w=np.loadtxt('mbii/2pt/txt/wgg_73.txt').T
C = np.loadtxt('mbii/cov/covmat-analytic.txt')
dy = np.sqrt(np.diag(C))[120:132]
plt.errorbar(x,x*w,yerr=x*dy,marker='.',color=colours['mbii'],linestyle='none')

#plt.axvspan(65,300,color='grey',alpha=0.1)


plt.subplot(344)
plt.axvspan(0.001,6,color='grey',alpha=0.1)

plt.title(r'$z=1.0$', fontsize=12)
plt.xscale('log')
plt.yscale('linear')
plt.xlim(xmin,xmax)
plt.axhline(0,color='k',ls=':')
plt.ylim(-20,120)
plt.xticks([0.1,1,10,100],visible=False)
plt.xticks([0.1,1,10,100],visible=False)
plt.yticks([-20, 0,20,40,60,80,100],visible=False, fontsize=fontsize)

x,w=np.loadtxt('mbii/2pt/txt/wgg_68.txt').T
C = np.loadtxt('mbii/cov/covmat-analytic.txt')
dy = np.sqrt(np.diag(C))[132:144]
plt.errorbar(x,x*w,yerr=x*dy,marker='.',color=colours['mbii'],linestyle='none')


plt.subplot(345)
plt.axvspan(0.001,6,color='grey',alpha=0.1)

plt.xscale('log')
plt.yscale('log')
plt.xlim(xmin,xmax)
plt.ylim(3e-3,4)
plt.xticks([0.1,1,10,100],visible=False)
plt.yticks([1e-2,1e-1,1],visible=True, fontsize=fontsize)
plt.ylabel(r'$w_{g+}$', fontsize=fontsize)


x,w=np.loadtxt('mbii/2pt/txt/wgp_85.txt').T
C = np.loadtxt('mbii/cov/covmat-analytic.txt')
dy = np.sqrt(np.diag(C))[0:12]
plt.errorbar(x,w,yerr=dy,marker='.',color=colours['mbii'],linestyle='none')

x,w=np.loadtxt('illustris/2pt/txt/wgp_135.txt').T
#C = np.loadtxt('mbii/cov/covmat-analytic.txt')
dy = np.zeros_like(x)
plt.errorbar(x,w,yerr=dy,marker='.',color=colours['illustris'],linestyle='none')

x,w=np.loadtxt('tng/2pt/txt/wgp_99.txt').T
#C = np.loadtxt('mbii/cov/covmat-analytic.txt')
dy = np.zeros_like(x)
plt.errorbar(x,w,yerr=dy,marker='.',color=colours['tng'],linestyle='none')



plt.subplot(346)
plt.axvspan(0.001,6,color='grey',alpha=0.1)

plt.xscale('log')
plt.yscale('log')
plt.xlim(xmin,xmax)
plt.ylim(3e-3,4)
plt.xticks([0.1,1,10,100],visible=False)
plt.yticks([1e-2,1e-1,1],visible=False, fontsize=fontsize)

x,w=np.loadtxt('mbii/2pt/txt/wgp_79.txt').T
C = np.loadtxt('mbii/cov/covmat-analytic.txt')
dy = np.sqrt(np.diag(C))[12:24]
plt.errorbar(x,w,yerr=dy,marker='.',color=colours['mbii'],linestyle='none')

plt.subplot(347)
plt.axvspan(0.001,6,color='grey',alpha=0.1)

x,w=np.loadtxt('mbii/2pt/txt/wgp_73.txt').T
C = np.loadtxt('mbii/cov/covmat-analytic.txt')
dy = np.sqrt(np.diag(C))[24:36]
plt.errorbar(x,w,yerr=dy,marker='.',color=colours['mbii'],linestyle='none')

plt.xscale('log')
plt.yscale('log')
plt.xlim(xmin,xmax)
plt.ylim(3e-3,4)
plt.xticks([0.1,1,10,100],visible=False)
plt.yticks([1e-2,1e-1,1],visible=False, fontsize=fontsize)

plt.subplot(348)
plt.axvspan(0.001,6,color='grey',alpha=0.1)

x,w=np.loadtxt('mbii/2pt/txt/wgp_68.txt').T
C = np.loadtxt('mbii/cov/covmat-analytic.txt')
dy = np.sqrt(np.diag(C))[36:48]
plt.errorbar(x,w,yerr=dy,marker='.',color=colours['mbii'],linestyle='none')

plt.xscale('log')
plt.yscale('log')
plt.xlim(xmin,xmax)
plt.ylim(3e-3,4)
plt.xticks([0.1,1,10,100],visible=False)
plt.yticks([1e-2,1e-1,1],visible=False, fontsize=fontsize)





plt.subplot(349)
plt.axvspan(0.001,6,color='grey',alpha=0.1)
plt.xscale('log')
plt.yscale('log')
plt.ylabel(r'$w_{++}$', fontsize=fontsize)
plt.xlim(xmin,xmax)
plt.ylim(9.5e-6,0.11)
plt.xlabel(r'$r_{\rm p}$ / $h^{-1}$ Mpc', fontsize=fontsize)
plt.xticks([0.1,1,10],visible=True, fontsize=fontsize)
plt.yticks([1e-5, 1e-3,1e-1],visible=True, fontsize=fontsize)


x,w=np.loadtxt('mbii/2pt/txt/wpp_85.txt').T
C = np.loadtxt('mbii/cov/covmat-analytic.txt')
dy = np.sqrt(np.diag(C))[48:60]
plt.errorbar(x,w,yerr=dy,marker='.',color=colours['mbii'],linestyle='none')

x,w=np.loadtxt('illustris/2pt/txt/wpp_135.txt').T
#C = np.loadtxt('mbii/cov/covmat-analytic.txt')
dy = np.zeros_like(x)
plt.errorbar(x,w,yerr=dy,marker='.',color=colours['illustris'],linestyle='none')

x,w=np.loadtxt('tng/2pt/txt/wpp_99.txt').T
#C = np.loadtxt('mbii/cov/covmat-analytic.txt')
dy = np.zeros_like(x)
plt.errorbar(x,w,yerr=dy,marker='.',color=colours['tng'],linestyle='none')





plt.subplot(3,4,10)
plt.axvspan(0.001,6,color='grey',alpha=0.1)
plt.xscale('log')
plt.yscale('log')
plt.xlim(xmin,xmax)
plt.ylim(9.5e-6,0.11)
plt.yticks([1e-5, 1e-3,1e-1],visible=False, fontsize=fontsize)
plt.xlabel(r'$r_{\rm p}$ / $h^{-1}$ Mpc', fontsize=fontsize)
plt.xticks([0.1,1,10],visible=True, fontsize=fontsize)

x,w=np.loadtxt('mbii/2pt/txt/wpp_79.txt').T
C = np.loadtxt('mbii/cov/covmat-analytic.txt')
dy = np.sqrt(np.diag(C))[60:72]
plt.errorbar(x,w,yerr=dy,marker='.',color=colours['mbii'],linestyle='none')


plt.subplot(3,4,11)
plt.axvspan(0.001,6,color='grey',alpha=0.1)
plt.xscale('log')
plt.yscale('log')
plt.xlim(xmin,xmax)

plt.xlabel(r'$r_{\rm p}$ / $h^{-1}$ Mpc', fontsize=fontsize)
plt.xticks([0.1,1,10],visible=True, fontsize=fontsize)
plt.ylim(9.5e-6,0.11)
plt.yticks([1e-5, 1e-3,1e-1],visible=False, fontsize=fontsize)

x,w=np.loadtxt('mbii/2pt/txt/wpp_73.txt').T
C = np.loadtxt('mbii/cov/covmat-analytic.txt')
dy = np.sqrt(np.diag(C))[72:84]
plt.errorbar(x,w,yerr=dy,marker='.',color=colours['mbii'],linestyle='none')



plt.subplot(3,4,12)
plt.axvspan(0.001,6,color='grey',alpha=0.1)
plt.xscale('log')
plt.yscale('log')
plt.xlim(xmin,xmax)
plt.ylim(9.5e-6,0.11)
plt.yticks([1e-5, 1e-3,1e-1],visible=False, fontsize=fontsize)

x,w=np.loadtxt('mbii/2pt/txt/wpp_68.txt').T
C = np.loadtxt('mbii/cov/covmat-analytic.txt')
dy = np.sqrt(np.diag(C))[84:96]
plt.errorbar(x,w,yerr=dy,marker='.',color=colours['mbii'],linestyle='none')


plt.xlabel(r'$r_{\rm p}$ / $h^{-1}$ Mpc', fontsize=fontsize)
plt.xticks([0.1,1,10,100],visible=True, fontsize=fontsize)


plt.subplots_adjust(bottom=0.14,left=0.14, wspace=0.0, hspace=0.15, right=0.98)
plt.savefig('dvecs_panels.pdf')
plt.savefig('dvecs_panels.png')
