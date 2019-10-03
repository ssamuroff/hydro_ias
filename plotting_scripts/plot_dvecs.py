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



def plot_gg(sim, snapshot, colour, errors=True, lim=[0,20], yticks=True):
	plt.xscale('log')
	plt.xlim(8e-2,210)
	plt.ylim(-20,120)
	plt.xticks([0.1,1,10,100],visible=False)
	plt.axhline(0,color='k',ls=':')
	plt.yticks([-20, 0,20,40,60,80,100],visible=yticks, fontsize=fontsize)

	x,w = np.loadtxt('%s/2pt/txt/wgg_%d.txt'%(sim, snapshot)).T
	if errors:
		C = np.loadtxt('%s/cov/covmat-analytic.txt'%sim)
		dy = np.sqrt(np.diag(C))[lim[0]:lim[1]]
	else:
		dy = np.zeros_like(x)
	plt.errorbar(x, x*w, yerr=x*dy, marker='.',color=colour, linestyle='none')


def plot_gp(sim, snapshot, colour, errors=True, lim=[0,20], yticks=True):
	plt.xscale('log')
	plt.yscale('log')
	plt.xlim(8e-2,210)
	plt.ylim(1.5e-3,5)
	plt.xticks([0.1,1,10,100],visible=False)
	plt.yticks([1e-2,1e-1,1e0],visible=yticks, fontsize=fontsize)

	x,w = np.loadtxt('%s/2pt/txt/wgp_%d.txt'%(sim, snapshot)).T
	if errors:
		C = np.loadtxt('%s/cov/covmat-analytic.txt'%sim)
		dy = np.sqrt(np.diag(C))[lim[0]:lim[1]]
	else:
		dy = np.zeros_like(x)
	plt.errorbar(x, w, yerr=dy, marker='.',color=colour, linestyle='none')

def plot_pp(sim, snapshot, colour, errors=True, lim=[0,20], yticks=True):
	plt.xscale('log')
	plt.yscale('log')
	plt.xlim(8e-2,210)
#	plt.ylim(-0.005,0.025)
	#plt.ylim(9.5e-6,0.11)
	plt.yticks([1e-4,1e-3,1e-2,1e-1],visible=yticks, fontsize=fontsize)
	#plt.yticks([-0.01, -0.005, 0, 0.005, 0.01],visible=yticks, fontsize=fontsize)

	x,w = np.loadtxt('%s/2pt/txt/wpp_%d.txt'%(sim, snapshot)).T
	if errors:
		C = np.loadtxt('%s/cov/covmat-analytic.txt'%sim)
		dy = np.sqrt(np.diag(C))[lim[0]:lim[1]]
	else:
		dy = np.zeros_like(x)
	mask = w!=0
	#import pdb ; pdb.set_trace()
	plt.errorbar(x[mask], w[mask], yerr=dy[mask], marker='.',color=colour, linestyle='none')
	#plt.axhline(0,color='k',ls=':')





plt.subplot(341)
plt.axvspan(0.001,6,color='grey',alpha=0.1)
plt.title(r'$z=0.0$', fontsize=12)
plt.ylabel(r'$r_\mathrm{p} \times w_{gg}$', fontsize=fontsize)


#plot_gg('illustris', 135, colours['illustris'], errors=True, lim=[-8,-1])

plot_gg('tng', 99, colours['tng'], errors=True, lim=[96,108])
plot_gg('mbii', 85, colours['mbii'], errors=True, lim=[96,108])
plot_gg('illustris', 135, colours['illustris'], errors=True, lim=[64,72])



plt.subplot(342)
plt.axvspan(0.001,6,color='grey',alpha=0.1)
plt.yticks(visible=False)
plt.title(r'$z=0.30$', fontsize=12)

#plot_gg('tng', 78, colours['tng'], errors=True, lim=[108,120], yticks=False)
plot_gg('mbii', 79, colours['mbii'], errors=True, lim=[108,120], yticks=False)
plot_gg('illustris', 114, colours['illustris'], errors=True, lim=[72,80], yticks=False)



plt.subplot(343)
plt.axvspan(0.001,6,color='grey',alpha=0.1)
plt.yticks(visible=False)
plt.title(r'$z=0.62$', fontsize=12)
#plot_gg('tng', 62, colours['tng'], errors=True, lim=[120,132], yticks=False)
plot_gg('mbii', 73, colours['mbii'], errors=True, lim=[120,132], yticks=False)
plot_gg('illustris', 98, colours['illustris'], errors=True, lim=[80,88], yticks=False)



plt.subplot(344)
plt.axvspan(0.001,6,color='grey',alpha=0.1)
plt.yticks(visible=False)

plt.title(r'$z=1.0$', fontsize=12)
#plot_gg('tng', 50, colours['tng'], errors=True, lim=[132,144], yticks=False)
plot_gg('mbii', 68, colours['mbii'], errors=True, lim=[132,144], yticks=False)
plot_gg('illustris', 85, colours['illustris'], errors=True, lim=[88,96], yticks=False)





plt.subplot(345)
plt.axvspan(0.001,6,color='grey',alpha=0.1)

plt.ylabel(r'$w_{g+}$', fontsize=fontsize)
plt.yticks([1e-2,1e-1,1],visible=True, fontsize=fontsize)


plot_gp('tng', 99, colours['tng'], errors=True, lim=[0,12])
plot_gp('mbii', 85, colours['mbii'], errors=True, lim=[0,12])
plot_gp('illustris', 135, colours['illustris'], errors=True, lim=[0,8])

plt.subplot(346)
plt.axvspan(0.001,6,color='grey',alpha=0.1)
plt.yticks(visible=False)

#plot_gp('tng', 78, colours['tng'], errors=True, lim=[12,24], yticks=False)
plot_gp('mbii', 79, colours['mbii'], errors=True, lim=[12,24], yticks=False)
plot_gp('illustris', 114, colours['illustris'], errors=True, lim=[8,16], yticks=False)


plt.subplot(347)
plt.axvspan(0.001,6,color='grey',alpha=0.1)
plt.yticks(visible=False)

#plot_gp('tng', 62, colours['tng'], errors=True, lim=[24,36], yticks=False)
plot_gp('mbii', 73, colours['mbii'], errors=True, lim=[24,36], yticks=False)
plot_gp('illustris', 98, colours['illustris'], errors=True, lim=[16,24], yticks=False)


plt.subplot(348)
plt.axvspan(0.001,6,color='grey',alpha=0.1)
plt.yticks(visible=False)

#plot_gp('tng', 50, colours['tng'], errors=True, lim=[36,48], yticks=False)
plot_gp('mbii', 68, colours['mbii'], errors=True, lim=[36,48], yticks=False)
plot_gp('illustris', 85, colours['illustris'], errors=True, lim=[24,32], yticks=False)



plt.subplot(349)
plt.axvspan(0.001,6,color='grey',alpha=0.1)

plt.ylabel(r'$w_{++}$', fontsize=fontsize)
#plt.yticks([1e-5, 1e-3,1e-1],visible=False, fontsize=fontsize)
plt.xlabel(r'$r_{\rm p}$ / $h^{-1}$ Mpc', fontsize=fontsize)

plot_pp('tng', 99, colours['tng'], errors=True, lim=[48,60])
plot_pp('mbii', 85, colours['mbii'], errors=True, lim=[48,60])
plot_pp('illustris', 135, colours['illustris'], errors=True, lim=[32,40])

plt.subplot(3,4,10)
plt.axvspan(0.001,6,color='grey',alpha=0.1)
plt.yticks(visible=False)

plt.xlabel(r'$r_{\rm p}$ / $h^{-1}$ Mpc', fontsize=fontsize)
#plot_pp('tng', 78, colours['tng'], errors=True, lim=[60,72], yticks=False)
plot_pp('mbii', 79, colours['mbii'], errors=True, lim=[60,72], yticks=False)
plot_pp('illustris', 114, colours['illustris'], errors=True, lim=[40,48], yticks=False)

plt.subplot(3,4,11)
plt.axvspan(0.001,6,color='grey',alpha=0.1)
plt.yticks(visible=False)

plt.xlabel(r'$r_{\rm p}$ / $h^{-1}$ Mpc', fontsize=fontsize)

#plot_pp('tng', 62, colours['tng'], errors=True, lim=[72,84], yticks=False)
plot_pp('mbii', 73, colours['mbii'], errors=True, lim=[72,84], yticks=False)
plot_pp('illustris', 98, colours['illustris'], errors=True, lim=[48,56], yticks=False)

plt.subplot(3,4,12)
plt.axvspan(0.001,6,color='grey',alpha=0.1)
plt.yticks(visible=False)

plt.xlabel(r'$r_{\rm p}$ / $h^{-1}$ Mpc', fontsize=fontsize)
#plot_pp('tng', 50, colours['tng'], errors=True, lim=[84,96], yticks=False)
plot_pp('mbii', 68, colours['mbii'], errors=True, lim=[84,96], yticks=False)
plot_pp('illustris', 85, colours['illustris'], errors=True, lim=[56,64], yticks=False)




plt.subplots_adjust(bottom=0.14,left=0.14, wspace=0.0, hspace=0.15, right=0.98)
plt.savefig('dvecs_panels.pdf')
plt.savefig('dvecs_panels.png')
