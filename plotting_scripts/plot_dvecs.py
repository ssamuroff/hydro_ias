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



def plot_gg(sim, snapshot, colour, errors=True, lim=[0,20], yticks=True, label=None, marker='.', facecolour=None):
	plt.xscale('log')
	plt.xlim(8e-2,210)
	plt.ylim(-35,120)
	plt.xticks([0.1,1,10,100],visible=False)
	plt.axhline(0,color='k',ls=':')
	plt.yticks([ 0,40,80,120],visible=yticks, fontsize=fontsize)
	if facecolour is None:
		facecolour = colour
	if marker=='.':
		s=None
	else:
		s = 2.0

	x,w = np.loadtxt('%s/2pt/txt/wgg_%d.txt'%(sim, snapshot)).T
	if errors:
		C = np.loadtxt('%s/cov/covmat-analytic.txt'%sim)
		dy = np.sqrt(np.diag(C))[lim[0]:lim[1]]
	else:
		dy = np.zeros_like(x)
	plt.errorbar(x, x*w, yerr=x*dy, 
		marker=marker,
		markeredgecolor=colour, 
		markersize=s,
		ecolor=colour,
		markerfacecolor=facecolour, 
		linestyle='none', 
		label=label)


def plot_gp(sim, snapshot, colour, errors=True, lim=[0,20], yticks=True, label=None, marker='.', facecolour=None):
	plt.xscale('log')
	#plt.yscale('log')
	plt.xlim(8e-2,210)
	plt.ylim(-0.1,2.3)
	#plt.ylim(1e-2,5)
	plt.xticks([0.1,1,10,100],visible=False)
	#plt.yticks([1e-2,1e-1,1e0],visible=yticks, fontsize=fontsize)
	plt.yticks([0,0.4,0.8,1.2,1.6,2.],visible=yticks, fontsize=fontsize)
	if facecolour is None:
		facecolour = colour
	if marker=='.':
		s=None
	else:
		s = 2.0

	plt.axhline(0,color='k',ls=':')

	x,w = np.loadtxt('%s/2pt/txt/wgp_%d.txt'%(sim, snapshot)).T
	if errors:
		C = np.loadtxt('%s/cov/covmat-analytic.txt'%sim)
		dy = np.sqrt(np.diag(C))[lim[0]:lim[1]]
	else:
		dy = np.zeros_like(x)
	plt.errorbar(x, x*w, yerr=x*dy, 
		marker=marker,
		markeredgecolor=colour, 
		markersize=s,
		ecolor=colour,
		markerfacecolor=facecolour, 
		linestyle='none', 
		label=label)

def plot_pp(sim, snapshot, colour, errors=True, lim=[0,20], yticks=True, label=None, marker='.', facecolour=None):
	plt.xscale('log')
	#plt.yscale('log')
	plt.xlim(8e-2,210)
	#plt.ylim(1e-4,5e-1)
	plt.ylim(-0.02,0.08)
	#plt.ylim(9.5e-6,0.11)
	#plt.yticks([1e-4,1e-3,1e-2,1e-1],visible=yticks, fontsize=fontsize)
	plt.xticks([1,10,100],visible=True, fontsize=fontsize)
	plt.yticks([-0.02, 0, 0.02, 0.04, 0.06],visible=yticks, fontsize=fontsize)
	if facecolour is None:
		facecolour = colour
	if marker=='.':
		s=None
	else:
		s = 2.0

	plt.axhline(0,color='k',ls=':')

	x,w = np.loadtxt('%s/2pt/txt/wpp_%d.txt'%(sim, snapshot)).T
	if errors:
		C = np.loadtxt('%s/cov/covmat-analytic.txt'%sim)
		dy = np.sqrt(np.diag(C))[lim[0]:lim[1]]
	else:
		dy = np.zeros_like(x)
	mask = w!=0
	#import pdb ; pdb.set_trace()
	plt.errorbar(x[mask], x[mask]*w[mask], yerr=x[mask]*dy[mask], 
		marker=marker,
		markeredgecolor=colour, 
		markersize=s,
		ecolor=colour,
		markerfacecolor=facecolour, 
		linestyle='none', 
		label=label)
	#plt.axhline(0,color='k',ls=':')


def plot_theory_pp(sim, snapshot, colour):
	plt.xscale('log')
	#plt.yscale('log')
	plt.xlim(8e-2,210)
	plt.ylim(-0.02,0.08)
	#plt.ylim(1e-4,5e-1)
	plt.yticks([-0.02, 0, 0.02, 0.04, 0.06],visible=True, fontsize=fontsize)
	plt.xticks([1,10,100],visible=True, fontsize=fontsize)

	w = np.loadtxt('data/theory/%s/wpp_%d.txt'%(sim, snapshot))
	x = np.loadtxt('data/theory/%s/r_p.txt'%(sim))
	
	plt.plot(x, x*w, ls='-', lw=1.5,color=colour)



def plot_theory_gp(sim, snapshot, colour):
	plt.xscale('log')
	#plt.yscale('log')
	plt.xlim(8e-2,210)
	plt.ylim(-0.1,2.3)
	#plt.ylim(1.5e-3,5)
	#plt.xticks([0.1,1,10,100],visible=False)
	plt.yticks([0,0.4,0.8,1.2,1.6,2.],visible=True, fontsize=fontsize)

	w = np.loadtxt('data/theory/%s/wgp_%d.txt'%(sim, snapshot))
	x = np.loadtxt('data/theory/%s/r_p.txt'%(sim))

	plt.plot(x, x*w, ls='-', lw=1.5,color=colour)

def plot_theory_gg(sim, snapshot, colour):
	plt.xscale('log')
	plt.xlim(8e-2,210)
	plt.ylim(-20,120)
	plt.xticks([0.1,1,10,100],visible=False)
	plt.axhline(0,color='k',ls=':')


	w = np.loadtxt('data/theory/%s/wgg_%d.txt'%(sim, snapshot))
	x = np.loadtxt('data/theory/%s/r_p.txt'%(sim))

	plt.plot(x, x*w, ls='-', lw=1.5,color=colour)



plt.subplot(341)
plt.axvspan(0.001,6,color='grey',alpha=0.1)
plt.title(r'$z=0.0$', fontsize=12)
plt.ylabel(r'$r_\mathrm{p} \times w_{gg}$', fontsize=fontsize)


#plot_gg('illustris', 135, colours['illustris'], errors=True, lim=[-8,-1])
plot_theory_gg('mbii', 85, colours['mbii'])
plot_theory_gg('tng', 99, colours['tng'])
plot_theory_gg('illustris', 135, colours['illustris'])

plot_gg('tng', 99, colours['tng'], errors=True, lim=[96,108])
plot_gg('mbii', 85, colours['mbii'], errors=True, marker='D', lim=[96,108])
plot_gg('illustris', 135, colours['illustris'], errors=True, lim=[64,72], facecolour='white')



plt.subplot(342)
plt.axvspan(0.001,6,color='grey',alpha=0.1)
#plt.yticks(visible=False)
plt.title(r'$z=0.30$', fontsize=12)

plot_theory_gg('mbii', 79, colours['mbii'])
plot_theory_gg('tng', 78, colours['tng'])
plot_theory_gg('illustris', 114, colours['illustris'])

plot_gg('tng', 78, colours['tng'], errors=True, lim=[108,120], yticks=False)
plot_gg('mbii', 79, colours['mbii'], errors=True, marker='D', lim=[108,120], yticks=False)
plot_gg('illustris', 114, colours['illustris'], errors=True, lim=[72,80], yticks=False, facecolour='white')



plt.subplot(343)
plt.axvspan(0.001,6,color='grey',alpha=0.1)
#plt.yticks(visible=False)
plt.title(r'$z=0.62$', fontsize=12)

plot_theory_gg('mbii', 73, colours['mbii'])
plot_theory_gg('tng', 62, colours['tng'])
plot_theory_gg('illustris', 98, colours['illustris'])

plot_gg('tng', 62, colours['tng'], errors=True, lim=[120,132], yticks=False)
plot_gg('mbii', 73, colours['mbii'], errors=True, marker='D', lim=[120,132], yticks=False)
plot_gg('illustris', 98, colours['illustris'], errors=True, lim=[80,88], yticks=False, facecolour='white')



plt.subplot(344)
plt.axvspan(0.001,6,color='grey',alpha=0.1)
#plt.yticks(visible=False)

plt.title(r'$z=1.0$', fontsize=12)

plot_theory_gg('mbii', 68, colours['mbii'])
plot_theory_gg('tng', 50, colours['tng'])
plot_theory_gg('illustris', 85, colours['illustris'])

plot_gg('tng', 50, colours['tng'], errors=True, lim=[132,144], yticks=False)
plot_gg('mbii', 68, colours['mbii'], errors=True, marker='D', lim=[132,144], yticks=False)
plot_gg('illustris', 85, colours['illustris'], errors=True, lim=[88,96], yticks=False, facecolour='white')





plt.subplot(345)
plt.axvspan(0.001,6,color='grey',alpha=0.1)

plt.ylabel(r'$r_{\rm p} \times w_{g+}$', fontsize=fontsize)
#plt.yticks([1e-2,1e-1,1],visible=True, fontsize=fontsize)

plot_theory_gp('mbii', 85, colours['mbii'])
plot_theory_gp('tng', 99, colours['tng'])
plot_theory_gp('illustris', 135, colours['illustris'])

plot_gp('tng', 99, colours['tng'], errors=True, lim=[0,12])
plot_gp('mbii', 85, colours['mbii'], errors=True, marker='D', lim=[0,12])
plot_gp('illustris', 135, colours['illustris'], errors=True, lim=[0,8], facecolour='white')

plt.subplot(346)
plt.axvspan(0.001,6,color='grey',alpha=0.1)
#plt.yticks(visible=False)

plot_theory_gp('mbii', 79, colours['mbii'])
plot_theory_gp('tng', 78, colours['tng'])
plot_theory_gp('illustris', 114, colours['illustris'])

plot_gp('tng', 78, colours['tng'], errors=True, lim=[12,24], yticks=False)
plot_gp('mbii', 79, colours['mbii'], errors=True, marker='D', lim=[12,24], yticks=False)
plot_gp('illustris', 114, colours['illustris'], errors=True, lim=[8,16], yticks=False, facecolour='white')


plt.subplot(347)
plt.axvspan(0.001,6,color='grey',alpha=0.1)
#plt.yticks(visible=False)

plot_theory_gp('mbii', 73, colours['mbii'])
plot_theory_gp('tng', 62, colours['tng'])
plot_theory_gp('illustris', 98, colours['illustris'])

plot_gp('tng', 62, colours['tng'], errors=True, lim=[24,36], yticks=False)
plot_gp('mbii', 73, colours['mbii'], errors=True, marker='D', lim=[24,36], yticks=False)
plot_gp('illustris', 98, colours['illustris'], errors=True, lim=[16,24], yticks=False, facecolour='white')


plt.subplot(348)
plt.axvspan(0.001,6,color='grey',alpha=0.1)
#plt.yticks(visible=False)

plot_theory_gp('mbii', 68, colours['mbii'])
plot_theory_gp('tng', 50, colours['tng'])
plot_theory_gp('illustris', 85, colours['illustris'])

plot_gp('tng', 50, colours['tng'], errors=True, lim=[36,48], yticks=False)
plot_gp('mbii', 68, colours['mbii'], errors=True, marker='D', lim=[36,48], yticks=False)
plot_gp('illustris', 85, colours['illustris'], errors=True, lim=[24,32], yticks=False, label='Illustris', facecolour='white')
plt.legend(loc='upper left', fontsize=9)



plt.subplot(349)
plt.axvspan(0.001,6,color='grey',alpha=0.1)

plt.ylabel(r'$r_{\rm p} \times w_{++}$', fontsize=fontsize)
#plt.yticks(visible=False)
plt.xlabel(r'$r_{\rm p}$ / $h^{-1}$ Mpc', fontsize=fontsize)

plot_theory_pp('mbii', 85, colours['mbii'])
plot_theory_pp('tng', 99, colours['tng'])
plot_theory_pp('illustris', 135, colours['illustris'])

plot_pp('tng', 99, colours['tng'], errors=True, lim=[48,60], yticks=True)
plot_pp('mbii', 85, colours['mbii'], errors=True, marker='D', lim=[48,60], yticks=True)
plot_pp('illustris', 135, colours['illustris'], errors=True, lim=[32,40], facecolour='white', yticks=True)
#plt.yticks([-0.02, 0, 0.02, 0.04, 0.06],visible=True, fontsize=fontsize)


plt.subplot(3,4,10)
plt.axvspan(0.001,6,color='grey',alpha=0.1)
#plt.yticks(visible=False)

plt.xlabel(r'$r_{\rm p}$ / $h^{-1}$ Mpc', fontsize=fontsize)

plot_theory_pp('mbii', 79, colours['mbii'])
plot_theory_pp('tng', 78, colours['tng'])
plot_theory_pp('illustris', 114, colours['illustris'])

plot_pp('tng', 78, colours['tng'], errors=True, lim=[60,72], yticks=False)
plot_pp('mbii', 79, colours['mbii'], errors=True, marker='D', lim=[60,72], yticks=False)
plot_pp('illustris', 114, colours['illustris'], errors=True, lim=[40,48], yticks=False, facecolour='white')

plt.subplot(3,4,11)
plt.axvspan(0.001,6,color='grey',alpha=0.1)
#plt.yticks(visible=False)

plt.xlabel(r'$r_{\rm p}$ / $h^{-1}$ Mpc', fontsize=fontsize)

plot_theory_pp('mbii', 73, colours['mbii'])
plot_theory_pp('tng', 62, colours['tng'])
plot_theory_pp('illustris', 98, colours['illustris'])

plot_pp('tng', 62, colours['tng'], errors=True, lim=[72,84], yticks=False, label='TNG')
plot_pp('mbii', 73, colours['mbii'], errors=True, marker='D', lim=[72,84], yticks=False)
plot_pp('illustris', 98, colours['illustris'], errors=True, lim=[48,56], yticks=False, facecolour='white')
plt.legend(loc='upper left', fontsize=9)

plt.subplot(3,4,12)
plt.axvspan(0.001,6,color='grey',alpha=0.1)
#plt.yticks(visible=False)

plt.xlabel(r'$r_{\rm p}$ / $h^{-1}$ Mpc', fontsize=fontsize)

plot_theory_pp('mbii', 68, colours['mbii'])
plot_theory_pp('tng', 50, colours['tng'])
plot_theory_pp('illustris', 85, colours['illustris'])

plot_pp('tng', 50, colours['tng'], errors=True, lim=[84,96], yticks=False)
plot_pp('mbii', 68, colours['mbii'], errors=True, marker='D', lim=[84,96], yticks=False, label='MBII')
plot_pp('illustris', 85, colours['illustris'], errors=True, lim=[56,64], yticks=False, facecolour='white')
plt.legend(loc='upper left', fontsize=9)



plt.subplots_adjust(bottom=0.14,left=0.14, wspace=0.0, hspace=0.15, right=0.98)
plt.savefig('dvecs_panels.pdf')
plt.savefig('dvecs_panels.png')
