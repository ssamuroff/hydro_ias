import numpy as np
import pylab as plt
plt.switch_backend('pdf')
plt.style.use('y1a1')

from matplotlib import rcParams
import fitsio as fi

fillcol='plum'

rcParams['xtick.major.size'] = 3.5
rcParams['xtick.minor.size'] = 1.7
rcParams['ytick.major.size'] = 3.5
rcParams['ytick.minor.size'] = 1.7
rcParams['xtick.direction']='in'
rcParams['ytick.direction']='in'

#rcParams['xtick.minor.visible']=True

theory_base = '/home/ssamurof/hydro_ias/data/theory/'

fontsize=10
colours = {'mbii':'midnightblue','illustris':'pink','tng':'darkmagenta'}
xmin = 8e-2
xmax = 210
dx=10**0.05

indices = {('tng', 99): 0, ('tng',78):1, ('tng',62):2, ('tng',50):3,
           ('mbii_w', 85): 0, ('mbii_w',79):1, ('mbii_w',73):2, ('mbii_w', 68):3,
           ('illustris_w', 135): 0, ('illustris_w', 114):1, ('illustris_w',98):2, ('illustris_w',85):3,}

include_theory=False


def plot_gg(sim, snapshot, colour, errors=True, lim=[0,20], yticks=True, label=None, marker='.', facecolour=None):
	plt.xscale('log')
	plt.xlim(8e-2,210)
	plt.ylim(-35,120)
	plt.xticks([0.1,1,10,100],visible=False)
	plt.axhline(0,color='k',ls=':')
	plt.yticks([-20, 0,20,40,60,80,100],visible=yticks, fontsize=fontsize)
	if facecolour is None:
		facecolour = colour
	if marker=='.':
		s=None
	else:
		s = 4.75

	f = fi.FITS('data/2pt/fits/2pt_%s_fidcovw_iteration4.fits'%sim)
	#import pdb ; pdb.set_trace()
	mask = f['wgg'].read()['BIN']==indices[(sim,snapshot)]
	x = f['wgg'].read()['SEP'][mask]
	w = f['wgg'].read()['VALUE'][mask]

	#x,w = np.loadtxt('%s/2pt/txt/wgg_%d.txt'%(sim, snapshot)).T
	if errors:
		C = f['COVMAT'].read()
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
	plt.yscale('linear')
	plt.xlim(8e-2,210)
	#plt.ylim(-0.02,0.06)
	plt.ylim(-0.4,2.5)
	plt.xticks([0.1,1,10,100],visible=False)
	#plt.yticks([1e-2,1e-1,1e0],visible=yticks, fontsize=fontsize)
	plt.yticks([0.,0.5,1,1.5,2],visible=yticks, fontsize=fontsize)
	if facecolour is None:
		facecolour = colour
	if marker=='.':
		s=None
	else:
		s = 4.75

	f = fi.FITS('data/2pt/fits/2pt_%s_fidcovw_iteration4.fits'%sim)
	#import pdb ; pdb.set_trace()
	mask = f['wgp'].read()['BIN']==indices[(sim,snapshot)]
	x = f['wgp'].read()['SEP'][mask]
	w = f['wgp'].read()['VALUE'][mask]

	#x,w = np.loadtxt('%s/2pt/txt/wgg_%d.txt'%(sim, snapshot)).T
	if errors:
		C = f['COVMAT'].read()
		dy = np.sqrt(np.diag(C))[lim[0]:lim[1]]
	else:
		dy = np.zeros_like(x)

	plt.axhline(0,color='k',ls=':')

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
	plt.yscale('linear')
	plt.xlim(8e-2,210)
	plt.ylim(-0.03,0.06)
#	plt.ylim(-0.005,0.025)
	#plt.ylim(9.5e-6,0.11)
	plt.yticks([-0.02,0,0.02,0.04,0.06],visible=yticks, fontsize=fontsize)
	#plt.yticks([-0.01, -0.005, 0, 0.005, 0.01],visible=yticks, fontsize=fontsize)
	if facecolour is None:
		facecolour = colour
	if marker=='.':
		s=None
	else:
		s = 4.75

	f = fi.FITS('data/2pt/fits/2pt_%s_fidcovw_iteration4.fits'%sim)
	#import pdb ; pdb.set_trace()
	mask = f['wpp'].read()['BIN']==indices[(sim,snapshot)]
	x = f['wpp'].read()['SEP'][mask]
	w = f['wpp'].read()['VALUE'][mask]

	#x,w = np.loadtxt('%s/2pt/txt/wgg_%d.txt'%(sim, snapshot)).T
	if errors:
		C = f['COVMAT'].read()
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
	plt.axhline(0,color='k',ls=':')


def plot_theory_pp(sim, redshift, colour):
	plt.xscale('log')
	plt.yscale('linear')
	plt.xlim(8e-2,210)
	plt.ylim(-0.03,0.06)

	w = np.loadtxt('data/theory/%s/wpp_%d.txt'%(sim, redshift))
	x = np.loadtxt('data/theory/%s/r_p.txt'%(sim))
	
	plt.plot(x, w, ls='-', lw=1.5,color=colour)



def plot_theory_gp(sim, redshift, colour):
	plt.xscale('log')
	plt.yscale('log')
	plt.xlim(8e-2,210)
	plt.ylim(1.5e-3,5)
	plt.xticks([0.1,1,10,100],visible=False)

	w = np.loadtxt('data/theory/%s/wgp_%d.txt'%(sim, redshift))
	x = np.loadtxt('data/theory/%s/r_p.txt'%(sim))

	plt.plot(x, w, ls='-', lw=1.5,color=colour)

def plot_theory_gg(sim, redshift, colour):
	plt.xscale('log')
	plt.xlim(8e-2,210)
	plt.ylim(-20,120)
	plt.xticks([0.1,1,10,100],visible=False)
	plt.axhline(0,color='k',ls=':')

	import pdb ; pdb.set_trace()


	w = np.loadtxt('data/theory/%s/galaxy_w/w_rp_limber_%1.3f.txt'%(sim, redshift))
	x = np.loadtxt('data/theory/%s/galaxy_w/r_p.txt'%(sim))

	plt.plot(x, x*w, ls='-', lw=1.,color=colour)



plt.subplot(341)
plt.axvspan(0.001,6,color=fillcol,alpha=0.3)
plt.title(r'$z=0.0$', fontsize=12)
plt.ylabel(r'$r_\mathrm{p} \times w_{gg} \; [h^{-1} \mathrm{Mpc}]$', fontsize=fontsize)


#plot_gg('illustris', 135, colours['illustris'], errors=True, lim=[-8,-1])


plot_gg('tng', 99, colours['tng'], errors=True, lim=[96,108])
plot_gg('mbii_w', 85, colours['mbii'], errors=True, marker='*', lim=[96,108], facecolour='white')
plot_gg('illustris_w', 135, colours['illustris'], errors=True, lim=[64,72], facecolour='white')


xt = np.loadtxt(theory_base+'tng/galaxy_w/r_p.txt')
yt = np.loadtxt(theory_base+'tng/galaxy_w/w_rp_limber_0.000.txt')
plt.plot(xt,xt*yt,color=colours['tng'])

xt = np.loadtxt(theory_base+'mbii/galaxy_w/r_p.txt')
yt = np.loadtxt(theory_base+'mbii/galaxy_w/w_rp_limber_0.062.txt')
plt.plot(xt,xt*yt,color=colours['mbii'])

xt = np.loadtxt(theory_base+'illustris/galaxy_w/r_p.txt')
yt = np.loadtxt(theory_base+'illustris/galaxy_w/w_rp_limber_0.000.txt')
plt.plot(xt,xt*yt,color=colours['illustris'])



if include_theory:



	plot_theory_gg('mbii', 85, colours['mbii'])
	plot_theory_gg('tng', 99, colours['tng'])
	plot_theory_gg('illustris', 135, colours['illustris'])


plt.subplot(342)
plt.axvspan(0.001,6,color=fillcol,alpha=0.3)
plt.yticks(visible=False)
plt.title(r'$z=0.30$', fontsize=12)

if include_theory:
	plot_theory_gg('mbii', 79, colours['mbii'])
	plot_theory_gg('tng', 78, colours['tng'])
	plot_theory_gg('illustris', 114, colours['illustris'])

plot_gg('tng', 78, colours['tng'], errors=True, lim=[108,120], yticks=False)
plot_gg('mbii_w', 79, colours['mbii'], errors=True, marker='*', lim=[108,120], yticks=False, facecolour='white')
plot_gg('illustris_w', 114, colours['illustris'], errors=True, lim=[72,80], yticks=False, facecolour='white')

xt = np.loadtxt(theory_base+'tng/galaxy_w/r_p.txt')
yt = np.loadtxt(theory_base+'tng/galaxy_w/w_rp_limber_0.300.txt')
plt.plot(xt,xt*yt,color=colours['tng'])

xt = np.loadtxt(theory_base+'mbii/galaxy_w/r_p.txt')
yt = np.loadtxt(theory_base+'mbii/galaxy_w/w_rp_limber_0.300.txt')
plt.plot(xt,xt*yt,color=colours['mbii'])

xt = np.loadtxt(theory_base+'illustris/galaxy_w/r_p.txt')
yt = np.loadtxt(theory_base+'illustris/galaxy_w/w_rp_limber_0.300.txt')
plt.plot(xt,xt*yt,color=colours['illustris'])





plt.subplot(343)
plt.axvspan(0.001,6,color=fillcol,alpha=0.3)
plt.yticks(visible=False)
plt.title(r'$z=0.62$', fontsize=12)

if include_theory:
	plot_theory_gg('mbii', 73, colours['mbii'])
	plot_theory_gg('tng', 62, colours['tng'])
	plot_theory_gg('illustris', 98, colours['illustris'])

plot_gg('tng', 62, colours['tng'], errors=True, lim=[120,132], yticks=False)
plot_gg('mbii_w', 73, colours['mbii'], errors=True, marker='*', lim=[120,132], yticks=False, facecolour='white')
plot_gg('illustris_w', 98, colours['illustris'], errors=True, lim=[80,88], yticks=False, facecolour='white')


xt = np.loadtxt(theory_base+'tng/galaxy_w/r_p.txt')
yt = np.loadtxt(theory_base+'tng/galaxy_w/w_rp_limber_0.625.txt')
plt.plot(xt,xt*yt,color=colours['tng'])


xt = np.loadtxt(theory_base+'mbii/galaxy_w/r_p.txt')
yt = np.loadtxt(theory_base+'mbii/galaxy_w/w_rp_limber_0.625.txt')
plt.plot(xt,xt*yt,color=colours['mbii'])

xt = np.loadtxt(theory_base+'illustris/galaxy_w/r_p.txt')
yt = np.loadtxt(theory_base+'illustris/galaxy_w/w_rp_limber_0.625.txt')
plt.plot(xt,xt*yt,color=colours['illustris'])





plt.subplot(344)
plt.axvspan(0.001,6,color=fillcol,alpha=0.3)
plt.yticks(visible=False)

plt.title(r'$z=1.0$', fontsize=12)

if include_theory:
	plot_theory_gg('mbii', 68, colours['mbii'])
	plot_theory_gg('tng', 50, colours['tng'])
	plot_theory_gg('illustris', 85, colours['illustris'])

plot_gg('tng', 50, colours['tng'], errors=True, lim=[132,144], yticks=False)
plot_gg('mbii_w', 68, colours['mbii'], errors=True, marker='*', lim=[132,144], yticks=False, facecolour='white')
plot_gg('illustris_w', 85, colours['illustris'], errors=True, lim=[88,96], yticks=False, facecolour='white')


xt = np.loadtxt(theory_base+'tng/galaxy_w/r_p.txt')
yt = np.loadtxt(theory_base+'tng/galaxy_w/w_rp_limber_1.000.txt')
plt.plot(xt,xt*yt,color=colours['tng'])


xt = np.loadtxt(theory_base+'mbii/galaxy_w/r_p.txt')
yt = np.loadtxt(theory_base+'mbii/galaxy_w/w_rp_limber_1.000.txt')
plt.plot(xt,xt*yt,color=colours['mbii'])

xt = np.loadtxt(theory_base+'illustris/galaxy_w/r_p.txt')
yt = np.loadtxt(theory_base+'illustris/galaxy_w/w_rp_limber_1.000.txt')
plt.plot(xt,xt*yt,color=colours['illustris'])




plt.subplot(345)
plt.axvspan(0.001,6,color=fillcol,alpha=0.3)

plt.ylabel(r'$r_{\rm p} \times w_{g+} \; [h^{-1} \mathrm{Mpc}]$', fontsize=fontsize)
#plt.yticks([1e-2,1e-1,1],visible=True, fontsize=fontsize)

if include_theory:
	plot_theory_gp('mbii', 85, colours['mbii'])
	plot_theory_gp('tng', 99, colours['tng'])
	plot_theory_gp('illustris', 135, colours['illustris'])

plot_gp('tng', 99, colours['tng'], errors=True, lim=[0,12], label='TNG')
plot_gp('mbii_w', 85, colours['mbii'], errors=True, marker='*', lim=[0,12], label='MBII', facecolour='white')
plot_gp('illustris_w', 135, colours['illustris'], errors=True, lim=[0,8], facecolour='white', label='Illustris')

xt = np.loadtxt(theory_base+'tng/galaxy_intrinsic_w/r_p.txt')
yt = np.loadtxt(theory_base+'tng/galaxy_intrinsic_w/w_rp_limber_0.000.txt')
plt.plot(xt,xt*yt,color=colours['tng'])

xt = np.loadtxt(theory_base+'mbii/galaxy_intrinsic_w/r_p.txt')
yt = np.loadtxt(theory_base+'mbii/galaxy_intrinsic_w/w_rp_limber_0.062_tatt.txt')
plt.plot(xt,xt*yt,color=colours['mbii'])

xt = np.loadtxt(theory_base+'illustris/galaxy_intrinsic_w/r_p.txt')
yt = np.loadtxt(theory_base+'illustris/galaxy_intrinsic_w/w_rp_limber_0.000.txt')
plt.plot(xt,xt*yt,color=colours['illustris'])




plt.legend(loc='upper left', fontsize=8)


plt.subplot(346)
plt.axvspan(0.001,6,color=fillcol,alpha=0.3)
plt.yticks(visible=False)

if include_theory:
	plot_theory_gp('mbii', 79, colours['mbii'])
	plot_theory_gp('tng', 78, colours['tng'])
	plot_theory_gp('illustris', 114, colours['illustris'])

plot_gp('tng', 78, colours['tng'], errors=True, lim=[12,24], yticks=False)
plot_gp('mbii_w', 79, colours['mbii'], errors=True, marker='*', lim=[12,24], yticks=False, facecolour='white')
plot_gp('illustris_w', 114, colours['illustris'], errors=True, lim=[8,16], yticks=False, facecolour='white')

xt = np.loadtxt(theory_base+'tng/galaxy_intrinsic_w/r_p.txt')
yt = np.loadtxt(theory_base+'tng/galaxy_intrinsic_w/w_rp_limber_0.300.txt')
plt.plot(xt,xt*yt,color=colours['tng'])

xt = np.loadtxt(theory_base+'mbii/galaxy_intrinsic_w/r_p.txt')
yt = np.loadtxt(theory_base+'mbii/galaxy_intrinsic_w/w_rp_limber_0.300.txt')
plt.plot(xt,xt*yt,color=colours['mbii'])

xt = np.loadtxt(theory_base+'illustris/galaxy_intrinsic_w/r_p.txt')
yt = np.loadtxt(theory_base+'illustris/galaxy_intrinsic_w/w_rp_limber_0.300.txt')
plt.plot(xt,xt*yt,color=colours['illustris'])






plt.subplot(347)
plt.axvspan(0.001,6,color=fillcol,alpha=0.3)
plt.yticks(visible=False)

if include_theory:
	plot_theory_gp('mbii', 73, colours['mbii'])
	plot_theory_gp('tng', 62, colours['tng'])
	plot_theory_gp('illustris', 98, colours['illustris'])

plot_gp('tng', 62, colours['tng'], errors=True, lim=[24,36], yticks=False)
plot_gp('mbii_w', 73, colours['mbii'], errors=True, marker='*', lim=[24,36], yticks=False, facecolour='white')
plot_gp('illustris_w', 98, colours['illustris'], errors=True, lim=[16,24], yticks=False, facecolour='white')

xt = np.loadtxt(theory_base+'tng/galaxy_intrinsic_w/r_p.txt')
yt = np.loadtxt(theory_base+'tng/galaxy_intrinsic_w/w_rp_limber_0.625.txt')
plt.plot(xt,xt*yt,color=colours['tng'])

xt = np.loadtxt(theory_base+'mbii/galaxy_intrinsic_w/r_p.txt')
yt = np.loadtxt(theory_base+'mbii/galaxy_intrinsic_w/w_rp_limber_0.625.txt')
plt.plot(xt,xt*yt,color=colours['mbii'])

xt = np.loadtxt(theory_base+'illustris/galaxy_intrinsic_w/r_p.txt')
yt = np.loadtxt(theory_base+'illustris/galaxy_intrinsic_w/w_rp_limber_0.625.txt')
plt.plot(xt,xt*yt,color=colours['illustris'])






plt.subplot(348)
plt.axvspan(0.001,6,color=fillcol,alpha=0.3)
plt.yticks(visible=False)

if include_theory:
	plot_theory_gp('mbii', 68, colours['mbii'])
	plot_theory_gp('tng', 50, colours['tng'])
	plot_theory_gp('illustris', 85, colours['illustris'])

plot_gp('tng', 50, colours['tng'], errors=True, lim=[36,48], yticks=False)
plot_gp('mbii_w', 68, colours['mbii'], errors=True, marker='*', lim=[36,48], yticks=False, facecolour='white')
plot_gp('illustris_w', 85, colours['illustris'], errors=True, lim=[24,32], yticks=False, label='Illustris', facecolour='white')
#plt.legend(loc='upper right', fontsize=9)


xt = np.loadtxt(theory_base+'tng/galaxy_intrinsic_w/r_p.txt')
yt = np.loadtxt(theory_base+'tng/galaxy_intrinsic_w/w_rp_limber_1.000.txt')
plt.plot(xt,xt*yt,color=colours['tng'])

xt = np.loadtxt(theory_base+'mbii/galaxy_intrinsic_w/r_p.txt')
yt = np.loadtxt(theory_base+'mbii/galaxy_intrinsic_w/w_rp_limber_1.000.txt')
plt.plot(xt,xt*yt,color=colours['mbii'])

xt = np.loadtxt(theory_base+'illustris/galaxy_intrinsic_w/r_p.txt')
yt = np.loadtxt(theory_base+'illustris/galaxy_intrinsic_w/w_rp_limber_1.000.txt')
plt.plot(xt,xt*yt,color=colours['illustris'])





plt.subplot(349)
plt.axvspan(0.001,6,color=fillcol,alpha=0.3)

plt.ylabel(r'$r_{\rm p} \times w_{++} \; [h^{-1} \mathrm{Mpc}]$', fontsize=fontsize)
#plt.yticks([1e-5, 1e-3,1e-1],visible=False, fontsize=fontsize)
plt.xlabel(r'$r_{\rm p}$ $\; [h^{-1} \mathrm{Mpc}]$', fontsize=fontsize)

if include_theory:
	plot_theory_pp('mbii', 85, colours['mbii'])
	plot_theory_pp('tng', 99, colours['tng'])
	plot_theory_pp('illustris', 135, colours['illustris'])

plot_pp('tng', 99, colours['tng'], errors=True, lim=[48,60])
plot_pp('mbii_w', 85, colours['mbii'], errors=True, marker='*', lim=[48,60], facecolour='white')
plot_pp('illustris_w', 135, colours['illustris'], errors=True, lim=[32,40], facecolour='white')

xt = np.loadtxt(theory_base+'tng/intrinsic_w/r_p.txt')
yt = np.loadtxt(theory_base+'tng/intrinsic_w/w_rp_limber_0.000.txt')
plt.plot(xt,xt*yt,color=colours['tng'])

xt = np.loadtxt(theory_base+'mbii/intrinsic_w/r_p.txt')
yt = np.loadtxt(theory_base+'mbii/intrinsic_w/w_rp_limber_0.062_tatt.txt')
plt.plot(xt,xt*yt,color=colours['mbii'])

xt = np.loadtxt(theory_base+'illustris/intrinsic_w/r_p.txt')
yt = np.loadtxt(theory_base+'illustris/intrinsic_w/w_rp_limber_0.000.txt')
plt.plot(xt,xt*yt,color=colours['illustris'])




plt.subplot(3,4,10)
plt.axvspan(0.001,6,color=fillcol,alpha=0.3)
plt.yticks(visible=False)

plt.xlabel(r'$r_{\rm p} \; [h^{-1} \mathrm{Mpc}]$', fontsize=fontsize)

if include_theory:
	plot_theory_pp('mbii', 79, colours['mbii'])
	plot_theory_pp('tng', 78, colours['tng'])
	plot_theory_pp('illustris', 114, colours['illustris'])

plot_pp('tng', 78, colours['tng'], errors=True, lim=[60,72], yticks=False)
plot_pp('mbii_w', 79, colours['mbii'], errors=True, marker='*', lim=[60,72], yticks=False, facecolour='white')
plot_pp('illustris_w', 114, colours['illustris'], errors=True, lim=[40,48], yticks=False, facecolour='white')

xt = np.loadtxt(theory_base+'tng/intrinsic_w/r_p.txt')
yt = np.loadtxt(theory_base+'tng/intrinsic_w/w_rp_limber_0.300.txt')
plt.plot(xt,xt*yt,color=colours['tng'])


xt = np.loadtxt(theory_base+'mbii/intrinsic_w/r_p.txt')
yt = np.loadtxt(theory_base+'mbii/intrinsic_w/w_rp_limber_0.300.txt')
plt.plot(xt,xt*yt,color=colours['mbii'])

xt = np.loadtxt(theory_base+'illustris/intrinsic_w/r_p.txt')
yt = np.loadtxt(theory_base+'illustris/intrinsic_w/w_rp_limber_0.300.txt')
plt.plot(xt,xt*yt,color=colours['illustris'])






plt.subplot(3,4,11)
plt.axvspan(0.001,6,color=fillcol,alpha=0.3)
plt.yticks(visible=False)

plt.xlabel(r'$r_{\rm p} \; [h^{-1} \mathrm{Mpc}]$', fontsize=fontsize)

if include_theory:
	plot_theory_pp('mbii', 73, colours['mbii'])
	plot_theory_pp('tng', 62, colours['tng'])
	plot_theory_pp('illustris', 98, colours['illustris'])

plot_pp('tng', 62, colours['tng'], errors=True, lim=[72,84], yticks=False, label='TNG')
plot_pp('mbii_w', 73, colours['mbii'], errors=True, marker='*', lim=[72,84], yticks=False, facecolour='white')
plot_pp('illustris_w', 98, colours['illustris'], errors=True, lim=[48,56], yticks=False, facecolour='white')
#plt.legend(loc='upper right', fontsize=9)

xt = np.loadtxt(theory_base+'tng/intrinsic_w/r_p.txt')
yt = np.loadtxt(theory_base+'tng/intrinsic_w/w_rp_limber_0.625.txt')
plt.plot(xt,xt*yt,color=colours['tng'])

xt = np.loadtxt(theory_base+'mbii/intrinsic_w/r_p.txt')
yt = np.loadtxt(theory_base+'mbii/intrinsic_w/w_rp_limber_0.625.txt')
plt.plot(xt,xt*yt,color=colours['mbii'])

xt = np.loadtxt(theory_base+'illustris/intrinsic_w/r_p.txt')
yt = np.loadtxt(theory_base+'illustris/intrinsic_w/w_rp_limber_0.625.txt')
plt.plot(xt,xt*yt,color=colours['illustris'])





plt.subplot(3,4,12)
plt.axvspan(0.001,6,color=fillcol,alpha=0.3)
plt.yticks(visible=False)

plt.xlabel(r'$r_{\rm p} \; [h^{-1} \mathrm{Mpc}]$', fontsize=fontsize)

if include_theory:
	plot_theory_pp('mbii', 68, colours['mbii'])
	plot_theory_pp('tng', 50, colours['tng'])
	plot_theory_pp('illustris', 85, colours['illustris'])

plot_pp('tng', 50, colours['tng'], errors=True, lim=[84,96], yticks=False)
plot_pp('mbii_w', 68, colours['mbii'], errors=True, marker='*', lim=[84,96], yticks=False, facecolour='white', label='MBII')
plot_pp('illustris_w', 85, colours['illustris'], errors=True, lim=[56,64], yticks=False, facecolour='white')
#plt.legend(loc='upper right', fontsize=9)

xt = np.loadtxt(theory_base+'tng/intrinsic_w/r_p.txt')
yt = np.loadtxt(theory_base+'tng/intrinsic_w/w_rp_limber_1.000.txt')
plt.plot(xt,xt*yt,color=colours['tng'])

xt = np.loadtxt(theory_base+'mbii/intrinsic_w/r_p.txt')
yt = np.loadtxt(theory_base+'mbii/intrinsic_w/w_rp_limber_1.000.txt')
plt.plot(xt,xt*yt,color=colours['mbii'])

xt = np.loadtxt(theory_base+'illustris/intrinsic_w/r_p.txt')
yt = np.loadtxt(theory_base+'illustris/intrinsic_w/w_rp_limber_1.000.txt')
plt.plot(xt,xt*yt,color=colours['illustris'])





plt.subplots_adjust(bottom=0.14,left=0.14, wspace=0.0, hspace=0.15, right=0.98)
plt.savefig('dvecs_panels.pdf')
plt.savefig('dvecs_panels.png')
