import numpy as np
import pylab as plt
import sys
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

path1 = sys.argv[-2]
path2 = sys.argv[-1]

D1 = np.sqrt(np.diag(np.loadtxt(path1)))
D2 = np.sqrt(np.diag(np.loadtxt(path2)))

label1 = 'jackknife'
label2 = 'analytic'

x,w=np.loadtxt('mbii/2pt/txt/wgg_85.txt').T

def set_basic(ctype, yticks=True):
	plt.axvspan(0.001,6,color='grey',alpha=0.1)
	plt.xscale('log')
	plt.yscale('log')
	plt.xlim(8e-2,210)
	plt.xticks([0.1,1,10,100],visible=False)

	if ctype=='gg':
		plt.ylim(0.7,40)
		plt.yticks([1e0,1e1],visible=yticks, fontsize=fontsize)

	elif ctype=='gp':
		plt.ylim(7e-3,2e-1)
		plt.yticks([1e-2,1e-1],visible=yticks, fontsize=fontsize)

	elif ctype=='pp':
		plt.ylim(5e-5,5e-2)
		plt.yticks([1e-4,1e-3,1e-2],visible=yticks, fontsize=fontsize)



fontsize=10
colours = {'mbii':'midnightblue','illustris':'forestgreen','tng':'darkmagenta'}
xmin = 8e-2
xmax = 210
dx=10**0.05


# eBOSS
plt.subplot(341)
set_basic('gg', yticks=True)

plt.title(r'$z=0.0$', fontsize=12)
plt.ylabel(r'$\sigma ( w_{gg} )$', fontsize=fontsize)

dy1 = D1[96:108]
dy2 = D2[96:108]

plt.plot(x, dy1, color='midnightblue', linestyle='-', lw=1.5)
plt.plot(x, dy2, color='k', linestyle='--', lw=1.5)



plt.subplot(342)
set_basic('gg', yticks=False)

plt.title(r'$z=0.30$', fontsize=12)

dy1 = D1[108:120]
dy2 = D2[108:120]

plt.plot(x, dy1, color='midnightblue', linestyle='-', lw=1.5)
plt.plot(x, dy2, color='k', linestyle='--', lw=1.5)


# redmagic low
plt.subplot(343)
set_basic('gg', yticks=False)

plt.title(r'$z=0.62$', fontsize=12)

dy1 = D1[120:132]
dy2 = D2[120:132]

plt.plot(x, dy1, color='midnightblue', linestyle='-', lw=1.5)
plt.plot(x, dy2, color='k', linestyle='--', lw=1.5)



plt.subplot(344)
set_basic('gg', yticks=False)

plt.title(r'$z=1.0$', fontsize=12)
dy1 = D1[132:144]
dy2 = D2[132:144]

plt.plot(x, dy1, color='midnightblue', linestyle='-', lw=1.5, label=label1)
plt.plot(x, dy2, color='k', linestyle='--', lw=1.5, label=label2)
plt.legend(loc='upper right', fontsize=10)


plt.subplot(345)
set_basic('gp', yticks=True)
plt.ylabel(r'$\sigma ( w_{g+} )$', fontsize=fontsize)

dy1 = D1[0:12]
dy2 = D2[0:12]

plt.plot(x, dy1, color='midnightblue', linestyle='-', lw=1.5)
plt.plot(x, dy2, color='k', linestyle='--', lw=1.5)


plt.subplot(346)

set_basic('gp', yticks=False)

dy1 = D1[12:24]
dy2 = D2[12:24]

plt.plot(x, dy1, color='midnightblue', linestyle='-', lw=1.5)
plt.plot(x, dy2, color='k', linestyle='--', lw=1.5)


plt.subplot(347)

set_basic('gp', yticks=False)

dy1 = D1[24:36]
dy2 = D2[24:36]

plt.plot(x, dy1, color='midnightblue', linestyle='-', lw=1.5)
plt.plot(x, dy2, color='k', linestyle='--', lw=1.5)


plt.subplot(348)

set_basic('gp', yticks=False)

dy1 = D1[36:48]
dy2 = D2[36:48]

plt.plot(x, dy1, color='midnightblue', linestyle='-', lw=1.5)
plt.plot(x, dy2, color='k', linestyle='--', lw=1.5)




plt.subplot(349)


set_basic('pp', yticks=True)
plt.ylabel(r'$\sigma ( w_{++} )$', fontsize=fontsize)

dy1 = D1[48:60]
dy2 = D2[48:60]

plt.plot(x, dy1, color='midnightblue', linestyle='-', lw=1.5)
plt.plot(x, dy2, color='k', linestyle='--', lw=1.5)

plt.xlabel(r'$r_{\rm p}$ / $h^{-1}$ Mpc', fontsize=fontsize)


plt.subplot(3,4,10)
set_basic('pp', yticks=False)

dy1 = D1[60:72]
dy2 = D2[60:72]

plt.plot(x, dy1, color='midnightblue', linestyle='-', lw=1.5)
plt.plot(x, dy2, color='k', linestyle='--', lw=1.5)

plt.xlabel(r'$r_{\rm p}$ / $h^{-1}$ Mpc', fontsize=fontsize)



plt.subplot(3,4,11)
set_basic('pp', yticks=False)

dy1 = D1[72:84]
dy2 = D2[72:84]

plt.plot(x, dy1, color='midnightblue', linestyle='-', lw=1.5)
plt.plot(x, dy2, color='k', linestyle='--', lw=1.5)

plt.xlabel(r'$r_{\rm p}$ / $h^{-1}$ Mpc', fontsize=fontsize)




plt.subplot(3,4,12)
set_basic('pp', yticks=False)

dy1 = D1[84:96]
dy2 = D2[84:96]

plt.plot(x, dy1, color='midnightblue', linestyle='-', lw=1.5)
plt.plot(x, dy2, color='k', linestyle='--', lw=1.5)

plt.xlabel(r'$r_{\rm p}$ / $h^{-1}$ Mpc', fontsize=fontsize)



plt.subplots_adjust(bottom=0.14,left=0.14, wspace=0.0, hspace=0.15, right=0.98)
plt.savefig('jk_analytic_compare_panels.pdf')
plt.savefig('jk_analytic_compare_panels.png')
