import numpy as np
from scipy.interpolate import interp1d
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

def get_x(b):
	return (b[:-1]+b[1:])/2


x_tng,w_tng=np.loadtxt('tng/2pt/txt/wgp_99.txt').T
x_ill,w_ill=np.loadtxt('illustris/2pt/txt/wgp_135.txt').T
x_mb2,w_mb2=np.loadtxt('mbii/2pt/txt/wgp_85.txt').T

I_tng = interp1d(np.log10(x_tng), w_tng)
I_mb2 = interp1d(np.log10(x_mb2), w_mb2)

y_tng = I_tng(np.log10(x_ill))
y_mb2 = I_mb2(np.log10(x_ill))
y_ill = w_ill

plt.subplot(211)

plt.plot(x_ill, y_tng, color='purple', label='TNG')
plt.plot(x_ill, y_mb2, color='k', label='MBII')
plt.plot(x_ill, y_ill, color='forestgreen', label='Ill')
plt.xscale('log')
plt.yscale('log')
plt.ylabel(r"$w_{g+}$")
plt.legend()



R1 = y_ill/y_mb2
R2 = y_ill/y_tng
R3 = y_mb2/y_tng

plt.subplot(212)
plt.plot(x_ill, R1, color='purple', label='Ill/MBII')
plt.plot(x_ill, R2, color='k', label='Ill/TNG')
plt.plot(x_ill, R3, color='forestgreen', label='MBII/TNG')
plt.xscale('log')
#plt.ylim(-1,1)
plt.ylabel(r"$w_{g+}/w^{'}_{g+}$")
plt.xlabel(r"$r_{\rm p}$ / $h^{-1}$ Mpc")
plt.legend()
plt.subplots_adjust(bottom=0.14,left=0.14,hspace=0,wspace=0)
plt.savefig('wgp_sim_ratios.png')
plt.close()


plt.plot(x_ill, y_tng, color='purple', label='TNG')
plt.plot(x_ill, y_mb2, color='k', label='MBII')
plt.plot(x_ill, y_ill, color='forestgreen', label='Ill')
plt.xscale('log')
plt.ylabel(r"$w_{g+}$")

plt.legend()
plt.subplots_adjust(bottom=0.14,left=0.14,hspace=0,wspace=0)
plt.savefig('wgp_lin_sim.png')
