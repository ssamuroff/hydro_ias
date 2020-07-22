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

fontsize=16
colours = {'mbii':'midnightblue','illustris':'pink','tng':'darkmagenta'}
xmin = 8e-2
xmax = 210
dx=10**0.05

snapshot=99

dat = np.genfromtxt('/home/rmandelb.proj/ssamurof/Illustris_Shapes/data/value_added_catalogs/TNG300-1_%d_vagc.dat'%snapshot, names=True)


mask=(dat['gal_id']==dat['central_id'])

Rc=dat['offset'][mask]
Rs=dat['offset'][np.invert(mask)]



Hc,binc=np.histogram(Rc,bins=np.logspace(-2, np.log10(5000), 300),normed=1)
Hs,bins=np.histogram(Rs,bins=np.logspace(-2, np.log10(5000), 300),normed=1)

x = (bins[1:]+bins[:-1])/2

plt.plot(x,0.3*Hc,color='darkmagenta',lw=1.5, label='Centrals')
plt.fill_between(x,0.3*Hc,color='darkmagenta',alpha=0.2)
plt.plot(x,Hs*8,color='plum',lw=1.5, label='Satellites')
plt.fill_between(x,Hs*8,color='plum',alpha=0.2)



plt.xlabel('Offset from halo centre / $h^{-1}$ kpc', fontsize=fontsize)
plt.ylabel('$p(R)$', fontsize=fontsize)
plt.xticks(visible=True,fontsize=fontsize)
plt.yticks(visible=True,fontsize=fontsize)
plt.legend()
plt.ylim(ymin=0)

plt.xlim(3e-1,3000)
plt.xscale('log')

plt.subplots_adjust(bottom=0.155,left=0.155, hspace=0, wspace=0,top=0.95)
plt.savefig('TNG300-1_%d_offsets.pdf'%snapshot)


