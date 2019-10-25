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


m = {99:0.045, 78: 0.045, 62: 0.055, 50: 0.022}
c = {99:1.84, 78: 1.84,  62: 1.95, 50: 1.19}

redshifts = {99:0, 78:0.3,62:0.62, 50:1.0 }

nlevel = 4
xlim = (-25,-18)
ylim = (-0.05,1.6)


sim = sys.argv[1]

for i,snapshot in enumerate([99,78,62,50]):
	m0 = m[snapshot]
	c0 = c[snapshot]

	path3 = 'data/cats/fits/%s_%d_non-reduced_galaxy_shapes.fits'%(sim,snapshot)

	f3=fi.FITS(path3)[1].read()

	gi = f3['gmag'] - f3['imag']
	ri = f3['rmag'] - f3['imag']

	rmask = gi > (f3['rmag']*m0 + c0)
	bmask = np.invert(rmask)

	plt.subplot(2,2,i+1)

	subset = np.random.choice(len(gi[rmask]), size=8000, replace=False)
	plt.scatter(f3['rmag'][rmask][subset], gi[rmask][subset], marker='.', alpha=0.08, color='darkred', s=2.0)

	counts,xbins,ybins, = np.histogram2d(f3['rmag'][rmask], gi[rmask], bins=60, range=[(-25,-18),(-0.05,1.1)], normed=1 )
	y = (ybins[:-1]+ybins[1:])/2
	x = (xbins[:-1]+xbins[1:])/2
	xx,yy=np.meshgrid(x,y)
	C = plt.contour(xx,yy,counts.T, nlevel, colors='darkred', linewidth=.5)

	subset = np.random.choice(len(gi[bmask]), size=8000, replace=False)
	plt.scatter(f3['rmag'][bmask][subset], gi[bmask][subset], marker='.', alpha=0.08, color='royalblue', s=2.0)

	counts,xbins,ybins, = np.histogram2d(f3['rmag'][bmask], gi[bmask], bins=60, range=[(-25,-18),(-0.05,1.1)], normed=1 )
	y = (ybins[:-1]+ybins[1:])/2
	x = (xbins[:-1]+xbins[1:])/2
	xx,yy=np.meshgrid(x,y)
	C = plt.contour(xx,yy,counts.T, nlevel, colors='royalblue', linewidth=.5)

	plt.xlim(-24,-18.5)
	plt.ylim(0,1.4)

	if i+1 in [3,4]:
		plt.xlabel("$r-$band Magnitude", fontsize=16)
		xticks=True
	else:
		xticks=False
		
	plt.xticks([-24,-22,-20],visible=xticks, fontsize=16)

	if i+1 in [1,3]:
		plt.ylabel("$g-i$", fontsize=16)
		yticks=True
	else:
		yticks=False
	plt.yticks([0,0.4,0.8,1.2],visible=yticks, fontsize=16)

	z = redshifts[snapshot]
	plt.annotate('$z=%1.2f$'%z, fontsize=16, xy=(-23.75, 1.21))


plt.subplots_adjust(bottom=0.14,left=0.14,right=0.95,top=0.95, hspace=0, wspace=0)

plt.savefig('gi_rmag_panels.pdf')
plt.close()