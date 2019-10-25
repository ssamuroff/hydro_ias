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


nlevel = 6
xlim = (-25,-15)
ylim = (-0.05,1.6)


sim = sys.argv[1]
snapshot = int(sys.argv[2])

if 'TNG' in sim:
	m0 = m[snapshot]
	c0 = c[snapshot]

path3 = 'data/cats/fits/%s_%d_non-reduced_galaxy_shapes.fits'%(sim,snapshot)

f3=fi.FITS(path3)[1].read()

gi = f3['gmag'] - f3['imag']
ri = f3['rmag'] - f3['imag']

if 'TNG' in sim:
	rmask = gi > (f3['rmag']*m0 + c0)
	bmask = np.invert(rmask)


counts,xbins,ybins, = np.histogram2d(f3['rmag'], gi, bins=60, range=[xlim,ylim], normed=1 )
y = (ybins[:-1]+ybins[1:])/2
x = (xbins[:-1]+xbins[1:])/2
xx,yy=np.meshgrid(x,y)

C = plt.contour(xx,yy,counts.T, nlevel, colors='purple', linewidth=.5)

plt.xlabel("$r-$band Magnitude", fontsize=16)
plt.ylabel("$g-i$", fontsize=16)
plt.xlim(xbins.min(),xbins.max())
plt.ylim(ybins.min(),ybins.max())


if 'TNG' in sim:
	xl = np.linspace(xlim[0],xlim[1],100)
	lin = m0*xl + c0
	plt.plot(xl,lin, color="forestgreen", ls="--", lw=2.5)


plt.subplots_adjust(bottom=0.14,left=0.14,right=0.95,top=0.95)
plt.savefig('%s_gi_rmag_%d.pdf'%(sim, snapshot))
plt.close()

counts,xbins,ybins, = np.histogram2d(f3['rmag'], ri, bins=60, range=[(-25,-18),(-0.05,1.1)], normed=1 )
y = (ybins[:-1]+ybins[1:])/2
x = (xbins[:-1]+xbins[1:])/2
xx,yy=np.meshgrid(x,y)

#C = plt.contour(xx,yy,counts.T, nlevel, colors='purple', linewidth=.5)


if 'TNG' in sim:
	counts,xbins,ybins, = np.histogram2d(f3['rmag'][rmask], ri[rmask], bins=60, range=[(-25,-18),(-0.05,1.1)], normed=1 )
	y = (ybins[:-1]+ybins[1:])/2
	x = (xbins[:-1]+xbins[1:])/2
	xx,yy=np.meshgrid(x,y)

	C = plt.contour(xx,yy,counts.T, nlevel, colors='darkred', linewidth=.5)


	counts,xbins,ybins, = np.histogram2d(f3['rmag'][bmask], ri[bmask], bins=60, range=[(-25,-18),(-0.05,1.1)], normed=1 )
	y = (ybins[:-1]+ybins[1:])/2
	x = (xbins[:-1]+xbins[1:])/2
	xx,yy=np.meshgrid(x,y)

	C = plt.contour(xx,yy,counts.T, nlevel, colors='royalblue', linewidth=.5)




plt.xlabel("$r-$band Magnitude", fontsize=16)
plt.ylabel("$r-i$", fontsize=16)
plt.xlim(xbins.min(),xbins.max())
plt.ylim(ybins.min(),ybins.max())
plt.subplots_adjust(bottom=0.14,left=0.14,right=0.95,top=0.95)

plt.savefig('%s_ri_rmag_%d.pdf'%(sim, snapshot))
plt.close()



H,b = np.histogram(ri,bins=np.linspace(0,0.4,80))
H_b,b = np.histogram(ri[bmask],bins=np.linspace(0,0.4,80))
H_r,b = np.histogram(ri[rmask],bins=np.linspace(0,0.4,80))

x = (b[:-1]+b[1:])/2

plt.plot(x,H,color='darkmagenta', lw=1.5)
plt.fill_between(x,H_r,color='darkred',alpha=0.2)
plt.fill_between(x,H_b,color='royalblue',alpha=0.2)
plt.xlabel("$r-i$", fontsize=16)
plt.yticks(visible=False)
plt.ylim(ymin=0)
plt.subplots_adjust(bottom=0.14,left=0.14,right=0.95,top=0.95)

plt.savefig('%s_ri_hist_%d.pdf'%(sim, snapshot))
plt.close()


print('Red fraction = %d/%d = %f'%(f3['rmag'][rmask].size, f3['rmag'].size, f3['rmag'][rmask].size*1./f3['rmag'].size))
Nr = f3['rmag'][rmask].size*1.
Nb = f3['rmag'][bmask].size*1.
V = 205**3
print(Nr/V, Nb/V)
sigma_r = (np.std(f3['e1'][rmask]) + np.std(f3['e2'][rmask]) )/2
sigma_b = (np.std(f3['e1'][bmask]) + np.std(f3['e2'][bmask]) )/2
print(sigma_r, sigma_b)