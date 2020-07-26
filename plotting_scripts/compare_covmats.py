import numpy as np
import sys
import fitsio as fi
import scipy.optimize as opt
import pylab as plt
plt.switch_backend('pdf')
plt.style.use('y1a1')

from matplotlib import rcParams

rcParams['xtick.major.size'] = 3.5
rcParams['xtick.minor.size'] = 1.7
rcParams['ytick.major.size'] = 3.5
rcParams['ytick.minor.size'] = 1.7
rcParams['xtick.direction']='in'
rcParams['ytick.direction']='in'

# script for plotting out the diagonals of two covariance matrices
# this will assume that the ordering is the same
# if they're not the same and you get nonsense results from the following,
# well, you were warned.

colours = ['darkmagenta','midnightblue']
ls = ['-','--']
labels=[r'\textbf{Jackknife}', r'\textbf{Analytic}']
fontsize=16

file1 = sys.argv[-1]
file2 = sys.argv[-2]

print(file1)
print(file2)


C1 = np.loadtxt(file1)
C2 = np.loadtxt(file2)

dy1 = np.sqrt(np.diag(C1))
dy2 = np.sqrt(np.diag(C2))





plt.subplot(311)

x = np.logspace(-1,np.log10(33),12)
i = len(x)
a = dy1[-i:]
b = dy2[-i:]

plt.plot(x,a,color=colours[0],ls=ls[0], label=labels[0])
plt.plot(x,b,color=colours[1],ls=ls[1], label=labels[1])

plt.xscale('log')
plt.yscale('log')
plt.xlim(0.1,200)

plt.xticks(visible=False)
plt.yticks([1e-0,1e1,1e2], visible=True, fontsize=fontsize-3)
#plt.ylim(8e-2,1e3)
plt.ylabel(r'$\sigma_{wgg}$', fontsize=fontsize)

plt.legend(loc='upper right', fontsize=10)

plt.subplot(312)

x = np.logspace(-1,np.log10(33),12)
i = len(x)
ngp = len(x)
a = dy1[:i]
b = dy2[:i]

plt.plot(x,a,color=colours[0],ls=ls[0], label=labels[0])
plt.plot(x,b,color=colours[1],ls=ls[1], label=labels[1])

plt.xscale('log')
plt.yscale('log')
plt.xlim(0.1,200)

plt.xticks(visible=False)
plt.yticks([1e-2,1e-1,1e0,1e1], visible=True, fontsize=fontsize-3)
plt.ylabel(r'$\sigma_{wg+}$', fontsize=fontsize)

plt.subplot(313)


x = np.logspace(-1,np.log10(33),12)
i = len(x)
a = dy1[ngp:ngp+i]
b = dy2[ngp:ngp+i]

plt.plot(x,a,color=colours[0],ls=ls[0], label=labels[0])
plt.plot(x,b,color=colours[1],ls=ls[1], label=labels[1])

plt.xscale('log')
plt.yscale('log')
plt.xlim(0.1,200)


plt.xticks(visible=True, fontsize=fontsize-3)
plt.yticks([1e-3,1e-2,1e-1,1e0], visible=True, fontsize=fontsize-3)
plt.ylabel(r'$\sigma_{w++}$', fontsize=fontsize)

plt.xlabel(r'$r_{\rm p}$ / $h^{-1}$ Mpc',fontsize=fontsize)

plt.subplots_adjust(bottom=0.14,left=0.14,right=0.65, hspace=0)
plt.savefig('covmat_diagonals.pdf')
plt.close()








plt.subplot(311)

x = np.logspace(-1,np.log10(33),12)
i = len(x)
a = dy1[-i:]
b = dy2[-i:]

plt.plot(x,(b-a)/a,color=colours[0],ls=ls[0])


plt.xscale('log')
plt.yscale('linear')
plt.xlim(0.1,200)
plt.axhline(0,color='k',ls=':')

plt.xticks(visible=False)
#plt.yticks([1e-1,1e1,1e3], visible=True, fontsize=fontsize-3)
#plt.ylim(8e-2,1e3)
plt.ylabel(r'$\delta \sigma_{wgg}/\sigma_{wgg}$', fontsize=fontsize)


plt.subplot(312)

x = np.logspace(-1,np.log10(33),12)
i = len(x)
ngp = len(x)
a = dy1[:i]
b = dy2[:i]

plt.plot(x,(b-a)/a,color=colours[0],ls=ls[0])

plt.xscale('log')
plt.yscale('linear')
plt.xlim(0.1,200)
plt.axhline(0,color='k',ls=':')

plt.xticks(visible=False)
#plt.yticks([1e-2,1e-1,1e0,1e1], visible=True, fontsize=fontsize-3)
plt.ylabel(r'$\delta \sigma_{wg+}/\sigma_{wg+}$', fontsize=fontsize)

plt.subplot(313)


x = np.logspace(-1,np.log10(33),12)
i = len(x)
a = dy1[ngp:ngp+i]
b = dy2[ngp:ngp+i]

plt.plot(x,(b-a)/a,color=colours[0],ls=ls[0])

plt.xscale('log')
plt.yscale('linear')
plt.xlim(0.1,200)
plt.axhline(0,color='k',ls=':')


plt.xticks(visible=True, fontsize=fontsize-3)
#plt.yticks([1e-3,1e-2,1e-1,1e0], visible=True, fontsize=fontsize-3)
plt.ylabel(r'$\delta \sigma_{w++}/\sigma_{w++}$', fontsize=fontsize)

plt.xlabel(r'$r_{\rm p}$ / $h^{-1}$ Mpc',fontsize=fontsize)

plt.subplots_adjust(bottom=0.14,left=0.14,right=0.65, hspace=0)
plt.savefig('covmat_diagonals_fracres.pdf')
plt.close()








a1,b1 = np.meshgrid(dy1,dy1)
a2,b2 = np.meshgrid(dy2,dy2)

corr1 = C1/a1/b1
corr2 = C2/a2/b2

S = np.zeros_like(corr1)
S+=corr1

i = np.arange(0,len(C1[0]),1)
xx,yy = np.meshgrid(i,i)
mask = xx>yy

S[mask] = corr2[mask]

plt.imshow(S,cmap='seismic',interpolation='none', origin='lower')
plt.clim(-1,1)
plt.colorbar()

plt.annotate(labels[0], xy=(2,133), fontsize=fontsize)
plt.annotate(labels[1], xy=(104,4), fontsize=fontsize)



plt.savefig('corrmat.pdf')
plt.savefig('corrmat.png')



plt.switch_backend('agg')
a1,b1 = np.meshgrid(dy1,dy1)
a2,b2 = np.meshgrid(dy2,dy2)

corr1 = C1/a1/b1
X1 = np.hstack((corr1[:12,:12], corr1[:12,48:60], corr1[:12,96:108]))
X2 = np.hstack((corr1[48:60,:12], corr1[48:60,48:60], corr1[48:60,96:108]))
X3 = np.hstack((corr1[96:108,:12], corr1[96:108,48:60], corr1[96:108,96:108]))
corr1 = np.vstack((X1,X2,X3))

corr2 = C2/a2/b2
X1 = np.hstack((corr2[:12,:12], corr2[:12,48:60], corr2[:12,96:108]))
X2 = np.hstack((corr2[48:60,:12], corr2[48:60,48:60], corr2[48:60,96:108]))
X3 = np.hstack((corr2[96:108,:12], corr2[96:108,48:60], corr2[96:108,96:108]))
corr2 = np.vstack((X1,X2,X3))

S = np.zeros_like(corr1)
S+=corr1

print(corr1.shape)
i = np.arange(0,len(corr1[0]),1)
xx,yy = np.meshgrid(i,i)
mask = xx>yy

plt.close()
plt.subplot(111,aspect='equal')
S[mask] = corr2[mask]

plt.pcolor(S,cmap='seismic')
plt.clim(-1,1)
plt.colorbar()

plt.yticks([0,10,20,30], fontsize=fontsize)
plt.xticks([0,10,20,30], fontsize=fontsize)

plt.annotate(labels[0], xy=(2,33.5), fontsize=fontsize)
plt.annotate(labels[1], xy=(26,1), fontsize=fontsize)
#plt.annotate(r'', fontsize=fontsize, xy=(6,-1))

bbox_props = dict(fc="none", ec="none", lw=2)
plt.annotate(r'$w_{g+}$', xy=(37, 99), fontsize=fontsize+2, xycoords='figure pixels', horizontalalignment='left', verticalalignment='top', bbox=bbox_props)
plt.annotate(r'$w_{++}$', xy=(37, 178), fontsize=fontsize+2, xycoords='figure pixels', horizontalalignment='left', verticalalignment='top', bbox=bbox_props)
plt.annotate(r'$w_{gg}$', xy=(37, 270), fontsize=fontsize+2, xycoords='figure pixels', horizontalalignment='left', verticalalignment='top', bbox=bbox_props)


plt.subplots_adjust(bottom=0.14,left=0.14)
plt.savefig('corrmat_z0.pdf')
plt.savefig('corrmat_z0.png')





