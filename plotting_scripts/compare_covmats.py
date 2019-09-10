import numpy as np
import sys
import fitsio as fi
import scipy.optimize as opt
import pylab as plt
plt.switch_backend('agg')
plt.style.use('y1a1')

# script for plotting out the diagonals of two covariance matrices
# this will assume that the ordering is the same
# if they're not the same and you get nonsense results from the following,
# well, you were warned.

colours = ['darkmagenta','midnightblue']
ls = ['-','--']
labels=['Jackknife', 'Analytic']
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

plt.annotate(labels[0], xy=(2,131), fontsize=fontsize)
plt.annotate(labels[1], xy=(108,2), fontsize=fontsize)



plt.savefig('corrmat.pdf')
