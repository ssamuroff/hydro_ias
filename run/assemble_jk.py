import argparse
import numpy as np
import sys
import os
import glob

import pylab as plt
plt.switch_backend('pdf')
plt.style.use('y1a1')


files = sys.argv[1:]

correlations = ['wgp','wpp', 'wgg']
snapshots = [85, 79, 73, 68]

base = os.path.dirname(files[0])

ijk = np.unique([int(f.split('-jk')[1].replace('.txt','')) for f in files])
print('%d jackknife patches'%len(ijk))

dv = []
x = np.logspace(-1,np.log10(33),12)
for i in ijk:
	d = []
	for c in correlations:
		d.append(np.concatenate([np.loadtxt(base+'/%s_%d-jk%d.txt'%(c,s,i)).T[1] for s in snapshots]))
	dv.append(np.concatenate(d))

for i in ijk:
	plt.plot(x, dv[i][:len(x)], color='purple', alpha=0.1)

plt.yscale('log')
plt.xscale('log')
plt.savefig('jktest.png')
plt.close()
dv = np.array(dv)
#dv[13,:]=dv[0,:]
#import pdb ; pdb.set_trace()

nreal = dv.shape[0]
wmean = np.mean(dv,axis=0)
ndim = len(wmean)
cov = np.zeros((ndim,ndim))

for row in range(ndim):
    for col in range(ndim):
        D1 = np.array((dv[:,row] - wmean[row])).T
        D2 = np.array((dv[:,col] - wmean[col])).T
        
        cov[row,col] = np.sum(D1*D2,axis=0) #/np.sqrt(len(D1))


K = (nreal*1.-1.)/nreal
cov*=K

a0,b0 = np.meshgrid(np.sqrt(np.diag(cov)),np.sqrt(np.diag(cov)))
corr = cov/a0/b0

plt.matshow(corr,cmap='seismic')
plt.clim(-1,1)
plt.colorbar()
plt.savefig('jk_corrmat.pdf')
plt.savefig('jk_corrmat.png')



ev = np.linalg.eigvalsh(cov)
print('Eigenvalues: ', ev)


print('Determinant: ', np.linalg.slogdet(cov))

print('matrix is invertible?')
try:
	cov_inv = np.linalg.inv(cov)
	print('yes')
except:
	print('no')

print('Shape:',cov.shape)
np.savetxt('jk_covmat.txt', cov)
