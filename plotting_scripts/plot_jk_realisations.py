import numpy as np
import sys
import fitsio as fi
import glob
import scipy.optimize as opt
import pylab as plt
plt.switch_backend('agg')
plt.style.use('y1a1')


path = sys.argv[1]
corr = sys.argv[2]
snapshot = int(sys.argv[3])

print('correlation type : %s'%corr)

files = glob.glob('%s/%s_%d-jk*txt'%(path,corr,snapshot))
print('Found %d files'%len(files))

dv = np.array([np.loadtxt(f).T[1] for f in files])
x = np.loadtxt(files[0]).T[0]

dv_mean = np.mean(dv, axis=0)

plt.subplot(211)
plt.title('Snapshot %d'%snapshot, fontsize=16)
for d in dv:
	plt.plot(x, d, color='purple', alpha=0.1)

plt.plot(x, dv_mean, color='k', ls=':')

plt.ylabel(r'$w_{%s}$'%(corr[1:]), fontsize=16)

plt.xscale('log')
plt.yscale('log')
plt.xticks(visible=False)
plt.xlim(0.1,70)

plt.subplot(212)
for d in dv:
	plt.plot(x, d-dv_mean, color='purple', alpha=0.1)

plt.ylabel(r'$\Delta w_{%s}$'%(corr[1:]), fontsize=16)
plt.xlabel(r'$r_{\rm p}$', fontsize=16)

plt.axhline(0, color='k', ls=':')

plt.xscale('log')
plt.xlim(0.1,70)

plt.subplots_adjust(hspace=0,wspace=0, left=0.18, bottom=0.14, right=0.98)

plt.savefig('jk_test.png')