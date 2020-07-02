import numpy as np
import fitsio as fi
import os
import pylab as plt
plt.switch_backend('agg')
plt.style.use('y1a1')

# files...
fname1 = '2pt_tng_fidcov_iteration4.fits'
fname2 = '2pt_mbii_w_fidcov_iteration4.fits'
outname='2pt_tng_mbii_w_fidcov_iteration4.fits'
os.system('cp %s %s'%(fname1, outname))

fout = fi.FITS(outname,'rw')
f1 = fi.FITS(fname1)
f2 = fi.FITS(fname2)

lengths = {}


# do gg,gp,pp in turn
for cname in ['wgg','wgp','wpp']:
	print(cname)
	w1 = f1[cname].read()
	w2 = f2[cname].read()

	di = (1+max(w1['BIN']))
	w2['BIN']+=di

	w = {}
	w['BIN'] = np.hstack((w1['BIN'],w2['BIN']))
	w['VALUE'] = np.hstack((w1['VALUE'],w2['VALUE']))
	w['SEP'] = np.hstack((w1['SEP'],w2['SEP']))

	fout[cname].write(w)

	lengths[cname] = (len(w1),len(w2))

# also the covariance matrix
# assume the simulations are totally independent datasets
# (because they hopefully are...)

C1=f1['COVMAT'][:,:]
C2=f2['COVMAT'][:,:]

n1 = C1.shape[0]
n2 = C2.shape[0]

C = np.zeros((n1+n2,n1+n2))
print(C.shape)
print(C1.shape)
print(C2.shape)

lgp1 = lengths['wgp'][0]
lgp2 = lengths['wgp'][1]
C[:lgp1,:lgp1] = C1[:lgp1,:lgp1]
C[lgp1:(lgp1+lgp2),lgp1:(lgp1+lgp2)] = C2[:lgp2,:lgp2]

s0 = (lgp1+lgp2)

lpp1 = lengths['wpp'][0]
lpp2 = lengths['wpp'][1]
C[s0:s0+lpp1,s0:s0+lpp1] = C1[lgp1:lgp1+lpp1,lgp1:lgp1+lpp1]
C[(s0+lpp1):(s0+lpp1+lpp2),(s0+lpp1):(s0+lpp1+lpp2)] = C2[lgp2:lgp2+lpp2,lgp2:lgp2+lpp2]

s0 = (s0+lpp1+lpp2)

lgg1 = lengths['wgg'][0]
lgg2 = lengths['wgg'][1]
C[s0:s0+lgg1,s0:s0+lgg1] = C1[lgp1+lpp1:lgp1+lpp1+lgg1,lgp1+lpp1:lgp1+lpp1+lgg1]
C[(s0+lgg1):(s0+lgg1+lgg2),(s0+lgg1):(s0+lgg1+lgg2)] = C2[lgp2+lpp2:lgp2+lpp2+lgg2,lgp2+lpp2:lgp2+lpp2+lgg2]

#Sorry. This is fucking awful. Sorry future me.

plt.matshow(np.log10(C),cmap='seismic')
plt.savefig('toohot.png')

fout['COVMAT'].write(C)
fout.close()
print('Done')