import pylab as plt
plt.switch_backend('agg')
plt.style.use('y1a1')

import numpy as np
import numpy.linalg as npl
import numpy.fft as npf
from scipy.fftpack import ifftn
from scipy.fftpack import fftn
from scipy.fftpack import fftfreq

from scipy.ndimage import gaussian_filter
from numpy.core.records import fromarrays
import fitsio as fi
import sys

# value to give the point mass
N = 1000
L = 205


# box dimension, in pixels
R = int(sys.argv[-1])

D = np.zeros((R,R,R))
tidal_tensor = np.zeros((R,R,R,3,3),dtype=np.float32)
print('Box dimension: %d pixels'%R)

# put a point mass in the centre
x0 = R/2
D[x0,x0,x0] = N
D/=D.mean()


# fourier space overdensity
print('Doing FFT...')
fft_dens = fftn(D) 
#import pdb ; pdb.set_trace()
k  = fftfreq(R, d=L*1./R)[np.mgrid[0:R,0:R,0:R]]

for i in range(3):
	for j in range(3):

		temp = fft_dens * k[i]*k[j]/(k[0]**2 + k[1]**2 + k[2]**2)

		# subtract off the trace...
		if (i==j):
			temp -= 1./3 * fft_dens

		temp[0,0,0] = 0

		tidal_tensor[:,:,:,i,j] = ifftn(temp).real


#import pdb ; pdb.set_trace()
#yf = 2 * N /x/x/x
h0=0.68

x = np.linspace(-R/2,R/2-1,R)*(L/R)
xx,yy,zz=np.meshgrid(np.arange(-R/2,R/2,1), np.arange(-R/2,R/2,1), np.arange(-R/2,R/2,1))
y = tidal_tensor[:,R/2,R/2,0,0]
yf = 4*4.3e-9*5.9e7 * np.array([D[(xx*xx+yy*yy+zz*zz)<r*r].sum()/(4./3.*np.pi*abs(r)*r*r) for r in x])

plt.close()
plt.plot(x,y,'*',color='darkmagenta',lw=1.5)
plt.plot(x[y<0],abs(y)[y<0],'*',mec='darkmagenta', mfc='none',lw=1.5)
plt.plot(x,yf,color='pink', lw=1.5)
#plt.xscale('log')
plt.yscale('log')
plt.ylim(1e0,1e6)
plt.savefig('/home/ssamurof/tmp%d.png'%R)
np.savetxt('grid.txt',tidal_tensor)
