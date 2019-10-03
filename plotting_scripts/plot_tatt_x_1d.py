import numpy as np
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



x = np.array([0.06,0.3,0.625,1.0])
c1_mbii = np.array([8.250003e-01, 5.419721e-01, 1.183762e+00,1.578299e+00])
dc1_mbii = np.array([4.424263e-01, 6.449061e-01, 4.431620e-01,3.906427e-01])
c2_mbii = np.array([9.955508e-01, 4.638071e+00, 7.108131e-01,1.053010e+00])/5.
dc2_mbii = np.array([4.770114e+00, 7.028168e+00, 4.794870e+00,4.251943e+00])/5.

x0 = np.array([0.00,0.3,0.625,1.0])
c1_tng = np.array([5.687184e-01, 5.371325e-01, 8.156881e-01, 7.746648e-01])
dc1_tng = np.array([9.912863e-02, 1.081440e-01, 1.301491e-01, 9.273534e-02 ])
c2_tng = np.array([5.135676e-01, 7.497391e-01, 4.337997e-01, -3.345032e-02])
dc2_tng = np.array([2.130554e-01, 2.102555e-01, 2.793867e-01, 2.091460e-01])

c1_ill = np.array([2.445613e-01, 7.898489e-01, 3.531482e-01, -1.599661e-01 ])
dc1_ill = np.array([1.491332e-01,  2.050607e-01, 1.567092e-01, 1.805179e-01 ])
c2_ill = np.array([1.615803e-01, 4.404832e-01, 1.733408e-01, 6.418761e-01])
dc2_ill = np.array([3.035423e-01, 4.080502e-01, 2.957249e-01, 2.858957e-01])



plt.close()
plt.subplot(111)
plt.errorbar(x+0.01,c1_tng,dc1_tng,color='darkmagenta',linestyle='none', label='IllustrisTNG $(A_1)$', marker='^')
plt.errorbar(x-0.01,c2_tng,dc2_tng,color='darkmagenta',linestyle='none', label='IllustrisTNG $(A_2)$', marker='v')
plt.errorbar(x+0.02,c1_mbii,dc1_mbii,color='midnightblue',linestyle='none', label='MBII $(A_1)$', marker='^')
plt.errorbar(x-0.02,c2_mbii,dc2_mbii,color='midnightblue',linestyle='none', label='MBII $(A_2)$', marker='v')
plt.errorbar(x+0.03,c1_ill,dc1_ill,color='forestgreen',linestyle='none', label='Illustris $(A_1)$', marker='^')
plt.errorbar(x-0.03,c2_ill,dc2_ill,color='forestgreen',linestyle='none', label='Illustris $(A_2)$', marker='v')

plt.xlim(0,1.1)
plt.xticks(visible=True)
#plt.yticks([1.5,2,2.5,3.0], fontsize=16)
plt.ylim(-0.9,4)
plt.axhline(0,color='k',ls=':')
plt.legend(fontsize=12, loc='upper left')
plt.ylabel(r'$A_{i}$', fontsize=16)

plt.xlabel('Redshift $z$', fontsize=16)
plt.subplots_adjust(hspace=0,wspace=0,bottom=0.14,left=0.14)
plt.savefig('ai_redshift.pdf')
plt.savefig('ai_redshift.png')