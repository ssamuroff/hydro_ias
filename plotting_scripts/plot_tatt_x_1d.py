import numpy as np
import pylab as plt
from pylab import gca
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
c1_mbii = np.array([2.233617e+00, 2.589747e+00, 3.946869, 3.652523])
dc1_mbii = np.array([1.011978e+00, 1.236754e+00, 1.116997, 1.614252])
c2_mbii = np.array([-2.042458e+00, -8.135084e-01, -2.708013, -1.381410])
dc2_mbii = np.array([1.220132e+00, 1.712656e+00, 1.605700, 2.677612])

anla_mbii = np.array([2.020695,2.802335,3.369944,4.228199 ])
danla_mbii = np.array([4.534523e-01, 4.830144e-01, 5.749183e-01, 6.010345e-01])
  

x0 = np.array([0.00,0.3,0.625,1.0])
c1_tng = np.array([1.280880, 1.719139, 1.419545e+00, 2.411039e+00])
dc1_tng = np.array([4.643513e-01, 3.741442e-01, 4.732864e-01, 4.345985e-01])
c2_tng = np.array([-8.104039e-04, 3.099278e-01,  6.262324e-01, 4.002749e-01])
dc2_tng = np.array([7.065338e-01, 6.757297e-01, 8.363686e-01, 8.714733e-01])

anla_tng = np.array([1.620596,2.337718,2.344033,3.043243])
danla_tng = np.array([2.150888e-01,1.784646e-01,2.256165e-01,2.285881e-01])


c1_ill = np.array([5.174988e-01,4.456189e-01,8.028623e-01,1.003845e+00])
dc1_ill = np.array([4.194155e-01,5.896701e-01,3.988615e-01,4.819666e-01])
c2_ill = np.array([6.446993e-01,-5.844958e-02,9.229198e-01,8.558878e-01])
dc2_ill = np.array([8.302430e-01,1.076488e+00,8.378793e-01,9.453206e-01])

anla_ill = np.array([1.503377,8.004734e-01,2.239533,2.402437 ])
danla_ill = np.array([3.315078e-01,4.870242e-01,3.524317e-01,4.676866e-01])



s=8.0

plt.close()
plt.subplot(111)
t1 = plt.errorbar(x+0.01,c1_tng,dc1_tng,color='darkmagenta',linestyle='none', label='TNG $(A_1)$', marker='^', markersize=s)
t2 = plt.errorbar(x-0.01,c2_tng,dc2_tng,color='darkmagenta',linestyle='none', label='TNG $(A_2)$', marker='v', markersize=s)
t3 = plt.errorbar(x-0.005,anla_tng,danla_tng,color='darkmagenta',linestyle='none', label=r'TNG $(A_1, \mathrm{NLA})$', marker='*', markersize=s)
m1 = plt.errorbar(x+0.02,c1_mbii,dc1_mbii,color='midnightblue',linestyle='none', label='MBII $(A_1)$', marker='^', markersize=s)
m2 = plt.errorbar(x-0.02,c2_mbii,dc2_mbii,color='midnightblue',linestyle='none', label='MBII $(A_2)$', marker='v', markersize=s)
m3 = plt.errorbar(x+0.005,anla_mbii,danla_mbii,color='midnightblue',linestyle='none', label=r'MBII $(A_1, \mathrm{NLA})$', marker='*', markersize=s)
i1 = plt.errorbar(x+0.03,c1_ill,dc1_ill,markeredgecolor='forestgreen', ecolor='forestgreen', markerfacecolor='white',linestyle='none', label='Illustris $(A_1)$', marker='^', markersize=s)
i2 = plt.errorbar(x-0.03,c2_ill,dc2_ill,markeredgecolor='forestgreen', ecolor='forestgreen', markerfacecolor='white',linestyle='none', label='Illustris $(A_2)$', marker='v', markersize=s)
i3 = plt.errorbar(x-0.035,anla_ill,danla_ill,markeredgecolor='forestgreen', ecolor='forestgreen', markerfacecolor='white',linestyle='none', label=r'Illustris $(A_1, \mathrm{NLA})$', marker='*', markersize=s)

plt.xlim(0,1.1)
plt.xticks(visible=True, fontsize=16)
plt.yticks(visible=True, fontsize=16)
#plt.yticks([1.5,2,2.5,3.0], fontsize=16)
plt.ylim(-4,7.2)
plt.axhline(0,color='k',ls=':')
l1 = plt.legend([t1,m1,i1], ["TNG $(A_1)$", "MBII $(A_1)$", "Illustris $(A_1)$"], fontsize=12, loc='upper left')
l2 = plt.legend([t2,m2,i2], ["TNG $(A_2)$", "MBII $(A_2)$", "Illustris $(A_2)$"], fontsize=12, loc='upper center')
l3 = plt.legend([t3,m3,i3], [r"TNG $(A_1, \mathrm{NLA})$", r"MBII $(A_1, \mathrm{NLA})$", r"Illustris $(A_1, \mathrm{NLA})$"], fontsize=12, loc='upper right')
gca().add_artist(l1)
gca().add_artist(l2)

plt.ylabel(r'$A_{i}$', fontsize=20)

plt.xlabel('Redshift', fontsize=20)
plt.subplots_adjust(hspace=0,wspace=0,bottom=0.14,left=0.14, right=0.98, top=0.98)
plt.savefig('ai_redshift.pdf')
plt.savefig('ai_redshift.png')
